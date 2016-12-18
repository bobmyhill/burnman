from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))


import burnman
from burnman.minerals import SLB_2011
from scipy.optimize import fsolve

from scipy.integrate import odeint
from scipy.interpolate import UnivariateSpline

# Declare the solid solutions which we are interested in using
ol = SLB_2011.ol()
wa = SLB_2011.wa()
ri = SLB_2011.ri()

mw = SLB_2011.mw()
pv = SLB_2011.pv()
ppv = SLB_2011.ppv()



# USER INPUT
Tp = 1600.
mg_number = 90.
n_points = 1001


# Find the entropy at the potential temperature of interest
ol.set_composition([mg_number/100., 1. - mg_number/100.])
ol.set_state(1.e5, Tp)
entropy = ol.S

# Define all the functions required to find the equilibrium minerals
def one_phase(args, S, m, P):
    T = args[0]
    m.set_state(P, T)
    return m.S - S

def two_phase(args, S, m1, m2, P, mg_number):
    T, p_m1, x_fe_m1 = args
    p_m2 = 1. - p_m1
    x_fe_m2 = ((1. - mg_number/100.) - x_fe_m1*p_m1)/p_m2
    m1.set_composition([1. - x_fe_m1, x_fe_m1])
    m2.set_composition([1. - x_fe_m2, x_fe_m2])
    m1.set_state(P, T)
    m2.set_state(P, T)

    return [(m1.S*p_m1 + m2.S*p_m2) - S,
            m1.partial_gibbs[0] - m2.partial_gibbs[0],
            m1.partial_gibbs[1] - m2.partial_gibbs[1]]

def m2_in(args, S, m1, m2, mg_number):
    P, T, x_fe_m2 = args

    x_fe_m1 = 1. - mg_number/100.
    m1.set_composition([1. - x_fe_m1, x_fe_m1])
    m2.set_composition([1. - x_fe_m2, x_fe_m2])
    m1.set_state(P, T)
    m2.set_state(P, T)
    return [m1.S - S,
            (m1.partial_gibbs[0] - m2.partial_gibbs[0]),
            (m1.partial_gibbs[1] - m2.partial_gibbs[1])]


def ri_pv_per_edge1(args, S, mg_number):
    P, T, x_fe_ri, x_fe_mw = args
    x_fe_total = 1. - mg_number/100.
    x_fe_pv = x_fe_total + (x_fe_total - x_fe_mw)
    ri.set_composition([1. - x_fe_ri, x_fe_ri])
    mw.set_composition([1. - x_fe_mw, x_fe_mw])
    pv.set_composition([1. - x_fe_pv, x_fe_pv, 0.])
    ri.set_state(P, T)
    mw.set_state(P, T)
    pv.set_state(P, T)

    return [ri.S - S,
            ri.partial_gibbs[0] - pv.partial_gibbs[0] - mw.partial_gibbs[0],
            ri.partial_gibbs[1] - pv.partial_gibbs[1] - mw.partial_gibbs[1],
            ((pv.partial_gibbs[1] - pv.partial_gibbs[0]) - 
             (mw.partial_gibbs[1] - mw.partial_gibbs[0]))]

def ri_pv_per_edge2(args, S, mg_number):
    P, T, x_fe_ri, x_fe_mw = args
    x_fe_total = 1. - mg_number/100.
    x_fe_pv = x_fe_total + (x_fe_total - x_fe_mw)
    ri.set_composition([1. - x_fe_ri, x_fe_ri])
    mw.set_composition([1. - x_fe_mw, x_fe_mw])
    pv.set_composition([1. - x_fe_pv, x_fe_pv, 0.])
    ri.set_state(P, T)
    mw.set_state(P, T)
    pv.set_state(P, T)

    return [(mw.S + pv.S) - S,
            ri.partial_gibbs[0] - pv.partial_gibbs[0] - mw.partial_gibbs[0],
            ri.partial_gibbs[1] - pv.partial_gibbs[1] - mw.partial_gibbs[1],
            ((pv.partial_gibbs[1] - pv.partial_gibbs[0]) - 
             (mw.partial_gibbs[1] - mw.partial_gibbs[0]))]


def ri_pv_mw(args, S, mg_number):
    T, p_ri, x_fe_ri, x_fe_mw = args
    x_fe_total = 1. - mg_number/100.
    x_fe_pv = (x_fe_total - x_fe_ri*p_ri - (1. - p_ri)*x_fe_mw)/(1. - p_ri)
    ri.set_composition([1. - x_fe_ri, x_fe_ri])
    mw.set_composition([1. - x_fe_mw, x_fe_mw])
    pv.set_composition([1. - x_fe_pv, x_fe_pv, 0.])
    ri.set_state(P, T)
    mw.set_state(P, T)
    pv.set_state(P, T)
    
    return [ri.S*p_ri + (1. - p_ri)*(pv.S + mw.S) - S,
            ri.partial_gibbs[0] - pv.partial_gibbs[0] - mw.partial_gibbs[0],
            ri.partial_gibbs[1] - pv.partial_gibbs[1] - mw.partial_gibbs[1],
            ((pv.partial_gibbs[1] - pv.partial_gibbs[0]) - 
             (mw.partial_gibbs[1] - mw.partial_gibbs[0]))]


def pv_per(args, S, m1, m2, mg_number):
    T, x_fe_mw = args
    x_fe_total = 1. - mg_number/100.
    x_fe_pv = x_fe_total + (x_fe_total - x_fe_mw)
    m1.set_composition([1. - x_fe_pv, x_fe_pv, 0.])
    m2.set_composition([1. - x_fe_mw, x_fe_mw])
    m1.set_state(P, T)
    m2.set_state(P, T)

    return [(m1.S + m2.S) - S,
            ((m1.partial_gibbs[1] - m1.partial_gibbs[0]) - 
             (m2.partial_gibbs[1] - m2.partial_gibbs[0]))]


def pv_ppv_per_edge(args, S, m1, m2, mg_number):
    P, T, x_fe_m1, x_fe_m2 = args
    x_fe_total = 1. - mg_number/100.
    x_fe_mw = x_fe_total + (x_fe_total - x_fe_m1)
    m1.set_composition([1. - x_fe_m1, x_fe_m1, 0.])
    m2.set_composition([1. - x_fe_m2, x_fe_m2, 0.])
    mw.set_composition([1. - x_fe_mw, x_fe_mw])
    m1.set_state(P, T)
    m2.set_state(P, T)
    mw.set_state(P, T)

    return [(m1.S + mw.S) - S,
            m2.partial_gibbs[0] - m1.partial_gibbs[0],
            m2.partial_gibbs[1] - m1.partial_gibbs[1],
            ((m2.partial_gibbs[1] - m2.partial_gibbs[0]) - 
             (mw.partial_gibbs[1] - mw.partial_gibbs[0]))]

def pv_ppv_per(args, S, mg_number):
    T, p_pv, x_fe_pv, x_fe_ppv = args
    x_fe_total = 1. - mg_number/100.
    x_fe_mw = 2.*x_fe_total - (x_fe_pv*p_pv) - (x_fe_ppv*(1. - p_pv))
    pv.set_composition([1. - x_fe_pv, x_fe_pv, 0.])
    ppv.set_composition([1. - x_fe_ppv, x_fe_ppv, 0.])
    mw.set_composition([1. - x_fe_mw, x_fe_mw])
    pv.set_state(P, T)
    ppv.set_state(P, T)
    mw.set_state(P, T)

    return [(pv.S*p_pv + ppv.S*(1. - p_pv) + mw.S) - S,
            ppv.partial_gibbs[0] - pv.partial_gibbs[0],
            ppv.partial_gibbs[1] - pv.partial_gibbs[1],
            ((pv.partial_gibbs[1] - pv.partial_gibbs[0]) - 
             (mw.partial_gibbs[1] - mw.partial_gibbs[0]))]


def PT_eqm(args, P, T, mg_number, code, m):
    if code == 'olpoly':
        p_m1, x_fe_m1 = args
        p_m2 = 1. - p_m1
        x_fe_m2 = ((1. - mg_number/100.) - x_fe_m1*p_m1)/p_m2
        m[0].set_composition([1. - x_fe_m1, x_fe_m1])
        m[1].set_composition([1. - x_fe_m2, x_fe_m2])
        m[0].set_state(P, T)
        m[1].set_state(P, T)
        
        return [m[0].partial_gibbs[0] - m[1].partial_gibbs[0],
                m[0].partial_gibbs[1] - m[1].partial_gibbs[1]]
    
    elif code == 'pvmw':
        x_fe_mw = args[0]
        x_fe_total = 1. - mg_number/100.
        x_fe_pv = x_fe_total + (x_fe_total - x_fe_mw)
        m[0].set_composition([1. - x_fe_pv, x_fe_pv, 0.])
        m[1].set_composition([1. - x_fe_mw, x_fe_mw])
        m[0].set_state(P, T)
        m[1].set_state(P, T)

        return [((m[0].partial_gibbs[1] - m[0].partial_gibbs[0]) - 
                 (m[1].partial_gibbs[1] - m[1].partial_gibbs[0]))]


    else:
        p_pv, x_fe_pv, x_fe_ppv = args
        x_fe_total = 1. - mg_number/100.
        x_fe_mw = 2.*x_fe_total - (x_fe_pv*p_pv) - (x_fe_ppv*(1. - p_pv))
        pv.set_composition([1. - x_fe_pv, x_fe_pv, 0.])
        ppv.set_composition([1. - x_fe_ppv, x_fe_ppv, 0.])
        mw.set_composition([1. - x_fe_mw, x_fe_mw])
        pv.set_state(P, T)
        ppv.set_state(P, T)
        mw.set_state(P, T)
        return [ppv.partial_gibbs[0] - pv.partial_gibbs[0],
                ppv.partial_gibbs[1] - pv.partial_gibbs[1],
                ((pv.partial_gibbs[1] - pv.partial_gibbs[0]) - 
                 (mw.partial_gibbs[1] - mw.partial_gibbs[0]))]


# Find the pressures at which we enter a different assemblage
p_wa_in = fsolve(m2_in, [18.e9, 1900., 0.1], args=(entropy, ol, wa, mg_number))[0]
p_ol_out = fsolve(m2_in, [18.e9, 1900., 0.1], args=(entropy, wa, ol, mg_number))[0]
p_ri_in = fsolve(m2_in, [18.e9, 1900., 0.1], args=(entropy, wa, ri, mg_number))[0]
p_wa_out = fsolve(m2_in, [18.e9, 1900., 0.1], args=(entropy, ri, wa, mg_number))[0]
p_pv_in = fsolve(ri_pv_per_edge1, [18.e9, 1900., 0.1, 0.1], args=(entropy, mg_number))[0]
p_ri_out = fsolve(ri_pv_per_edge2, [18.e9, 1900., 0.1, 0.1], args=(entropy, mg_number))[0]
t_pv_in = fsolve(ri_pv_per_edge1, [18.e9, 1900., 0.1, 0.1], args=(entropy, mg_number))[1]
t_ri_out = fsolve(ri_pv_per_edge2, [18.e9, 1900., 0.1, 0.1], args=(entropy, mg_number))[1]
p_ppv_in = fsolve(pv_ppv_per_edge, [100.e9, 1900., 0.1, 0.1], args=(entropy, pv, ppv, mg_number))[0]
p_pv_out = fsolve(pv_ppv_per_edge, [100.e9, 1900., 0.1, 0.1], args=(entropy, ppv, pv, mg_number))[0]


print(np.array([p_wa_in, p_ol_out, p_ri_in, p_wa_out, p_pv_in, p_ri_out, p_ppv_in, p_pv_out])/1.e9)


# Now we calculate the isentrope
pressures = np.linspace(1.e5, 130.e9, n_points)
#pressures = np.linspace(23.3e9, 23.4e9, n_points)
gravity = pressures * 0. + 10.

temperatures = np.empty_like(pressures)
volumes = np.empty_like(pressures)
densities = np.empty_like(pressures)
specific_heats = np.empty_like(pressures)
alphas = np.empty_like(pressures)
depths = np.empty_like(pressures)
compressibilities = np.empty_like(pressures)
for i, P in enumerate(pressures):
    if P < p_wa_in:
        ol.set_composition([mg_number/100., 1. - mg_number/100.])
        temperatures[i] = fsolve(one_phase, [1700.], args=(entropy, ol, P))[0]
        rock=burnman.Composite([ol], [1.])
    elif P < p_ol_out:
        sol = fsolve(two_phase, [1700., 0.01, 0.3], args=(entropy, ol, wa, P, mg_number))
        rock=burnman.Composite([ol, wa], [sol[1], (1. - sol[1])])
        temperatures[i] = sol[0]
    elif P < p_ri_in:
        wa.set_composition([mg_number/100., 1. - mg_number/100.])
        temperatures[i] = fsolve(one_phase, [1800.], args=(entropy, wa, P))[0]
        rock=burnman.Composite([wa], [1.])
    elif P < p_wa_out:
        sol = fsolve(two_phase, [1700., 0.01, 0.3], args=(entropy, wa, ri, P, mg_number))
        rock=burnman.Composite([wa, ri], [sol[1], (1. - sol[1])])
        temperatures[i] = sol[0]
    elif P < p_pv_in:
        ri.set_composition([mg_number/100., 1. - mg_number/100.])
        temperatures[i] = fsolve(one_phase, [1900.], args=(entropy, ri, P))[0]
        rock=burnman.Composite([ri], [1.])
    elif P < p_ri_out:
        sol = fsolve(ri_pv_mw, [t_pv_in, 0.01, 0.1, 0.1], args=(entropy, P))
        temperatures[i] = sol[0]
        fs = np.array([sol[1], (1. - sol[1]), (1. - sol[1])])
        rock=burnman.Composite([ri, pv, mw], fs/np.sum(fs))
    elif P < p_ppv_in:
        temperatures[i] = fsolve(pv_per, [2000., 0.1], args=(entropy, pv, mw, mg_number))[0]
        rock=burnman.Composite([pv, mw], [0.5, 0.5])
    elif P < p_pv_out:
        sol = fsolve(pv_ppv_per, [2600., 0.01, 0.1, 0.1], args=(entropy, mg_number))
        temperatures[i] = sol[0]
        rock=burnman.Composite([pv, ppv, mw], [sol[1]/2., (1. - sol[1])/2., 0.5])
    else:
        temperatures[i] = fsolve(pv_per, [2600., 0.1], args=(entropy, ppv, mw, mg_number))[0]
        rock=burnman.Composite([ppv, mw], [0.5, 0.5])

    volumes[i] = rock.V
    densities[i] = rock.rho
    specific_heats[i] = rock.heat_capacity_p/rock.rho/rock.V # J/K/mol / (kg/m^3) / (m^3/mol) 
    alphas[i] = rock.alpha
    compressibilities[i] = rock.isothermal_compressibility


# second
dS = 0.1
entropy = entropy + dS
temperatures2 = np.empty_like(pressures)
specific_heats_relaxed = np.empty_like(pressures)
alphas_relaxed = np.empty_like(pressures)
for i, P in enumerate(pressures):
    if P < p_wa_in:
        ol.set_composition([mg_number/100., 1. - mg_number/100.])
        temperatures2[i] = fsolve(one_phase, [1700.], args=(entropy, ol, P))[0]
        rock=burnman.Composite([ol], [1.])
    elif P < p_ol_out:
        sol = fsolve(two_phase, [1700., 0.01, 0.3], args=(entropy, ol, wa, P, mg_number))
        rock=burnman.Composite([ol, wa], [sol[1], (1. - sol[1])])
        temperatures2[i] = sol[0]
    elif P < p_ri_in:
        wa.set_composition([mg_number/100., 1. - mg_number/100.])
        temperatures2[i] = fsolve(one_phase, [1800.], args=(entropy, wa, P))[0]
        rock=burnman.Composite([wa], [1.])
    elif P < p_wa_out:
        sol = fsolve(two_phase, [1700., 0.01, 0.3], args=(entropy, wa, ri, P, mg_number))
        rock=burnman.Composite([wa, ri], [sol[1], (1. - sol[1])])
        temperatures2[i] = sol[0]
    elif P < p_pv_in:
        ri.set_composition([mg_number/100., 1. - mg_number/100.])
        temperatures2[i] = fsolve(one_phase, [1900.], args=(entropy, ri, P))[0]
        rock=burnman.Composite([ri], [1.])
    elif P < p_ri_out:
        sol = fsolve(ri_pv_mw, [1900., 0.01, 0.1, 0.1], args=(entropy, P))
        temperatures[i] = sol[0]
        fs = np.array([sol[1], (1. - sol[1]), (1. - sol[1])])
        rock=burnman.Composite([ri, pv, mw], fs/np.sum(fs))
    elif P < p_ppv_in:
        temperatures2[i] = fsolve(pv_per, [2000., 0.1], args=(entropy, pv, mw, mg_number))[0]
        rock=burnman.Composite([pv, mw], [0.5, 0.5])
    elif P < p_pv_out:
        sol = fsolve(pv_ppv_per, [2600., 0.01, 0.1, 0.1], args=(entropy, mg_number))
        temperatures2[i] = sol[0]
        sol3 = sol[1:]
        rock=burnman.Composite([pv, ppv, mw], [sol[1]/2., (1. - sol[1])/2., 0.5])
    else:
        temperatures2[i] = fsolve(pv_per, [2600., 0.1], args=(entropy, ppv, mw, mg_number))[0]
        rock=burnman.Composite([ppv, mw], [0.5, 0.5])

    # depth, pressure, temperature, density, gravity, Cp (per kilo), thermal expansivity
    specific_heats_relaxed[i] = temperatures[i]*dS/(temperatures2[i] - temperatures[i])/rock.rho/rock.V # J/K/mol / (kg/m^3) / (m^3/mol)
    #specific_heats_relaxed[i] = rock.heat_capacity_p/rock.rho/rock.V
    alphas_relaxed[i] = 2./(volumes[i] + rock.V) * (rock.V - volumes[i])/(temperatures2[i] - temperatures[i])


dP = 10000.
compressibilities_relaxed = np.empty_like(pressures)
for i, P in enumerate(pressures):
    if P < p_wa_in:
        rock=burnman.Composite([ol], [1.])
        ol.set_composition([mg_number/100., 1. - mg_number/100.])
        rock.set_state(P + dP, temperatures[i])
    elif P < p_ol_out:
        sol = fsolve(PT_eqm, [0.01, 0.3], args=(P+dP, temperatures[i], mg_number, 'olpoly', [ol, wa]))
        rock=burnman.Composite([ol, wa], [sol[0], (1. - sol[0])])
    elif P < p_ri_in:
        rock=burnman.Composite([wa], [1.])
        wa.set_composition([mg_number/100., 1. - mg_number/100.])
        rock.set_state(P + dP, temperatures[i])
    elif P < p_wa_out:
        sol = fsolve(PT_eqm, [0.1, 0.35], args=(P+dP, temperatures[i], mg_number, 'olpoly', [wa, ri]))
        rock=burnman.Composite([wa, ri], [sol[0], (1. - sol[0])])
    elif P < p_pv_in:
        rock=burnman.Composite([ri], [1.])
        ri.set_composition([mg_number/100., 1. - mg_number/100.])
        rock.set_state(P + dP, temperatures[i])
    elif P < p_ppv_in:
        sol = fsolve(PT_eqm, [0.1], args=(P+dP, temperatures[i], mg_number, 'pvmw', [pv, mw]))
        rock=burnman.Composite([pv, mw], [0.5, 0.5])
    elif P < p_pv_out:
        sol = fsolve(PT_eqm, sol3, args=(P+dP, temperatures[i], mg_number, 'ppvpvmw', [pv, ppv, mw]))
        rock=burnman.Composite([pv, ppv, mw], [sol[0]/2., (1. - sol[0])/2., 0.5])
    else:
        sol = fsolve(PT_eqm, [0.1], args=(P+dP, temperatures[i], mg_number, 'pvmw', [ppv,  mw]))
        rock=burnman.Composite([ppv, mw], [0.5, 0.5])
    
    compressibilities_relaxed[i] = -(rock.V - volumes[i])/volumes[i]/dP



g0 = 9.81
n_gravity_iterations = 5
for i in xrange(n_gravity_iterations):    
    # Integrate the hydrostatic equation
    # Make a spline fit of densities as a function of pressures
    rhofunc = UnivariateSpline(pressures, densities)
    # Make a spline fit of gravity as a function of depth
    gfunc = UnivariateSpline(pressures, gravity)

    # integrate the hydrostatic equation
    depths = np.ravel(odeint((lambda p, x: 1./(gfunc(x) * rhofunc(x))), 0.0, pressures))

    radii = 6371.e3 - depths

    rhofunc = UnivariateSpline(radii[::-1], densities[::-1])
    poisson = lambda p, x: 4.0 * np.pi * burnman.constants.G * rhofunc(x) * x * x
    gravity = np.ravel(odeint(poisson, g0*radii[0]*radii[0], radii))
    gravity = gravity / radii / radii


for i in xrange(1,7):
    plt.subplot(2, 4, i)
    plt.xlabel('Pressure (GPa)')
    plt.xlabel('Depths (km)')

x = pressures/1.e9
x = depths/1.e3

plt.subplot(2, 4, 1)
plt.plot(x, temperatures)
plt.ylabel('Temperature (K)')
plt.subplot(2, 4, 2)
plt.plot(x, depths/1.e3)
plt.ylabel('Depth (km)')
plt.subplot(2, 4, 3)
plt.plot(x, gravity)
plt.ylabel('Gravity (m/s^2)')
plt.subplot(2, 4, 4)
plt.plot(x, densities)
plt.ylabel('Density (kg/m^3)')
plt.subplot(2, 4, 5)
plt.plot(x, specific_heats)
plt.plot(x, specific_heats_relaxed)
plt.ylabel('Cp (J/K/kg)')
plt.subplot(2, 4, 6)
plt.plot(x, alphas)
plt.plot(x, alphas_relaxed)
plt.ylabel('alpha (/K)')
plt.subplot(2, 4, 7)
plt.plot(x, compressibilities)
plt.plot(x, compressibilities_relaxed)
plt.ylabel('compressibilities (/Pa)')
plt.show()




# depth, pressure, temperature, density, gravity, Cp (per kilo), thermal expansivity
np.savetxt('isentrope_properties.txt', X=np.array([depths, pressures, temperatures, densities, gravity, alphas, specific_heats, compressibilities]).T,
           header='POINTS: '+str(n_points)+' \ndepth (m), pressure (Pa), temperature (K), density (kg/m^3), gravity (m/s^2), thermal expansivity (/K), Cp (J/K/kg), beta (/Pa)',
           fmt='%.10e', delimiter='\t')
