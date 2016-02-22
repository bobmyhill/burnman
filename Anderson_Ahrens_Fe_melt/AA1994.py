import scipy.integrate as integrate
from scipy.optimize import fsolve
import os, sys, numpy as np, matplotlib.pyplot as plt
import matplotlib.image as mpimg

if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))
import burnman
    
# HEAT CAPACITY CONTRIBUTIONS
##############################

# High temperature limit of the kinetic contribution to the heat capacity
# Anderson and Ahrens (1994), just after equation 29.
def _C_v_kin(V, T, params):
    return 1.5*burnman.constants.gas_constant

# Equation A1
def _C_v_el(V, T, params):
    Vfrac = V/params['V_0']
    
    A = params['a'][0] + params['a'][1]*Vfrac
    B = params['b'][0] + params['b'][1]*Vfrac*Vfrac
    Theta = params['Theta'][0]*np.power(Vfrac, -params['Theta'][1])
    
    C_e = A*(1. - (Theta*Theta)/(Theta*Theta + T*T)) + B*np.power(T, 0.6)
    
    return C_e

# Equation A15
def _C_v_pot(V, T, params):
    rhofrac = params['V_0']/V
        
    xi = params['xi_0']*np.power(rhofrac, -0.6)
    F = 1./(1. + np.exp((rhofrac - params['F'][0])/params['F'][1]))
    lmda = (F*(params['lmda'][0]*rhofrac + params['lmda'][1]) + params['lmda'][2])*np.power(rhofrac, 0.4)
    C_pot = (lmda*T + xi*params['theta']) / (params['theta'] + T)

    return C_pot

# Equation 29
def C_v(V, T, params):
    return _C_v_kin(V, T, params) + _C_v_pot(V, T, params) + _C_v_el(V, T, params)

# INTERNAL ENERGY CONTRIBUTIONS
################################

def _internal_energy_kin(Ts, T, V, params):
    return 1.5*burnman.constants.gas_constant*(T - Ts)

def _internal_energy_el(Ts, T, V, params):
    Vfrac = V/params['V_0']
    
    A = params['a'][0] + params['a'][1]*Vfrac
    B = params['b'][0] + params['b'][1]*Vfrac*Vfrac
    Theta = params['Theta'][0]*np.power(Vfrac, -params['Theta'][1])

    E_el = A*(T - Ts - Theta*(np.arctan(T/Theta) - np.arctan(Ts/Theta))) + 0.625*B*(np.power(T, 1.6) - np.power(Ts, 1.6))
    return E_el

def _internal_energy_pot(Ts, T, V, params):
    rhofrac = params['V_0']/V
        
    xi = params['xi_0']*np.power(rhofrac, -0.6)
    F = 1./(1. + np.exp((rhofrac - params['F'][0])/params['F'][1]))
    lmda = (F*(params['lmda'][0]*rhofrac + params['lmda'][1]) + params['lmda'][2])*np.power(rhofrac, 0.4)
    
    return (lmda*(T - Ts) + params['theta']*(xi - lmda)*np.log((params['theta'] + T)/(params['theta'] + Ts)))

# ENTROPY CONTRIBUTIONS
########################

def _entropy_kin(Ts, T, V, params):
    if np.abs(T- Ts) > 1.e-10:
        return 1.5*burnman.constants.gas_constant*(np.log(T) - np.log(Ts))
    else:
        return 0.
    
def _entropy_el(Ts, T, V, params):
    if np.abs(T- Ts) > 1.e-10:
        Vfrac = V/params['V_0']
        
        A = params['a'][0] + params['a'][1]*Vfrac
        B = params['b'][0] + params['b'][1]*Vfrac*Vfrac
        Theta = params['Theta'][0]*np.power(Vfrac, -params['Theta'][1])
    
        return (A*(np.log(T/Ts) - 0.5*np.log(T*T*(Theta*Theta + Ts*Ts)/(Ts*Ts*(Theta*Theta + T*T)))) + 5./3.*B*(np.power(T, 0.6) - np.power(Ts, 0.6)))
    else:
        return 0.
    
def _entropy_pot(Ts, T, V, params):
    if np.abs(T- Ts) > 1.e-10:
        rhofrac = params['V_0']/V
        
        xi = params['xi_0']*np.power(rhofrac, -0.6)
        F = 1./(1. + np.exp((rhofrac - params['F'][0])/params['F'][1]))
        lmda = (F*(params['lmda'][0]*rhofrac + params['lmda'][1]) + params['lmda'][2])*np.power(rhofrac, 0.4)
        
        return (lmda*np.log((params['theta'] + T)/(params['theta'] + Ts)) + xi*np.log((T*(params['theta'] + Ts))/(Ts*(params['theta'] + T))))
    else:
        return 0.
    
def entropy(T, V, params):
    Ts = _isentropic_temperature(V, params)
    return params['S_0'] + _entropy_kin(Ts, T, V, params) + _entropy_el(Ts, T, V, params) + _entropy_pot(Ts, T, V, params)




'''
# Form for the grueneisen parameter is taken from Anderson and Ahrens (1994; Equation 15)
def _isentropic_grueneisen_over_V(V, params):
    rhofrac = params['V_0']/V
    groverV = (params['grueneisen_0'] + (params['grueneisen_prime']*np.power(rhofrac, params['grueneisen_n']) *
                                      (_isentropic_energy_change(V, params))))/V
    return groverV
    
# Temperature along an isentrope (Chen and Ahrens 1995; Equation 9)
def _isentropic_temperature(V, params):
    intgroverVdV = integrate.quad(lambda x: _isentropic_grueneisen_over_V(x, params), params['V_0'], V)[0]
    return params['T_0']*np.exp(-intgroverVdV)
'''

# Temperature along an isentrope (Anderson and Ahrens; Equation B5)
def _isentropic_temperature(V, params):
    rhofrac = params['V_0']/V # rho/rho0 = V0/V
    x = np.power(rhofrac, 1./3.) # equation 18
    ksi1 = 0.75*(4. - params['Kprime_S']) # equation 19
    ksi2 = 0.375*(params['K_S']*params['Kprime_prime_S'] + params['Kprime_S']*(params['Kprime_S'] - 7.)) + 143./24. # equation 20

    # equation B6 -- B10
    a1 = ksi2 / 8.
    a2 = ( ksi1 + 3. * ksi2 ) / 6.
    a3 = ( 1. + 2.*ksi1 + 3.*ksi2 ) / 4.
    a4 = (1. + ksi1 + ksi2)/2.
    a5 = (6. + 4.*ksi1 + 3.*ksi2)/24.

    # equation B5
    Ts = params['T_0']*np.exp(params['grueneisen_0']*np.log(rhofrac)
                              + 13.5*params['grueneisen_prime']*params['V_0']*params['K_S'] *
                              (   (a1/(3*params['grueneisen_n'] + 8.))*(np.power(x,(3*params['grueneisen_n'] + 8.)) - 1.)
                                - (a2/(3*params['grueneisen_n'] + 6.))*(np.power(x,(3*params['grueneisen_n'] + 6.)) - 1.)
                                + (a3/(3*params['grueneisen_n'] + 4.))*(np.power(x,(3*params['grueneisen_n'] + 4.)) - 1.)
                                - (a4/(3*params['grueneisen_n'] + 2.))*(np.power(x,(3*params['grueneisen_n'] + 2.)) - 1.)
                                + (a5/(3*params['grueneisen_n'] + 0.))*(np.power(x,(3*params['grueneisen_n'] + 0.)) - 1.)))
                                
    return Ts



# Pressure along the reference isentrope
def _isentropic_pressure(V, params):
    x = np.power(params['V_0']/V, 1./3.)
    ksi1 = 0.75*(4. - params['Kprime_S'])
    ksi2 = 0.375*(params['K_S']*params['Kprime_prime_S'] + params['Kprime_S']*(params['Kprime_S'] - 7.)) + 143./24.
    x2 = x*x
    x3 = x*x*x
    x5 = x3*x2
    x7 = x5*x2

    Ps = 1.5*params['K_S'] * (x7 - x5) * (1. + ksi1 - ksi1*x2 + ksi2 * (x2 - 1.) * (x2 - 1.))

    return Ps

# Birch Murnaghan equation of state expression for the energy change along an isentrope
# Anderson and Ahrens, 1994 (Equation 21)
def _isentropic_energy_change(V, params):
    x = np.power(params['V_0']/V, 1./3.)
    ksi1 = 0.75*(4. - params['Kprime_S'])
    ksi2 = 0.375*(params['K_S']*params['Kprime_prime_S'] + params['Kprime_S']*(params['Kprime_S'] - 7.)) + 143./24.
    x2 = x*x
    x4 = x2*x2
    x6 = x4*x2
    x8 = x4*x4
    
    E_S = 4.5*params['V_0']*params['K_S'] * ((ksi1 + 1.) * (x4/4. - x2/2. + 0.25) - ksi1*(x6/6. - x4/4. + 1./12.)
                                             + ksi2*(x8/8. - x6/2. + 0.75*x4 - x2/2. + 0.125))
    return E_S

# int Cv dT 
def _isochoric_energy_change(Ts, T, V, params):
    return _internal_energy_kin(Ts, T, V, params) + _internal_energy_el(Ts, T, V, params) + _internal_energy_pot(Ts, T, V, params)

    
# (Chen and Ahrens 1995; Equations 8, 11)
def internal_energy(T, V, params):
    Ts = _isentropic_temperature(V, params)
    return params['E_0'] + _isentropic_energy_change(V, params) + _isochoric_energy_change(Ts, T, V, params)

# Equation 23
def pressure(T, V, params):
    Ts = _isentropic_temperature(V, params)
    dE = _isochoric_energy_change(Ts, T, V, params)
    E1 = params['E_0'] + _isentropic_energy_change(V, params)
    E2 = E1 + dE
    dP = (params['grueneisen_0']*dE + 0.5*params['grueneisen_prime']*np.power(params['V_0']/V, params['grueneisen_n'])*(E2*E2 - E1*E1))/V
    return _isentropic_pressure(V, params) + dP
    
# F = E - TS
def helmholtz_free_energy(T, V, params):
    return internal_energy(T, V, params) - T*entropy(T, V, params)

def _volume(V, P, T, params):
    return P - pressure(T, V, params)

def volume(P, T, params):
    return fsolve(_volume, params['V_0']*0.1, args=(P, T, params))[0]
    
# G = E - TS + PV
def gibbs_free_energy(P, T, params):
    V = volume(P, T, params)
    gibbs = helmholtz_free_energy(T, V, params) + P*V
    return gibbs

rho_0 = 7019.
m = 0.055845
V_0 = m/rho_0
D = 7766.
Lambda = 1146.
params={
    'P_0': 1.e5,
    'T_0': 1811.,
    'S_0': 100., # to fit
    'molar_mass': m,
    'V_0': V_0,
    'E_0': 0.,
    'K_S': 109.7e9,
    'Kprime_S': 4.661,
    'Kprime_prime_S': -0.043e-9,
    'grueneisen_0': 1.735,
    'grueneisen_prime': -0.130/m*1.e-6,
    'grueneisen_n': -1.870,
    'a': [248.92*m, 289.48*m],
    'b': [0.4057*m, -1.1499*m],
    'Theta': [1747.3, 1.537],
    'theta': 5000.,
    'lmda': [302.07*m, -325.23*m, 30.45*m],
    'xi_0': 282.67*m,
    'F': [D/rho_0, Lambda/rho_0]
    }


dV = 1.e-10

Vs = np.linspace(0.48*params['V_0'], 1.0*params['V_0'], 21)
Ts = np.empty_like(Vs)
Ps = np.empty_like(Vs)
Vs2 = np.empty_like(Vs)
for i, V in enumerate(Vs):
    Ts[i] = _isentropic_temperature(V, params)
    Ps[i] = pressure(Ts[i], V, params) 

    # Check pressures
    dP = 1.
    Vs2[i] = (gibbs_free_energy(Ps[i]+dP, Ts[i], params) - gibbs_free_energy(Ps[i], Ts[i], params))/dP

    # Check entropies
    #dT = 1.
    #print -(gibbs_free_energy(Ps[i], 100.+Ts[i]+dT, params) - gibbs_free_energy(Ps[i], 100.+Ts[i], params))/dT
    
fig1 = mpimg.imread('PTrho_reference_isentrope.png')

plt.imshow(fig1, extent=[0.0, 500., 6., 15.], aspect='auto')
plt.plot(Ps/1.e9, m/Vs/1.e3, linewidth=2, marker='o')
plt.plot(Ps/1.e9, m/Vs2/1.e3, linewidth=2, linestyle='--')
plt.plot(1.e-4, m/V_0/1.e3, marker='o') 
plt.title('AA1994 Figure B1 (1/2)')
plt.show()

plt.imshow(fig1, extent=[0.0, 500., 1500., 7000.], aspect='auto')
plt.plot(Ps/1.e9, Ts, linewidth=2)
plt.title('AA1994 Figure B1 (2/2)')
plt.show()




temperatures = np.linspace(1800., 2400., 100)
rhos = np.empty_like(temperatures)

P = 1.e5
for i, T in enumerate(temperatures):
    rhos[i] = m/volume(P, T, params)/1.e3

fig1 = mpimg.imread('Trho_1bar.png')
plt.imshow(fig1, extent=[1800., 2400., 6.65, 7.1], aspect='auto')
plt.plot(temperatures, rhos)
plt.ylim(6.65, 7.1)
plt.title('AA1994 Figure 1')
plt.show()


temperatures = np.linspace(1000., 15000., 101)
Cvs = np.empty_like(temperatures)

rhos = np.empty_like(temperatures)
densities = [5.e3,10.e3, 15.e3]
for rho in densities:
    V = m/rho
    for i, T in enumerate(temperatures):
        Cvs[i] = C_v(V, T, params)/burnman.constants.gas_constant

    plt.plot(temperatures, Cvs)

fig1 = mpimg.imread('TCv_different_densities.png')
plt.imshow(fig1, extent=[1000., 15000., 0., 6.], aspect='auto')
plt.ylim(0., 6.)
plt.title('AA1994, Figure 5')
plt.show()


# Check properties
P = 1.e5
T = 1811
dT = 10.
S0 = (gibbs_free_energy(P, T, params) - gibbs_free_energy(P, T+dT, params))/dT
S1 = (gibbs_free_energy(P, T+dT, params) - gibbs_free_energy(P, T+2.*dT, params))/dT

V0 = volume(P, T, params)
V1 = volume(P, T+dT, params) 



print 'C_p:', 835.*m, T*(S1 - S0)/dT
print 'alpha:', 9.27e-5, 1./V0*((V1 - V0)/dT)
print 'K_S:', params['K_S']
print 'V_0', params['V_0'], V0

print 'grueneisen:', params['grueneisen_0'], (V1 - V0)*params['K_S']/(T*(S1 - S0))



def hugoniot(P_ref, T_ref, pressures, params):
    
    def Ediff(T, P, P_ref, U_ref, V_ref):
        V = volume(P, T, params)
        U = internal_energy(T, V, params)
        
        return (U - U_ref) - 0.5*(P - P_ref)*(V_ref - V)

    V_ref = volume(P_ref, T_ref, params)
    U_ref = internal_energy(T_ref, V_ref, params)


    temperatures = np.empty_like(pressures)
    volumes = np.empty_like(pressures)
    
    for i, P in enumerate(pressures):
        temperatures[i] = fsolve(Ediff, [T_ref], args = (P, P_ref, U_ref, V_ref))[0]
        volumes[i] = volume(P, temperatures[i], params)

    return temperatures, volumes


pressures = np.linspace(300.e9, 1500.e9, 101)
temperatures, volumes = hugoniot(9000., 280., pressures, params)

plt.plot(volumes, pressures/1.e9)
plt.show() 

plt.plot(temperatures, pressures/1.e9)
plt.show() 
