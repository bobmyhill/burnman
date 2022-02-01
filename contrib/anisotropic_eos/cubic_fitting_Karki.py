# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
cubic_fitting
-------------

This script creates an AnisotropicMineral object corresponding to
periclase (a cubic mineral). If run_fitting is set to True, the script uses
experimental data to find the optimal anisotropic parameters.
If set to False, it uses pre-optimized parameters.
The data is used only to optimize the anisotropic parameters;
the isotropic equation of state is taken from
Stixrude and Lithgow-Bertelloni (2011).

The script ends by making three plots; one with elastic moduli
at high pressure, one with the corresponding shear moduli,
and one with the elastic moduli at 1 bar.
"""

from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from tools import print_table_for_mineral_constants

import burnman

from burnman import AnisotropicMineral
from lmfit import Model

def fn_quartic(x, a, b, c, d, e):
    return a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4

def fn_linear_plus_scaled_einstein(x, a, b):
    return a/b*(1. - 0.5*(b*x) - (b*x)/(np.exp(b*x) - 1.))

run_fitting = True

per = burnman.minerals.SLB_2011.periclase()
per.set_state(1.e5, 300.)
a = np.cbrt(per.params['V_0'])
cell_parameters = np.array([a, a, a, 90, 90, 90])

K_0 = (3. * 3.012178e+11 + 6. * 9.372143e+10) / 9.
Kprime_0 = (3. * (3.910155e+11 - 3.012178e+11) + 6. * (1.082509e+11 - 9.372143e10)) / 9.e10
per.params['K_0'] = K_0
per.params['Kprime_0'] = Kprime_0

print(K_0/1.e9, Kprime_0)

per_data = np.loadtxt('data/Karki_2000_periclase_CSijs.dat')

per2 = burnman.Mineral(per.params)


def make_voigt_matrix_inverse(Cs):
    T, P, C11, C12, C44 = Cs
    C = np.zeros((6, 6))
    C[:3, :3] = C12
    for i in range(3):
        C[i, i] = C11
    for i in range(3, 6):
        C[i, i] = C44
        
    compliance = np.linalg.inv(C)
    
    beta = np.sum(compliance[:3, :3])
    print(beta)
    
    dpsidf = compliance / beta
    
    per2.set_state(P, T)
    f = -np.log(per2.V / per2.params['V_0'])
    f2 = 0.5*(np.power(per2.V / per2.params['V_0'], -2./3.) - 1.)

    return np.array([T, P, f, f2, dpsidf[0, 0], dpsidf[0, 1], dpsidf[3, 3]])

dpsidf_data = np.empty((per_data.shape[0], per_data.shape[1]+2))
for i in range(len(per_data)):
    dpsidf_data[i] = make_voigt_matrix_inverse(per_data[i])

fig = plt.figure(figsize=(12, 4))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

pmodel = Model(fn_quartic)

dpsidf0 = dpsidf_data[0, 4:]

for idx in [0]:  # only plot 300 K data
    T = dpsidf_data[idx*16, 0]
    P = dpsidf_data[idx*16:(idx+1)*16, 1]
    f = dpsidf_data[idx*16:(idx+1)*16, 2] # /2.6
    f2 = dpsidf_data[idx*16:(idx+1)*16, 3]
    d = dpsidf_data[idx*16:(idx+1)*16, 4:].T
    intd = cumulative_trapezoid(d, f, initial=0)

    for j in range(3):
        print(d[j,0])
        ln, = ax[j].plot(f, intd[j] - dpsidf0[j]*f, label=f'{T} K')
        # ax[j].plot(f2, intd[j] - dpsidf0[j]*f, label=f'{T} K', color=ln.get_color(), linestyle=':')

        params = pmodel.make_params(a=0, b=0, c=0, d=0, e=0)
        params['a'].vary = False  # only good for 300 K data
        # params['e'].vary = False
        result = pmodel.fit(intd[j], params, x=f)
        #ax[j].plot(f, np.gradient(np.gradient(result.best_fit, f, edge_order=2), f, edge_order=2), linestyle=':')
        #ax[j].plot(f, result.best_fit, linestyle=':')


#plt.ylim(1., 2.5)
labels = ['11', '12', '44']
for i in range(3):
    ax[i].legend()
    ax[i].set_xlabel('$f$')
    ax[i].set_ylabel(f'$\\psi_{{{labels[i]}}} - d\\psi_{{{labels[i]}}}/df(0)f$')
    
x = np.linspace(0., 0.4, 101)

a = 0.27
b = 18.
ax[0].plot(x, fn_linear_plus_scaled_einstein(x, a, b))
a = -0.135
b = 18.
ax[1].plot(x, fn_linear_plus_scaled_einstein(x, a, b))
a = -1.35
b = 18.
ax[2].plot(x, fn_linear_plus_scaled_einstein(x, a, b))
ax[2].plot(x, x*x*2)
fig.set_tight_layout(True)
plt.show()

exit()


def make_cubic_mineral_from_parameters(x):
    f_order = 3
    Pth_order = 1
    constants = np.zeros((6, 6, f_order+1, Pth_order+1))

    S11_0 = x[0]
    dS11df = x[1]
    d2S11df2 = x[2]
    dS11dPth = x[3] * 1.e-11
    d2S11dfdPth = x[4] * 1.e-11

    S44_0 = x[5]
    dS44df = x[6]
    d2S44df2 = x[7]
    dS44dPth = x[8] * 1.e-11
    d2S44dfdPth = x[9] * 1.e-11

    S12_0 = (1. - 3.*S11_0)/6.
    dS12df = -dS11df/2.
    d2S12df2 = -d2S11df2/2.
    dS12dPth = -dS11dPth/2.
    d2S12dfdPth = -d2S11dfdPth/2.

    constants[:3, :3, 1, 0] = S12_0
    constants[:3, :3, 2, 0] = dS12df
    constants[:3, :3, 3, 0] = d2S12df2
    constants[:3, :3, 0, 1] = dS12dPth
    constants[:3, :3, 1, 1] = d2S12dfdPth
    for i in range(3):
        constants[i, i, 1, 0] = S11_0
        constants[i, i, 2, 0] = dS11df
        constants[i, i, 3, 0] = d2S11df2

        constants[i, i, 0, 1] = dS11dPth
        constants[i, i, 1, 1] = d2S11dfdPth

    for i in range(3, 6):
        constants[i, i, 1, 0] = S44_0
        constants[i, i, 2, 0] = dS44df
        constants[i, i, 3, 0] = d2S44df2

        constants[i, i, 0, 1] = dS44dPth
        constants[i, i, 1, 1] = d2S44dfdPth

    params = per.params.copy()
    params['K_0'] = x[10] * 1.e11
    params['Kprime_0'] = x[11]
    params['Debye_0'] = x[12] * 1.e2
    params['grueneisen_0'] = x[13]
    per2 = burnman.Mineral(per.params)
    return AnisotropicMineral(per2, cell_parameters, constants)


parameters = []

if run_fitting:
    def cubic_misfit(x):
        m = make_cubic_mineral_from_parameters(x)

        chisqr = 0.
        for d in per_data:
            T, P = d[:2]
            if P < 1.e6:
                P = 1.e6
            C11S, C12S, C44S = d[2:]
            C11Serr = 2.e9
            C12Serr = 2.e9
            C44Serr = 2.e9

            m.set_state(P, T)
            if T < 1500.: # NEEP! NEEP! Data goes to 3000.
                chisqr += np.power((m.isentropic_stiffness_tensor[0, 0]
                                    - C11S)/C11Serr, 2.)
                chisqr += np.power((m.isentropic_stiffness_tensor[0, 1]
                                    - C12S)/C12Serr, 2.)
                chisqr += np.power((m.isentropic_stiffness_tensor[3, 3]
                                    - C44S)/C44Serr, 2.)

        print(x)
        print(chisqr)

        return chisqr
    guess = [1./3., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.6, 4., per.params['Debye_0']/1.e2, per.params['grueneisen_0']]
    guess = [ 0.68958617,  1.29086128,  2.10604605,  0.27526503,  0.54288477,  1.16665555,
             -3.92189089, -1.65159697, -0.17861876,  0.60128407, 1.6, 4., per.params['Debye_0']/1.e2, per.params['grueneisen_0']]
    sol = minimize(cubic_misfit, guess,
                   method='COBYLA') # , options={'rhobeg': 0.1})

    parameters = sol.x

if not run_fitting:
    parameters = [0.64434719,  0.97982023,  2.28703418,  0.04069744,
                  0.83313498, 1.02999379, -3.39390829, -2.02738898,
                  0.06480835,  0.52939447]
    parameters = [ 0.71347368,  1.4314243,   2.12871283,  0.24777966,  0.4750499,   1.27520559,
                  -3.98204241 ,-1.49992909, -0.21123366 , 0.70132031]
    
    #parameters = [ 0.68958617,  1.29086128,  2.10604605,  0.27526503,  0.54288477,  1.16665555,
    #         -3.92189089, -1.65159697, -0.17861876,  0.60128407]
    #parameters = [0.68807652, 1.40351479, 2.08535703, 0.42409575, 0.48804268, 1.22879546,
    #              -3.92561477, -1.46975486, -0.23071934, 0.55567641]
    # [ 0.61040186  0.32603485  0.0907998   0.35249197  0.29193188  1.74722438, -1.23556194  0.67754783 -0.32308102  0.01860834]

m = make_cubic_mineral_from_parameters(parameters)

m.set_state(1.e5, 300.)

fig = plt.figure(figsize=(4, 12))
fig2 = plt.figure(figsize=(4, 4))
ax = [fig.add_subplot(3, 1, i) for i in range(1, 4)]
ax.append(fig2.add_subplot(1, 1, 1))

pressures = np.linspace(0., 150.e9, 101)
G_iso = np.empty_like(pressures)
G_aniso = np.empty_like(pressures)
C11 = np.empty_like(pressures)
C12 = np.empty_like(pressures)
C44 = np.empty_like(pressures)
G_slb = np.empty_like(pressures)
G_V = np.empty_like(pressures)
G_R = np.empty_like(pressures)

f = np.empty_like(pressures)
dXdf = np.empty_like(pressures)

temperatures = [300., 1000., 2000., 3000.]
for T in temperatures:
    for i, P in enumerate(pressures):

        per.set_state(P, T)
        m.set_state(P, T)
        C11[i] = m.isentropic_stiffness_tensor[0, 0]
        C12[i] = m.isentropic_stiffness_tensor[0, 1]
        C44[i] = m.isentropic_stiffness_tensor[3, 3]
        G_V[i] = m.isentropic_shear_modulus_voigt
        G_R[i] = m.isentropic_shear_modulus_reuss

        G_slb[i] = per.shear_modulus

    ax[0].plot(pressures/1.e9, C11/1.e9, label=f'{T} K')
    ax[1].plot(pressures/1.e9, C12/1.e9, label=f'{T} K')
    ax[2].plot(pressures/1.e9, C44/1.e9, label=f'{T} K')

    ln = ax[3].plot(pressures/1.e9, G_R/1.e9)
    ax[3].plot(pressures/1.e9, G_V/1.e9, color=ln[0].get_color(), label=f'{T} K')

    ax[3].fill_between(pressures/1.e9, G_R/1.e9, G_V/1.e9,
                       alpha=0.3, color=ln[0].get_color())

    ax[3].plot(pressures/1.e9, G_slb/1.e9, label=f'{T} K (SLB2011)',
               linestyle='--', color=ln[0].get_color())

    # T, PGPa, C11S, C12S, C44S
    T_data = np.array([[d[1], d[2], d[3], d[4]]
                       for d in per_data if np.abs(d[0] - T) < 1])/1.e9

    ax[0].scatter(T_data[:, 0], T_data[:, 1])
    ax[1].scatter(T_data[:, 0], T_data[:, 2])
    ax[2].scatter(T_data[:, 0], T_data[:, 3])

for i in range(4):
    ax[i].set_xlabel('Pressure (GPa)')

ax[0].set_ylabel('$C_{N 11}$ (GPa)')
ax[1].set_ylabel('$C_{N 12}$ (GPa)')
ax[2].set_ylabel('$C_{N 44}$ (GPa)')
ax[3].set_ylabel('$G$ (GPa)')

for i in range(4):
    ax[i].legend()

fig.set_tight_layout(True)
fig.savefig('periclase_stiffness_tensor_Karki.pdf')
fig2.set_tight_layout(True)
fig2.savefig('periclase_shear_modulus_Karki.pdf')
plt.show()


fig = plt.figure(figsize=(4, 4))
ax = [fig.add_subplot(1, 1, 1)]

temperatures = np.linspace(10., 3000., 101)
P = 1.e5
for i, T in enumerate(temperatures):
    m.set_state(P, T)
    per.set_state(P, T)
    C11[i] = m.isentropic_stiffness_tensor[0,0]
    C12[i] = m.isentropic_stiffness_tensor[0,1]
    C44[i] = m.isentropic_stiffness_tensor[3,3]
    G_V[i] = m.isentropic_shear_modulus_voigt
    G_R[i] = m.isentropic_shear_modulus_reuss

    G_slb[i] = per.shear_modulus
ax[0].plot(temperatures, C11/1.e9, label='$C_{N 11}$')
ax[0].plot(temperatures, C12/1.e9, label='$C_{N 12}$')
ax[0].plot(temperatures, C44/1.e9, label='$C_{44}$')
ln = ax[0].plot(temperatures, G_R/1.e9, color=ln[0].get_color(), label=f'$G$')
ax[0].plot(temperatures, G_V/1.e9, color=ln[0].get_color())
ax[0].fill_between(temperatures, G_R/1.e9, G_V/1.e9,
                   alpha = 0.3, color=ln[0].get_color(), zorder=2)
ax[0].plot(temperatures, G_slb/1.e9, label='$G$ (SLB2011)',
           color=ln[0].get_color(), linestyle='--', zorder=1)

# T, PGPa, Perr, rho, rhoerr, C11S, C11Serr, C12S, C12Serr, C44S, C44Serr
LP_data = np.array([[d[0], d[2]/1.e9, d[3]/1.e9, d[4]/1.e9] for d in per_data if d[1] < 0.1])

ax[0].scatter(LP_data[:, 0], LP_data[:, 1])
ax[0].scatter(LP_data[:, 0], LP_data[:, 2])
ax[0].scatter(LP_data[:, 0], LP_data[:, 3])

ax[0].set_xlabel('Temperature (K)')
ax[0].set_ylabel('Elastic modulus (GPa)')

print_table_for_mineral_constants(m, [(1, 1), (1, 2), (4, 4)])

ax[0].legend()

fig.set_tight_layout(True)
fig.savefig('periclase_properties_1bar_Karki.pdf')
plt.show()
