import os, sys, numpy as np, matplotlib.pyplot as plt

if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import tools
from burnman.processchemistry import formula_mass, dictionarize_formula
from burnman import minerals
import matplotlib.image as mpimg
from scipy.optimize import brentq

crst = minerals.HP_2011_ds62.crst()
qtz = minerals.HP_2011_ds62.q()
coe = minerals.HP_2011_ds62.coe()
stv = minerals.HP_2011_ds62.stv()


def C_p_ref(T, c):
    return ( c[0] + c[1]*T + c[2]/T/T + c[3]/np.sqrt(T) +
             c[4]/T/T/T + c[5]*T*T + c[6]*T*T*T + c[7]/T )

def _intCpdT(T0, temperature, params):
    
    integral = lambda T, c: ( c[0]*T + 0.5*c[1]*T*T - c[2]/T + 2.*c[3]*np.sqrt(T)
                              -0.5*c[4]/T/T + 1./3.*c[5]*T*T*T + 1./4.*c[6]*T*T*T*T
                              + c[7]*np.log(T))
    
    return integral(temperature, params['C_p']) - integral(T0, params['C_p'])

def _intCpoverTdT(T0, temperature, params):
    integral = lambda T, c: ( c[0]*np.log(T) + c[1]*T -0.5*c[2]/T/T - 2*c[3]/np.sqrt(T)
                              -1./3.*c[4]/T/T/T + 1./2.*c[5]*T*T + 1./3.*c[6]*T*T*T
                              - c[7]/T )
    
    return integral(temperature, params['C_p']) - integral(T0, params['C_p'])

def _1bar_V_params(temperature, params):
    intadT = lambda T, a: ( 0.5*a[0]*T*T + a[1]*T + a[2]*np.log(T)
                            -a[3]/T - 0.5*a[4]/T/T )
    
    V = params['V_0']*np.exp(intadT(temperature, params['alpha']) -
                             intadT(params['T_0'], params['alpha']))
    K = ( params['K_0'] +
          (temperature - params['T_0'])*params['dKdT'] +
          np.power(temperature - params['T_0'], 2.)*params['d2KdT2'] )
    Kprime = ( params['Kprime_0'] +
               params['dKdT'] * (temperature - params['T_0']) *
               np.log(temperature/params['T_0']) )
    return (V, K, Kprime)


def volume(pressure, temperature, params):
    VT, KT, Kprime_T = _1bar_V_params(temperature, params)
    xi1 = 3.*(4. - Kprime_T)/4.
    xi2 = 0. # 3./8.*(K0*Kprimeprime_0 + Kprime_0*(Kprime_0 - 7.)) + 143./24. # 4th order

    # x = (rho/rho0)^(1/3) = (V/V0)^(-1/3)
    delta_P = lambda x, P, K0, xi1, xi2 : P - (3./2.*K0*(np.power(x, 7.) - np.power(x, 5.))*
                                               (1. + xi1 - xi1*x*x + xi2*(x*x - 1.)*(x*x - 1.)))

    x = brentq(delta_P, 0.001, 1000., args=(pressure, KT, xi1, xi2))
    return VT*np.power(x, -3.)
    
def _intPdV(x, V0, K0, xi1, xi2):
    x2 = x*x
    x4 = x2*x2
    x6 = x4*x2
    x8 = x4*x4
    
    return -9./2.*V0*K0*((xi1 + 1.)*(x4/4. - x2/2. + 1./4.) -
                         xi1*(x6/6. - x4/4. + 1./12.) +
                         xi2*(x8/8 - x6/2 + 3.*x4/4. - x2/2. + 1./8.))

def _intVdP(pressure, temperature, params):
    VT, KT, Kprime_T = _1bar_V_params(temperature, params)
    V = volume(pressure, temperature, params)
    VP = V*pressure - VT*params['P_0']
    x = np.power(V/VT, -1./3.)
    
    xi1 = 3.*(4. - Kprime_T)/4.
    xi2 = 0. # 3./8.*(K0*Kprimeprime_0 + Kprime_0*(Kprime_0 - 7.)) + 143./24. # 4th order

    return VP - _intPdV(x, VT, KT, xi1, xi2)
    
def gibbs_free_energy(pressure, temperature, params):
    return ( params['H_0'] - temperature*params['S_0'] +
             _intCpdT(params['T_0'], temperature, params) -
             temperature*_intCpoverTdT(params['T_0'], temperature, params) +
             _intVdP(pressure, temperature, params) )


    
SiO2_liq_B = burnman.Mineral(params={'equation_of_state': 'boza',
                                     'n': 3.,
                                     'formula': {'Si': 1., 'O': 2.},
                                     'molar_mass': 0.06008,
                                     'P_0': 1.e5,
                                     'T_0': 298.,
                                     'H_0': -911.7462e3,
                                     'S_0': 33.887,
                                     'C_p': np.array([88.455, -3.00137e-3, -48.527e5,
                                                      -114.33, 7.2829e8, 0.71332e-6,
                                                      0.0059239e-9, 0.]),
                                     'V_0': 27.872e-6,
                                     'K_0': 5.e9,
                                     'Kprime_0': 15.8,
                                     'dKdT': 0.,
                                     'd2KdT2': 0.,
                                     'dKprimedT': 0.,
                                     'alpha': np.array([-0.3611e-7, 843.1e-7, 0.0, 0.0, 0.0])})
    
crst_B = burnman.Mineral(params={'equation_of_state': 'boza',
                                'n': 3.,
                                'formula': {'Si': 1., 'O': 2.},
                                'molar_mass': 0.06008,
                                 'P_0': 1.e5,
                                 'T_0': 298.,
                                 'H_0': -906.3772e3,
                                 'S_0': 46.029,
                                 'C_p': np.array([83.514, 0., -24.5536e5,
                                                  -374.693, 2.8007e8, 0.,
                                                  0., 0.]),
                                 'V_0': 25.739e-6,
                                 'K_0': 20.696e9,
                                 'Kprime_0': 6.0,
                                 'dKdT': 0.,
                                 'd2KdT2': 0.,
                                 'dKprimedT': 0.,
                                 'alpha': np.array([0.040248e-7, 243.616e-7,
                                                    27.5739e-3, 0.0, 0.0])})

qtz_B = burnman.Mineral(params={'equation_of_state': 'boza',
                                'n': 3.,
                                'formula': {'Si': 1., 'O': 2.},
                                'molar_mass': 0.06008,
                                 'P_0': 1.e5,
                                 'T_0': 298.,
                                 'H_0': -908.6268e3,
                                 'S_0': 44.207,
                                 'C_p': np.array([80.012, 0., -35.4668e5,
                                                  -240.276, 4.9157e8, 0.,
                                                  0., 0.]),

                                
                                 'V_0': 22.875e-6,
                                 'K_0': 51.495e9,
                                 'Kprime_0': 5.3,
                                 'dKdT': -135.3e5,
                                 'd2KdT2': 0.,
                                 'dKprimedT': 0.,
                                 'alpha': np.array([0.34694e-7, 206.036e-7,
                                                    0.1307e-3, -1.6376, 0.0])})

coe_B = burnman.Mineral(params={'equation_of_state': 'boza',
                                'n': 3.,
                                'formula': {'Si': 1., 'O': 2.},
                                'molar_mass': 0.06008,
                                 'P_0': 1.e5,
                                 'T_0': 298.,
                                 'H_0': -906.900e3,
                                 'S_0': 40.5,
                                 'C_p': np.array([78., 0., -31.e5,
                                                  0., 5.858e8, 0.,
                                                  0., -0.6689e4]),
                                 'V_0': 20.64e-6,
                                 'K_0': 92.879e9,
                                 'Kprime_0': 8.4,
                                 'dKdT': -91.e5,
                                 'd2KdT2': 0.,
                                 'dKprimedT': 0.,
                                 'alpha': np.array([0.05e-7, 54.3e-7,
                                                    0., 0.0, 0.0])})

stv_B = burnman.Mineral(params={'equation_of_state': 'boza',
                                'n': 3.,
                                'formula': {'Si': 1., 'O': 2.},
                                'molar_mass': 0.06008,
                                 'P_0': 1.e5,
                                 'T_0': 298.,
                                 'H_0': -874.865e3,
                                 'S_0': 27.809,
                                 'C_p': np.array([59.945, -13.967e-3, -220.34e5,
                                                  1109.7, 44.972e8, 9.1326e-6,
                                                  -1.3264e-9, 0.]),
                                 'V_0': 14.e-6,
                                 'K_0': 255.e9,
                                 'Kprime_0': 6.59,
                                 'dKdT': -345.e5,
                                 'd2KdT2': 0.,
                                 'dKprimedT': 0.,
                                 'alpha': np.array([0.05904e-7, 167.77e-7,
                                                    2.2872e-3, -2.1518, 237.0245])})


temperatures = np.linspace(1700., 2300., 101)
pressures = [1.e5] * len(temperatures)

fig = plt.figure()
ax_V = fig.add_subplot(4, 1, 1)
ax_G = fig.add_subplot(4, 1, 2)
ax_S = fig.add_subplot(4, 1, 3)
ax_C = fig.add_subplot(4, 1, 4)
ax_V.plot(temperatures, SiO2_liq_B.evaluate(['V'], pressures, temperatures)[0])
ax_V.plot(temperatures, crst_B.evaluate(['V'], pressures, temperatures)[0])
ax_V.plot(temperatures, crst.evaluate(['V'], pressures, temperatures)[0])


ax_G.plot(temperatures, (SiO2_liq_B.evaluate(['gibbs'], pressures, temperatures)[0] -
                         crst_B.evaluate(['gibbs'], pressures, temperatures)[0]))

ax_S.plot(temperatures, (SiO2_liq_B.evaluate(['S'], pressures, temperatures)[0] -
                         crst_B.evaluate(['S'], pressures, temperatures)[0]))


ax_C.plot(temperatures, SiO2_liq_B.evaluate(['heat_capacity_p'], pressures, temperatures)[0]) 
ax_C.plot(temperatures, crst_B.evaluate(['heat_capacity_p'], pressures, temperatures)[0]) 
ax_C.plot(temperatures, crst.evaluate(['heat_capacity_p'], pressures, temperatures)[0]) 
plt.show()

crst = crst_B
coe = coe_B
qtz = qtz_B
stv = stv_B
liq = SiO2_liq_B


pressures = np.linspace(1.e5, 15.e9, 101)
temperatures = np.array([2000.]*len(pressures))

for m in [crst, qtz, coe, stv, liq]:
    plt.plot(pressures/1.e9, m.evaluate(['gibbs'], pressures, temperatures)[0])
plt.show()


'''
P_0 = 1.e5
T_0 = 1999.
crst.set_state(P_0, T_0)
F_0 = crst.gibbs
S_0 = crst.S + 4.46
V_0 = crst.V
Kprime_inf = 3.2
gamma_inf = 0.5*Kprime_inf - 1./6.
formula = dictionarize_formula('SiO2')
liq = burnman.Mineral(params={'equation_of_state': 'simple_melt',
                              'formula': formula,
                              'n': sum(formula.values()),
                              'V_0': V_0,
                              'K_0': 13.5e9,
                              'Kprime_0': 5.5,
                              'Kprime_inf': Kprime_inf,
                              'molar_mass': formula_mass(formula),
                              'G_0': 0.e9, # melt
                              'Gprime_inf': 1.,
                              'gamma_0': 0.05,
                              'gamma_inf': gamma_inf,
                              'q_0': 1.,
                              'C_v': 83.,
                              'P_0': P_0,
                              'T_0': T_0,
                              'F_0': F_0,
                              'S_0': S_0,
                              'lambda_0': 5.,
                              'lambda_inf': 4.})
'''

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
fig1 = mpimg.imread('figures/Shen_Lazor_1995_SiO2_melt.png')
ax.imshow(fig1, extent=[0., 80., 1500.-273.15, 5000.-273.15], aspect='auto')

fig1 = mpimg.imread('figures/Zhang_et_al_1993_SiO2_melt.png')
ax.imshow(fig1, extent=[0., 15., 1400., 3000.], aspect='auto')


inv = []
for (m1, m2, P_guess, T_guess) in [(crst, qtz, 0.4e9, 1990.),
                                   (qtz, coe, 4.e9, 2500.),
                                   (coe, stv, 14.e9, 3000.)]:
    pt = tools.invariant_point([liq, m1], [1., -1.],
                               [liq, m2], [1., -1.],
                               pressure_temperature_initial_guess=[P_guess, T_guess])
    inv.append(pt)
    temperatures = np.linspace(1200., pt[1], 101)
    pressures = np.zeros_like(temperatures)
    for i, T in enumerate(temperatures):
        pressures[i] = tools.equilibrium_pressure([m1, m2], [1., -1.],
                                                  T, pressure_initial_guess = P_guess)

    ax.plot(pressures/1.e9, temperatures-273.15, label='{0}-{1}'.format(m1.name, m2.name))
    

for (m, pressures, T_guess) in [(crst, np.linspace(1.e5, inv[0][0], 101), inv[0][1]),
                                (qtz, np.linspace(inv[0][0], inv[1][0], 101), inv[1][1]),
                                (coe, np.linspace(inv[1][0], inv[2][0], 101), inv[2][1]),
                                (stv, np.linspace(inv[2][0], 80.e9, 101), inv[2][1])]:
    temperatures = np.zeros_like(pressures)
    for i, P in enumerate(pressures):
        temperatures[i] = tools.equilibrium_temperature([liq, m], [1., -1.],
                                                        P, temperature_initial_guess = T_guess)

    ax.plot(pressures/1.e9, temperatures-273.15, label='{0}-liq'.format(m.name))




PTs = [[1.e5, 2000.93408591],
       [1.e9, 2191.60813947],
       [2.e9, 2382.27861806],
       [5.e9, 2757.18107273],
       [10.e9, 3316.26545404],
       [20.e9, 4027.25345035],
       [25.e9, 4340.73094649]]

ax.scatter(np.array(zip(*PTs)[0])/1.e9, np.array(zip(*PTs)[1])-273.15)
ax.set_xlabel('Pressure (GPa)')
ax.set_ylabel('Temperature ($^{\circ}$C)')
ax.legend(loc='upper left')


fig.savefig("SiO2_melt_first_guess.pdf", bbox_inches='tight', dpi=100)


plt.show()




exit()

'''
Klim = liq.params['K_0']*np.power((1. - liq.params['Kprime_inf']/liq.params['Kprime_0']),
                                  liq.params['Kprime_0']/liq.params['Kprime_inf'])
Plim = Klim/(liq.params['Kprime_inf'] - liq.params['Kprime_0'])

Vlim = ( liq.params['V_0'] *
         np.power ( liq.params['Kprime_0'] /
                    (liq.params['Kprime_0'] - liq.params['Kprime_inf']),
                    liq.params['Kprime_0'] /
                    liq.params['Kprime_inf'] /
                    liq.params['Kprime_inf'] ) *
         np.exp(-1./liq.params['Kprime_inf']) )

print('{0} GPa'.format(Klim/1.e9))
'''


liq.set_state(100.e9, 2000.)
#tools.check_eos_consistency(liq, verbose=False)

fig = plt.figure()
ax_V = fig.add_subplot(3, 1, 1)
ax_C = fig.add_subplot(3, 1, 2)
ax_Kp = fig.add_subplot(3, 1, 3)
for (pressures, T) in [(np.linspace(1.e5, 10.e9, 101), 2000.),
                       (np.linspace(1.e9, 20.e9, 101), 3000.),
                       (np.linspace(2.e9, 40.e9, 101), 4000.)]:
                
    temperatures = [T] * len(pressures)
    volumes, C_p, K_T = liq.evaluate(['V', 'heat_capacity_p', 'isothermal_bulk_modulus'], pressures, temperatures)
    ax_V.plot(pressures/1.e9, volumes*1.e6, label='{0:.0f} K'.format(T))
    ax_C.plot(pressures/1.e9, C_p, label='{0:.0f} K'.format(T))
    ax_Kp.plot(pressures/1.e9, np.gradient(K_T, pressures), label='{0:.0f} K'.format(T))

ax_V.set_xlabel('P (GPa)')
ax_C.set_xlabel('P (GPa)')
ax_Kp.set_xlabel('P (GPa)')
ax_V.set_ylabel('Volume')
ax_C.set_ylabel('$C_p$')
ax_Kp.set_ylabel('K\'')
ax_V.legend(loc='upper right')
ax_C.legend(loc='upper right')
ax_Kp.legend(loc='upper right')
plt.show()
