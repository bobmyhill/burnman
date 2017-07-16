import os, sys, numpy as np, matplotlib.pyplot as plt

if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import tools
from burnman.processchemistry import formula_mass, dictionarize_formula
from burnman import minerals
import matplotlib.image as mpimg

crst = minerals.HP_2011_ds62.crst()
qtz = minerals.HP_2011_ds62.q()
coe = minerals.HP_2011_ds62.coe()
stv = minerals.HP_2011_ds62.stv()


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
    

for (m, pressures, T_guess) in [(crst, np.linspace(1.e5, inv[0][0], 101), 1990.),
                                (qtz, np.linspace(inv[0][0], inv[1][0], 101), 1990.),
                                (coe, np.linspace(inv[1][0], inv[2][0], 101), 1990.),
                                (stv, np.linspace(inv[2][0], 80.e9, 101), 1990.)]:
    temperatures = np.zeros_like(pressures)
    for i, P in enumerate(pressures):
        temperatures[i] = tools.equilibrium_temperature([liq, m], [1., -1.],
                                                        P, temperature_initial_guess = T_guess)

    ax.plot(pressures/1.e9, temperatures-273.15, label='{0}-liq'.format(m.name))
    
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
