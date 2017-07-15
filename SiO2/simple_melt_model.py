import os, sys, numpy as np, matplotlib.pyplot as plt
# from scipy.special import gammaincc # no good for negative a
from mpmath import gammainc
from scipy.special import expi, hyp2f1
from scipy.optimize import brentq

if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import tools

Kprime_inf = 3.0
gamma_inf = 0.5*Kprime_inf - 1./6.
liq = burnman.Mineral(params={'equation_of_state': 'simple_melt',
                              'formula': {'Si': 1., 'O': 2.},
                              'n': 3.,
                              'V_0': 27.3e-6,
                              'K_0': 10.e9,
                              'Kprime_0': 10.0,
                              'Kprime_inf': Kprime_inf,
                              'molar_mass': 0.06008,
                              'G_0': 2.e9,
                              'Gprime_inf': 1.,
                              'gamma_0': 0.1,
                              'gamma_inf': gamma_inf,
                              'q_0': 1.,
                              'C_v': 83.,
                              'P_0': 0.,
                              'T_0': 1999.,
                              'F_0': 0.,
                              'S_0': 0.,
                              'lambda_0': 5.,
                              'lambda_inf': 4.})



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
ax_V = fig.add_subplot(1, 2, 1)
ax_C = fig.add_subplot(1, 2, 2)
for (pressures, T) in [(np.linspace(1.e5, 10.e9, 101), 2000.),
                       (np.linspace(1.e9, 20.e9, 101), 3000.),
                       (np.linspace(2.e9, 40.e9, 101), 4000.)]:
                
    temperatures = [T] * len(pressures)
    volumes, C_p = liq.evaluate(['V', 'heat_capacity_p'], pressures, temperatures)
    ax_V.plot(pressures/1.e9, volumes*1.e6, label='{0:.0f} K'.format(T))
    ax_C.plot(pressures/1.e9, C_p, label='{0:.0f} K'.format(T))

ax_V.legend(loc='upper right')
ax_C.legend(loc='upper right')
plt.show()
