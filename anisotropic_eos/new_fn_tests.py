from __future__ import absolute_import

# Here we import standard python modules that are required for
# usage of BurnMan.  In particular, numpy is used for handling
# numerical arrays and mathematical operations on them, and
# matplotlib is used for generating plots of results of calculations
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))
import numpy as np
import matplotlib.pyplot as plt
import burnman


class outer_core_rkprime(burnman.Mineral):

    """
    Stacey and Davis, 2004 PEPI (Table 5)
    """

    def __init__(self):
        self.params = {
            'equation_of_state': 'rkprime',
            'P_0': 0.,
            'V_0': 0.055845/6562.54,
            'K_0': 124.553e9,
            'Kprime_0': 4.9599,
            'Kprime_inf': 3.0,
            'molar_mass': 0.055845
        }
        burnman.Mineral.__init__(self)

ocrk = outer_core_rkprime()
pressures = np.linspace(-12.e9, 100.e9, 101)

V, E, K = ocrk.evaluate(['V', 'molar_internal_energy', 'K_T'], pressures, pressures*0.)

#plt.plot(V/ocrk.params['V_0'], E)
plt.plot(pressures, np.gradient(K, pressures))
plt.show()
exit()



def tait_constants(params):
    """
    returns parameters for the modified Tait equation of state
    derived from K_T and its two first pressure derivatives
    EQ 4 from Holland and Powell, 2011
    """
    a = (1. + params['Kprime_0']) / (
        1. + params['Kprime_0'] + params['K_0'] * params['Kdprime_0'])
    b = params['Kprime_0'] / params['K_0'] - \
        params['Kdprime_0'] / (1. + params['Kprime_0'])
    c = (1. + params['Kprime_0'] + params['K_0'] * params['Kdprime_0']) / (
        params['Kprime_0'] * params['Kprime_0'] + params['Kprime_0'] - params['K_0'] * params['Kdprime_0'])
    return a, b, c

def volume(pressure, a, b, c):
    """
    Returns volume/V_0 as a function of pressure [Pa] and temperature [K]
    EQ 12
    """
    x = 1 - a * (1. - np.power((1. + b * (pressure)), -1.0 * c))
    return x

def bulk_modulus(P, a, b, c):
    return ((1 + b*P)*(a + (1. - a)*(np.power(1 + b*P, c))))/(a*b*c)


params = {'K_0': 485.e9,
          'Kprime_0': 11.85,
          'Kdprime_0': -42./500.e9}
a, b, c = tait_constants(params)

per = burnman.minerals.HP_2011_ds62.per()

pressures = np.linspace(-40.e9, 300.e9, 501)
volumes = per.evaluate(['V'], pressures, 300. + 0.*pressures)[0]/per.params['V_0']
bulk_moduli = per.evaluate(['K_T'], pressures, 300. + 0.*pressures)[0]/per.params['K_0']

L = np.power(volumes, 1./3.)
K_L = -L*np.gradient(pressures, L, edge_order=2)
V = np.power(volume(pressures, a, b, c), 3.)

fig = plt.figure(figsize=(12, 5))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

ax[0].plot(pressures, V)
ax[0].plot(pressures, volumes, linestyle=':')

ax[1].plot(pressures, bulk_modulus(pressures, a, b, c)*(a*b*c))
ax[1].plot(pressures, bulk_moduli, linestyle=':')


ax[2].plot(pressures, np.gradient(bulk_modulus(pressures, a, b, c)*(a*b*c), pressures, edge_order=2))
ax[2].plot(pressures, np.gradient(bulk_moduli, pressures, edge_order=2), linestyle=':')
"""
plt.plot(pressures/1.e9, K_L)
plt.plot(pressures/1.e9, np.gradient(K_L, pressures, edge_order=2))

#plt.plot(pressures/1.e9, bulk_modulus(pressures, a, b, c), linestyle=':')
plt.plot(pressures/1.e9,
         np.gradient(bulk_modulus(pressures, a, b, c), pressures, edge_order=2),
         linestyle=':')
"""

plt.show()
exit()


print(tait_constants(params))
a, b, c = tait_constants(params)

pressures = np.linspace(0.e9, 300.e9, 101)

def bulk_modulus(P, a, b, c):
    return ((1 + b*P)*(a + (1. - a)*(np.power(1 + b*P, c))))/(a*b*c)

def Kprime(P, a, b, c):
    return (a + (1. + a*(-1. - c) + c)*np.power(1 + b*P, c))/(a*c)


fig = plt.figure(figsize=(12, 5))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

K = bulk_modulus(pressures, a, b, c)
S = 1./K
print(S)
Smod = S - 8.e-12*(Kprime(0, a, b, c))/((1. + b*pressures)*Kprime(pressures, a, b, c))
print(Smod)
invSmod = 1./Smod

ax[0].plot(pressures/1.e9, invSmod)
#ax[1].plot(pressures/1.e9, np.gradient(1./S, pressures))
ax[1].plot(pressures/1.e9, np.gradient(invSmod, pressures))


plt.show()
