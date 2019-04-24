from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from scipy.optimize import minimize, fsolve, curve_fit
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))
import burnman
from burnman.solidsolution import SolidSolution as Solution
from burnman.solutionbases import transform_solution_to_new_basis

#########################
# ENDMEMBER DEFINITIONS #
#########################


class Re_metal (burnman.Mineral):
    def __init__(self):
        self.params = {'name': 'rhenium metal',
                       'formula': {'Re': 1.0},
                       'equation_of_state': 'hp_tmt',
                       'H_0': 0., # element
                       'S_0': 36.526, # Barin
                       'V_0': 8.8726e-6, # Anzellini et al., 2014
                       'Cp': [2.94473137e+01, 1.50972491e-03,
                              2.13054329e+04, -8.84258085e+01], # from Arblaster data up to 1800 K
                       'a_0': 6.2e-6, # wiki
                       'K_0': 352.6e9, # Anzellini et al., 2014
                       'Kprime_0': 4.56, # Anzellini et al., 2014
                       'Kdprime_0': -4.56/352.6e9, # HP heuristic
                       'n': 1.0,
                       'molar_mass': 0.186207}
        burnman.Mineral.__init__(self)

class ReO2_ortho (burnman.Mineral):
    def __init__(self):
        self.params = {'name': 'rhenium (IV) oxide',
                       'formula': {'Re': 1.0, 'O': 2.0},
                       'equation_of_state': 'hp_tmt',
                       'H_0': -444.350e3, # Pownceby and O'Neill, 1994
                       'S_0': 47.83, # Stuve and Ferrante, Barin
                       'V_0': 8.8726e-6 + 9.944e-6,  # rough fit to Campbell, 2006
                       'Cp': [ 6.69551851e+01,  1.33883977e-02,
                               -1.47341623e+06, -7.94983573e-02], # Barin
                       'a_0': 13.7e-6,  # rough fit to Campbell, 2006
                       'K_0': 189.e9,  # rough fit to Campbell, 2006
                       'Kprime_0': 4., # rough fit to Campbell, 2006
                       'Kdprime_0': -4./189.e9, # rough fit to Campbell, 2006
                       'n': 3.0,
                       'molar_mass': 0.218206}
        burnman.Mineral.__init__(self)



"""
def func_Cp(temperature, a, b, c, d):
    return (a + b * temperature +
            c * np.power(temperature, -2.) +
            d * np.power(temperature, -0.5))

T, Cp = np.loadtxt('data/Re_Cp_Arblaster_1996.dat', unpack=True)
mask = [i for i, temp in enumerate(T) if temp < 1800.]
Re_Cp_params = curve_fit(func_Cp, T[mask], Cp[mask], [1., 1., 1., 1.])[0]

plt.plot(T, Cp)
plt.plot(T, func_Cp(T, *Re_Cp_params))



T, Cp = np.loadtxt('data/ReO2_Cp_Barin.dat', unpack=True)
mask = [i for i, temp in enumerate(T) if temp < 1800.]
ReO2_Cp_params = curve_fit(func_Cp, T[mask], Cp[mask], [1., 1., 1., 1.])[0]

plt.plot(T, Cp)
temperatures = np.linspace(300., 2000., 101)
plt.plot(temperatures, func_Cp(temperatures, *ReO2_Cp_params))
plt.show()

print(ReO2_Cp_params)
exit()

print(0.218206/0.01161e6 - 8.8726e-6)
"""
"""
Re = Re_metal()
O2 = burnman.minerals.HP_2011_fluids.O2()

def gibbs_formation_ReO2_Jacob(T):
    Re.set_state(1.e5, T)
    O2.set_state(1.e5, T)
    return -451510. + 295.011*T - 14.3261*T*np.log(T) + O2.gibbs + Re.gibbs

def gibbs_formation_ReO2_PO(T):
    Re.set_state(1.e5, T)
    O2.set_state(1.e5, T)
    return -451020. + 297.595*T - 14.6585*T*np.log(T) + O2.gibbs + Re.gibbs

temperatures = np.linspace(290., 310., 101)
G = [gibbs_formation_ReO2_Jacob(T) for T in temperatures]
plt.plot(temperatures, -np.gradient(G, temperatures))
G = [gibbs_formation_ReO2_PO(T) for T in temperatures]
plt.plot(temperatures, -np.gradient(G, temperatures))
plt.show()
exit()
"""
"""
Re = Re_metal()
ReO2 = ReO2_ortho()
pressures = np.linspace(1.e5, 10.e9, 101)
TC = [20., 200., 400, 600, 800, 1000.]

def mu_O2(T):
    Re.set_state(1.e5, T)
    ReO2.set_state(1.e5, T)
    O2.set_state(1.e5, T)
    return ReO2.gibbs - Re.gibbs - O2.gibbs

print(mu_O2(823.), mu_O2(1073)) # see Pownceby and O'Neill for comparison.

exit()


img = mpimg.imread('figures/ReO2_Re_volume_difference.png')
plt.imshow(img, extent=[0.0, 10.0, 9.4, 10.2], alpha=0.3, aspect='auto')

for T in TC:
    temperatures = pressures*0. + T + 273.15
    plt.plot(pressures/1.e9, (ReO2.evaluate('V', pressures, temperatures)[0] -
                              Re.evaluate('V', pressures, temperatures)[0])*1.e6)

plt.xlim(0., 10)
plt.ylim(9.4, 10.2)
plt.show()
"""
