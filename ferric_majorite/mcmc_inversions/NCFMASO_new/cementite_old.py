import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from scipy.optimize import fsolve

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))

import burnman


# Paramagnetic phase = 2.327cm^3, 190.8, 4.68

class cementite (burnman.Mineral):
    def __init__(self):
        self.params = {'name': 'cementite (paramagnetic phase)',
                       'formula': {'Fe': 3.0, 'C': 1.0},
                       'equation_of_state': 'hp_tmt',
                       'H_0': -6.082e3 + 298.15*104.6, # Barin
                       'S_0': 104.6 - 4.2, # Hallstedt et al.
                       'V_0': 23.27e-6, # Approx fit to Litasov et al.
                       'Cp': [135., 0.015, 0., -900.], # rough estimate from Dick compilation
                       'a_0': 4.3e-05, # Approx fit to Litasov et al.
                       'K_0': 194.e9, # Approx fit to Litasov et al.
                       'Kprime_0': 4.5, # Approx fit to Litasov et al.
                       'Kdprime_0': -4.5/194.e9, # HP heuristic
                       'n': 4.0,
                       'molar_mass': 0.17954}
        self.property_modifiers = [['bragg_williams', {'deltaH': 485*8.31446*(3.+1.)/2.-600., # Tc = ~ 485 K
                                                       'deltaV': 4.2e-07,
                                                       'Wh': 485*8.31446*(3.+1.)/2.-1400.,
                                                       'Wv': 4.2e-07,
                                                       'n': 3.0,
                                                       'factor': 1.}]]
        burnman.Mineral.__init__(self)

cem = cementite()

fcc_iron = burnman.minerals.SE_2015.fcc_iron()
bcc_iron = burnman.minerals.SE_2015.bcc_iron()
diam = burnman.minerals.HGP_2018_ds633.diam()
gph = burnman.minerals.HGP_2018_ds633.gph()

#made_cem = burnman.CombinedMineral([fcc_iron, diam], [3., 1.], [0., 0., -0.6e-6])


cem_cp_img = mpimg.imread('cementite_cp_dick_2011.png')
plt.imshow(cem_cp_img, extent=[0., 1500., 0., 4.5*8.31446*4.], aspect='auto', alpha=0.3)


temperatures = np.linspace(100., 1500., 101)
pressures = 0.*temperatures + 1.e5
plt.plot(temperatures, cem.evaluate(['molar_heat_capacity_p'], pressures, temperatures)[0])
#plt.show()

"""
cem_V_img = mpimg.imread('cementite_PVT_Litasov_2013.png')
plt.imshow(cem_V_img, extent=[0., 35., 135., 165.], aspect='auto', alpha=0.3)

pressures = np.linspace(0., 35.e9, 101)
for T in np.linspace(300., 1500., 7):
    temperatures = T + 0.*pressures
    plt.plot(pressures/1.e9, cem.evaluate(['V'], pressures, temperatures)[0] / burnman.constants.Avogadro * 1.e30 * 4.)
plt.show()
"""

gph.set_state(1.e5, 300.)
bcc_iron.set_state(1.e5, 300.)
cem.set_state(1.e5, 300.)
print(gph.gibbs, bcc_iron.gibbs, cem.gibbs)
print(cem.S)

T = [298.15, 300.,      400., 485.,       485.01,  500., 1500.]
Cp = [105.868, 106.023, 114.391, 121.503, 113.282, 113.470, 126.022]
plt.plot(T, Cp)
plt.show()
exit()

temperatures = np.linspace(300., 1500., 101)
pressures = 0.*temperatures + 1.e5
cem_elements = burnman.CombinedMineral([bcc_iron, gph], [3., 1.])
G_el, H_el, S_el = cem_elements.evaluate(['gibbs', 'H', 'S'], pressures, temperatures)
G, H, S = cem.evaluate(['gibbs', 'H', 'S'], pressures, temperatures)
plt.plot(temperatures, H - H_el) # Hallstedt figure 2
#plt.plot(temperatures, G - G_el) #
#plt.plot(temperatures, S) #
#plt.plot(temperatures, S_el) #
plt.show()
