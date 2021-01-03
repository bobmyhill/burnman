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
from burnman.mineral import Mineral
from burnman.processchemistry import formula_mass

class cementite( Mineral ):
    """
    Cementite from Fei and Brosh (2014)
    """
    def __init__(self):
        formula={'Fe': 3.0, 'C': 1.0}
        m = formula_mass(formula)
        self.params = {
            'name': 'cementite',
            'formula': formula,
            'equation_of_state': 'brosh_calphad',
            'molar_mass': m,
            'n': sum(formula.values()),
            'gibbs_coefficients': [[1811., [1225.7, 124.134, -23.5143, 77359.,
                                            0., 0., 0., -0.439752e-2, -5.8927e-8,
                                            0., 0., 0., 0.]]],
            'V_0': 5.755e-6*4.,
            'K_0': 200.e9,
            'Kprime_0': 5.2,
            'theta_0': 400,
            'grueneisen_0': 1.7,
            'delta': [5., 10.],
            'b': [1.,  4.]
        }
        self.property_modifiers = [
            ['magnetic_chs', {'structural_parameter': 0.28,
                              'curie_temperature': [485., 0.],
                              'magnetic_moment': [1.008, 0.]}]]
        Mineral.__init__(self)

class Fe7C3( Mineral ):
    """
    Fe7C3 from Fei and Brosh (2014)
    """
    def __init__(self):
        formula={'Fe': 7.0, 'C': 3.0}
        m = formula_mass(formula)
        self.params = {
            'name': 'Fe7C3',
            'formula': formula,
            'equation_of_state': 'brosh_calphad',
            'molar_mass': m,
            'n': sum(formula.values()),
            'gibbs_coefficients': [[1811., [1225.7, 124.134, -23.5143, 77359.,
                                            0., 0., 0., -0.439752e-2, -5.8927e-8,
                                            0., 0., 0., 0.]],
                                   [6000., [-25383.6, 299.3126, -46., 0.,
                                            0., 0., 2.29603e31, 0., 0.,
                                            0., 0., 0., 0.]],
                                   [12000., [-25383.4, 299.3122, -45.99997, 0.,
                                             0., 0., 0., 0., 0.,
                                             0., 0., 0., 0.]]],
            'V_0': 5.515e-6*10.,
            'K_0': 255.e9,
            'Kprime_0': 4.,
            'theta_0': 445,
            'grueneisen_0': 1.7,
            'delta': [4., 10.],
            'b': [1.,  4.]
        }
        self.property_modifiers = [
            ['magnetic_chs', {'structural_parameter': 0.28,
                              'curie_temperature': [525., 0.],
                              'magnetic_moment': [3.5, 0.]}]]
        Mineral.__init__(self)

class diamond( Mineral ):
    """
    Diamond from Fei and Brosh (2014)
    """
    def __init__(self):
        formula={'C': 1.0}
        m = formula_mass(formula)
        self.params = {
            'name': 'diamond',
            'formula': formula,
            'equation_of_state': 'brosh_calphad',
            'molar_mass': m,
            'n': sum(formula.values()),
            'gibbs_coefficients': [[1811., [1225.7, 124.134, -23.5143, 77359.,
                                            0., 0., 0., -0.439752e-2, -5.8927e-8,
                                            0., 0., 0., 0.]],
                                   [6000., [-25383.6, 299.3126, -46., 0.,
                                            0., 0., 2.29603e31, 0., 0.,
                                            0., 0., 0., 0.]],
                                   [12000., [-25383.4, 299.3122, -45.99997, 0.,
                                             0., 0., 0., 0., 0.,
                                             0., 0., 0., 0.]]],
            'V_0': 3.4145e-6,
            'K_0': 447.e9,
            'Kprime_0': 3.5,
            'theta_0': 1650.,
            'grueneisen_0': 0.93,
            'delta': [5., 5.],
            'b': [1.,  10.]
        }
        Mineral.__init__(self)


cem = cementite()

fcc_iron = burnman.minerals.SE_2015.fcc_iron()
bcc_iron = burnman.minerals.SE_2015.bcc_iron()
diam = burnman.minerals.HGP_2018_ds633.diam()
gph = burnman.minerals.HGP_2018_ds633.gph()

#made_cem = burnman.CombinedMineral([fcc_iron, diam], [3., 1.], [0., 0., -0.6e-6])

"""
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
    plt.plot(pressures/1.e9, cem.evaluate(['V'], pressures, temperatures)[0] / burnman.constants.Avogadro * 1.e30 * 4., label=T)
plt.legend()
plt.show()

exit()

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
