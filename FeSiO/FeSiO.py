# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
from burnman.chemicalpotentials import *
import FeO_MgO_SiO2_liquid

atomic_masses=read_masses()

# In this model, the mixing properties are based on intermediate compounds
# There are:
# Two FeO0.5 intermediates (to provide an asymmetric mixing model)
# One Fe0.5Si0.5 intermediate (to fit the melting curve of Lord et al., 2010)
# One Fe0.5Si0.5O0.5 intermediate 

class FeFeO_liquid (burnman.Mineral):
    def __init__(self):
        self.params = {'S_0': 127.196, 'a_0': 1.7695938319333659e-05, 'K_0': 317979497241.90607, 'Kprime_0': 3.4381483641094981, 'T_0': 1809.0, 'T_einstein': 116.57508622201439, 'Kdprime_0': -1.0812484433529035e-11, 'V_0': 8.3095091057899312e-06, 'name': 'Liquid iron excesses', 'H_0': 418851.48305945576, 'molar_mass': 0.0638447, 'equation_of_state': 'hp_tmt', 'n': 1.5, 'formula': {'Fe': 1.0, 'O': 0.5}, 'Cp': [67.90299999999999, -0.0012802499999999997, 2488700.0, -602.1800000000001], 'P_0': 50000000000.0}
        burnman.Mineral.__init__(self)

class FeOFe_liquid (burnman.Mineral):
    def __init__(self):
        self.params = {'S_0': 127.196, 'a_0': 1.7700557044608401e-05, 'K_0': 343087075974.04547, 'Kprime_0': 2.8503861254491683, 'T_0': 1809.0, 'T_einstein': 116.57508622201439, 'Kdprime_0': -8.3080545000325215e-12, 'V_0': 8.3073408497496906e-06, 'name': 'Liquid iron excesses', 'H_0': 417653.28195234778, 'molar_mass': 0.0638447, 'equation_of_state': 'hp_tmt', 'n': 1.5, 'formula': {'Fe': 1.0, 'O': 0.5}, 'Cp': [67.90299999999999, -0.0012802499999999997, 2488700.0, -602.1800000000001], 'P_0': 50000000000.0}
        burnman.Mineral.__init__(self)


class FeSiO_liquid (burnman.Mineral):
    def __init__(self):
        formula='Fe0.5Si0.5O0.5'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Liquid Fe0.5Si0.5O0.5',
            'formula': formula,
            'n': sum(formula.values()), 
            'molar_mass': formula_mass(formula, atomic_masses), 
            'T_0': 1809.0, 
            'P_0': 50.e9,
            'S_0': 89.202 + 165.19, 
            'a_0': (1.8963e-05 + 1.6534e-05)/2., 
            'K_0': (325.63e9 + 279.96e9)/2., 
            'T_einstein': (99.249 + 227.82), 
            'Kprime_0': (4.1181 + 3.1829)/2., 
            'Kdprime_0': (-1.3702e-11 + -1.2457e-11)/2., 
            'V_0': (6.2864e-06 + 1.0577e-05)/2.,  
            'H_0': (403124.0 + 424324.0)/2., 
            'equation_of_state': 'hp_tmt', 
            'Cp': [(75.261 + 60.545)/2., 
                   (-0.0086939 + 0.0061334)/2., 
                   (3401400.0 + 1576000.0)/2., 
                   (-812.75 + -391.61)/2.] 
            }
        burnman.Mineral.__init__(self)


FeFeO_liq_0 = FeFeO_liquid()
FeFeO_liq_1 = FeOFe_liquid()
FeSi_liq = burnman.minerals.Fe_Si_O.FeSi_liquid()
FeSiO_liq = FeSiO_liquid()

Fe_liq=burnman.minerals.Myhill_calibration_iron.liquid_iron_HP()
Si_liq=burnman.minerals.Fe_Si_O.Si_liquid()
FeO_liq = burnman.minerals.Fe_Si_O.FeO_liquid_HP()


class metallic_Fe_Si_O_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='metallic liquid Fe-Si-O solution'
        self.type='full_subregular'
        self.endmembers = [[Fe_liq, '[Fe]'], [Si_liq, '[Si]'], [FeO_liq, 'Fe[O]']]
        self.intermediates = [[[FeSi_liq, FeSi_liq],
                               [FeFeO_liq_0,FeFeO_liq_1]],
                              [[FeSiO_liq,FeSiO_liq]]]

        burnman.SolidSolution.__init__(self, molar_fractions)


# Now there are two melts which we want to equilibrate with each other. 
# The first is an MgO-FeO-SiO2 melt, which gives us three chemical potentials:
# mu_MgO, mu_FeO, mu_SiO2
# The second is the metallic melt, which also gives us three chemical potentials
# mu_Fe, mu_FeO, mu_Si
# To find the equilibrium between these two phases, we need to equate two chemical potentials:
# mu_FeO = mu_FeO
# mu_SiO2 = mu_Si + 2*(mu_FeO - mu_Fe)


#silicate_melt = FeO_MgO_SiO2_liquid.FeO_MgO_SiO2_liquid()
metallic_melt = metallic_Fe_Si_O_liquid()

P = 25.e9
T = 3000.

#FeO_MgO_SiO2 wt percents
oxides = ['FeO', 'MgO', 'SiO2']
molar_masses = np.array([71.844, 40.3044, 60.08])
wt_composition_silicate_melt = np.array([4.4, 40.0, 53.8])

molar_fractions = (wt_composition_silicate_melt/molar_masses) \
    / np.sum(wt_composition_silicate_melt/molar_masses)
print oxides
print molar_fractions

exit()


#component_formulae=['FeO', 'SiO2', 'O2']
#component_formulae_dict=[dictionarize_formula(f) for f in component_formulae]
#chem_potentials=chemical_potentials(FMQ, component_formulae_dict)
