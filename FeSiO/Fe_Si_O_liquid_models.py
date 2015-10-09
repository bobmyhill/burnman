# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
from burnman.chemicalpotentials import *
from make_intermediate import *
from HP_convert import *
atomic_masses=read_masses()

# In this model, the mixing properties are based on intermediate compounds
# There are:
# Two FeO0.5 intermediates (to provide an asymmetric mixing model)
# One Fe0.5Si0.5 intermediate (to fit the melting curve of Lord et al., 2010)
# One Fe0.5Si0.5O0.5 intermediate 

class FeFeO_liquid (burnman.Mineral):
    def __init__(self):
        self.params = {'S_0': 127.196 - 2.*0.5*burnman.constants.gas_constant*np.log(0.5), 
                       'a_0': 1.7695938319333659e-05, 'K_0': 317979497241.90607, 'Kprime_0': 3.4381483641094981, 'T_0': 1809.0, 'T_einstein': 116.57508622201439, 'Kdprime_0': -1.0812484433529035e-11, 'V_0': 8.3095091057899312e-06, 'name': 'Liquid iron excesses', 'H_0': 418851.48305945576, 'molar_mass': 0.0638447, 'equation_of_state': 'hp_tmt', 'n': 1.5, 'formula': {'Fe': 1.0, 'O': 0.5}, 'Cp': [67.90299999999999, -0.0012802499999999997, 2488700.0, -602.1800000000001], 'P_0': 50000000000.0}
        burnman.Mineral.__init__(self)

class FeOFe_liquid (burnman.Mineral):
    def __init__(self):
        self.params = {'S_0': 127.196 - 2.*0.5*burnman.constants.gas_constant*np.log(0.5), 
                       'a_0': 1.7700557044608401e-05, 'K_0': 343087075974.04547, 'Kprime_0': 2.8503861254491683, 'T_0': 1809.0, 'T_einstein': 116.57508622201439, 'Kdprime_0': -8.3080545000325215e-12, 'V_0': 8.3073408497496906e-06, 'name': 'Liquid iron excesses', 'H_0': 417653.28195234778, 'molar_mass': 0.0638447, 'equation_of_state': 'hp_tmt', 'n': 1.5, 'formula': {'Fe': 1.0, 'O': 0.5}, 'Cp': [67.90299999999999, -0.0012802499999999997, 2488700.0, -602.1800000000001], 'P_0': 50000000000.0}
        burnman.Mineral.__init__(self)

class Si_liquid_DQF (burnman.Mineral):
    def __init__(self):
        formula='Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Si liquid',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'T_0': 1873.15, 
            'P_0': 1.e5, 
            'H_0': 54000. ,
            'S_0':  78.72,
            'V_0': 8.689e-06 ,
            'Cp': [37.656, 0., 0., 0.] ,
            'a_0': 2.87e-05 ,
            'K_0': 51.47e+9 ,
            'Kprime_0': 5.04 ,
            'Kdprime_0': -5.04/51.47e+9 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        burnman.Mineral.__init__(self)

Fe_liq=burnman.minerals.Myhill_calibration_iron.liquid_iron_HP()
#Si_liq=burnman.minerals.Fe_Si_O.Si_liquid()
Si_liq_DQF=Si_liquid_DQF()
FeO_liq = burnman.minerals.Fe_Si_O.FeO_liquid_HP()

FeFeO_liq_0 = FeFeO_liquid()
FeFeO_liq_1 = FeOFe_liquid()
FeSi_liq = burnman.minerals.Fe_Si_O.FeSi_liquid()

# Now make the FeSiO liquid intermediate 
HP_convert(Si_liq_DQF, 500., 2000., 1809., 50.e9)
HP_convert(FeSi_liq, 500., 2000., 1809., 50.e9)
# H_ex, S_ex, Sconf, V_ex, K_ex, a_ex
FeSiO_excesses = [0.e3, 0., -burnman.constants.gas_constant*np.log(0.5), 0., 0., 0.]
FeSiO_liq = make_intermediate(Si_liq_DQF, FeO_liq, FeSiO_excesses)

class metallic_Fe_Si_O_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='metallic liquid Fe-Si-O solution'
        self.type='full_subregular'
        self.endmembers = [[Fe_liq, '[Fe]'], [Si_liq_DQF, '[Si]'], [FeO_liq, 'Fe[O]']]
        self.intermediates = [[[FeSi_liq, FeSi_liq],
                               [FeFeO_liq_0,FeFeO_liq_1]],
                              [[FeSiO_liq,FeSiO_liq]]]

        burnman.SolidSolution.__init__(self, molar_fractions)



