
# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

"""
Sundman 1991 models for bcc, fcc iron
Written in the Holland and Powell, 2011 form

EoS terms for bcc are from HP_2011_ds62 for iron
EoS terms for fcc are from an unpublished calibration 
Thermodynamic properties for HCP are also unpublished.

Full details and the fitting procedure can be found in iron_property_derivation.py (in the misc directory)
"""

from burnman.mineral import Mineral
from burnman.solidsolution import SolidSolution
from burnman.solutionmodel import *
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass

atomic_masses=read_masses()


"""
ENDMEMBERS
"""

class bcc_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'BCC iron',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 9149.0 ,
            'S_0': 36.868 ,
            'V_0': 7.09e-06 ,
            'Cp': [21.09, 0.0101455, -221508., 47.1947] ,
            'a_0': 3.8e-05 ,
            'K_0': 1.64e+11 ,
            'Kprime_0': 5.16 ,
            'Kdprime_0': -3.1e-11 ,
            'einstein_T': 470.*0.806,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses),
            'curie_temperature': [1043., 0.0] ,
            'magnetic_moment': [2.22, 0.0] ,
            'magnetic_structural_parameter': 0.4 }
        Mineral.__init__(self)
'''
class fcc_iron (Mineral): # No magnetism
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {'S_0': 35.907, 'a_0': 4.9057399193621506e-05, 'K_0': 151102660510.68918, 'einstein_T': 378.82000000000005, 'Kprime_0': 5.5, 'T_0': 298.15, 'Kdprime_0': -3.6399094373397373e-11, 'V_0': 6.9364989947404524e-06, 'name': 'FCC iron', 'H_0': 7973.0, 'molar_mass': 0.055845, 'equation_of_state': 'hp_tmt', 'n': 1.0, 'formula': {'Fe': 1.0}, 'Cp': [22.24, 0.0088656, -221517.0, 47.1998], 'P_0': 99999.99}
        Mineral.__init__(self)
'''
class fcc_iron (Mineral): # No magnetism
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {'S_0': 35.907, 'V_0': 6.9359301440628439e-06, 'name': 'FCC iron', 'H_0': 7973.0, 'a_0': 4.9105508240851435e-05, 'K_0': 150885687310.07358, 'molar_mass': 0.055845, 'equation_of_state': 'hp_tmt', 'n': 1.0, 'P_0': 99999.99, 'formula': {'Fe': 1.0}, 'einstein_T': 378.82000000000005, 'Kprime_0': 5.6, 'T_0': 298.15, 'Cp': [22.24, 0.0088656, -221517.0, 47.1998], 'Kdprime_0': -3.7114189555248337e-11}
        Mineral.__init__(self)
'''
class hcp_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {'S_0': 31.426500159503139, 'a_0': 4.7602079599957996e-05, 'K_0': 162295349883.27124, 'einstein_T': 378.82000000000005, 'Kprime_0': 5.1312284277268976, 'T_0': 298.15, 'Kdprime_0': -3.1616607816659349e-11, 'V_0': 6.7813155012469347e-06, 'name': 'HCP iron', 'H_0': 5902.3667637576928, 'molar_mass': 0.055845, 'equation_of_state': 'hp_tmt', 'n': 1.0, 'formula': {'Fe': 1.0}, 'Cp': [22.24, 0.0088656, -221517.0, 47.1998], 'P_0': 99999.99}
        Mineral.__init__(self)
'''
class hcp_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {'S_0': 31.331366709266057, 'V_0': 6.7813155012469347e-06, 'name': 'HCP iron', 'H_0': 5785.4777535512894, 'a_0': 4.7602079599957996e-05, 'K_0': 162295349883.27124, 'molar_mass': 0.055845, 'equation_of_state': 'hp_tmt', 'n': 1.0, 'P_0': 99999.99, 'formula': {'Fe': 1.0}, 'einstein_T': 378.82000000000005, 'Kprime_0': 5.1312284277268976, 'T_0': 298.15, 'Cp': [22.24, 0.0088656, -221517.0, 47.1998], 'Kdprime_0': -3.1616607816659349e-11}
        Mineral.__init__(self)

class liquid_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Liquid iron',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'T_0': 1809.,
            'P_0': 0.999999e5,
            'H_0': 74206. , # Barin
            'S_0': 100.725 , # Barin
            'V_0': 55.845/6.85e6 ,
            'Cp': [46.024, 0., 0., 0.] , # Barin
            'a_0': 6.5e-05 ,
            'K_0': 95.e+9 ,
            'Kprime_0': 5.4 ,
            'Kdprime_0': -5.40/95.e+9,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)


class hcp_iron_HP (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {'S_0': 71.803, 'V_0': 5.8175e-06, 'name': 'HCP iron', 'H_0': 351039.0, 'a_0': 2.6143e-05, 'K_0': 336240000000.0, 'molar_mass': 0.055845, 'equation_of_state': 'hp_tmt', 'n': 1.0, 'Kdprime_0': -1.377e-11, 'Kprime_0': 4.3846, 'einstein_T': 378.82, 'formula': {'Fe': 1.0}, 'T_0': 1809.0, 'Cp': [114.4, -0.014984, 11124000.0, -2563.1], 'P_0': 50000000000.0}
        Mineral.__init__(self)


class fcc_iron_HP (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {'S_0': 75.879, 'V_0': 5.9147e-06, 'name': 'FCC iron', 'H_0': 358436.0, 'a_0': 2.4852e-05, 'K_0': 339380000000.0, 'molar_mass': 0.055845, 'equation_of_state': 'hp_tmt', 'n': 1.0, 'P_0': 50000000000.0, 'formula': {'Fe': 1.0}, 'einstein_T': 378.82, 'Kprime_0': 4.671, 'T_0': 1809.0, 'Cp': [148.21, -0.02239, 15632000.0, -3548.3], 'Kdprime_0': -1.4502e-11}
        Mineral.__init__(self)

class liquid_iron_HP (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {'S_0': 89.202, 'a_0': 1.8963e-05, 'K_0': 325630000000.0, 'einstein_T': 99.249, 'Kprime_0': 4.1181, 'T_0': 1809.0, 'Kdprime_0': -1.3702e-11, 'V_0': 6.2864e-06, 'name': 'Liquid iron', 'H_0': 403124.0, 'molar_mass': 0.055845, 'equation_of_state': 'hp_tmt', 'n': 1.0, 'formula': {'Fe': 1.0}, 'Cp': [75.261, -0.0086939, 3401400.0, -812.75], 'P_0': 50000000000.0}
        Mineral.__init__(self)
