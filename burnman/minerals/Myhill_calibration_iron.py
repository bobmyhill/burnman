
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
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses),
            'curie_temperature': [1043., 0.0] ,
            'magnetic_moment': [2.22, 0.0] ,
            'magnetic_structural_parameter': 0.4 }
        Mineral.__init__(self)

class fcc_iron (Mineral): # No magnetism
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FCC iron',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 7973.0 ,
            'S_0': 35.907 ,
            'V_0': 6.9767e-06 ,
            'Cp': [22.24, 0.0088656, -221517., 47.1998] ,
            'a_0': 4.6766e-05 ,
            'K_0': 150.4e+9 ,
            'Kprime_0': 5.2 ,
            'Kdprime_0': -5.2/150.4e+9 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class hcp_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'HCP iron',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 4300. ,
            'S_0': 29.,
            'V_0': 6.766e-06 ,
            'Cp': [22.24, 0.0088656, -221517., 47.1998] ,
            'a_0': 4.25e-05 , # fixed to make ~ straight reaction line
            'K_0': 1.59e+11 ,
            'Kprime_0': 5.289 ,
            'Kdprime_0': -3.325e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
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
            'V_0': 55.845/6.97e6 ,
            'Cp': [46.024, 0., 0., 0.] , # Barin
            'a_0': 6.0e-05 ,
            'K_0': 100.e+9 ,
            'Kprime_0': 5.04 ,
            'Kdprime_0': -5.04/100.e+9,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)


class hcp_iron_HP (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {'S_0': 72.714, 'a_0': 2.1091e-05, 'K_0': 347580000000.0, 'einstein_T': 300.11, 'Kprime_0': 4.4873, 'T_0': 1809.0, 'Kdprime_0': -1.3507e-11, 'V_0': 5.7642e-06, 'name': 'HCP iron', 'H_0': 351143.0, 'molar_mass': 0.055845, 'equation_of_state': 'hp_tmt', 'n': 1.0, 'formula': {'Fe': 1.0}, 'Cp': [66.789, -0.0035302, 5093300.0, -1201.2], 'P_0': 50000000000.0}
        Mineral.__init__(self)


class fcc_iron_HP (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {'S_0': 78.293, 'a_0': 2.2295e-05, 'K_0': 334050000000.0, 'einstein_T': 251.16, 'Kprime_0': 4.3805, 'T_0': 1809.0, 'Kdprime_0': -1.3818e-11, 'V_0': 5.9045e-06, 'name': 'FCC iron', 'H_0': 361864.0, 'molar_mass': 0.055845, 'equation_of_state': 'hp_tmt', 'n': 1.0, 'formula': {'Fe': 1.0}, 'Cp': [81.537, -0.0071466, 6942300.0, -1622.2], 'P_0': 50000000000.0}
        Mineral.__init__(self)

class liquid_iron_HP (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {'S_0': 89.748, 'a_0': 1.8986e-05, 'K_0': 316030000000.0, 'einstein_T': 99.249, 'Kprime_0': 3.8498, 'T_0': 1809.0, 'Kdprime_0': -1.3251e-11, 'V_0': 6.1712e-06, 'name': 'Liquid iron', 'H_0': 398216.0, 'molar_mass': 0.055845, 'equation_of_state': 'hp_tmt', 'n': 1.0, 'formula': {'Fe': 1.0}, 'Cp': [67.865, -0.0068169, 2498900.0, -603.64], 'P_0': 50000000000.0}
        Mineral.__init__(self)
