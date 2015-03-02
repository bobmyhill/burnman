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
            'a_0': 3.56e-05 ,
            'K_0': 1.64e+11 ,
            'Kprime_0': 5.16 ,
            'Kdprime_0': -3.1e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses),
            'curie_temperature': [1043., 0.0] ,
            'magnetic_moment': [2.22, 0.0] ,
            'magnetic_structural_parameter': 0.4 }
        Mineral.__init__(self)

class fcc_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FCC iron',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 7973.0 ,
            'S_0': 35.907 ,
            'V_0': 6.909e-06 ,
            'Cp': [22.24, 0.0088656, -221517., 47.1998] ,
            'a_0': 4.66e-05 ,
            'K_0': 1.51e+11 ,
            'Kprime_0': 5.2 ,
            'Kdprime_0': -3.37e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses),
            'curie_temperature': [201., 0.0] ,
            'magnetic_moment': [2.1, 0.0] ,
            'magnetic_structural_parameter': 0.28 }
        Mineral.__init__(self)

class hcp_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'HCP iron',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 5050. ,
            'S_0': 30.7,
            'V_0': 6.766e-06 ,
            'Cp': [22.24, 0.0088656, -221517., 47.1998] ,
            'a_0': 4.135e-05 ,
            'K_0': 1.59e+11 ,
            'Kprime_0': 5.289 ,
            'Kdprime_0': -3.325e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

