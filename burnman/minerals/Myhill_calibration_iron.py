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

class fcc_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FCC iron',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'T_0': 2000. ,
            'P_0': 0.9999999e5, 
            'H_0': 65118.7544662 ,
            'S_0': 95.4623681681 ,
            'V_0': 7.793e-06 ,
            'Cp': [22.24, 0.0088656, -221517., 47.1998] ,
            'a_0': 8.86600454731e-05 ,
            'K_0': 87.46e+9 ,
            'Kprime_0': 5.2 ,
            'Kdprime_0': -5.2/87.46e+9 ,
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
            'T_0': 3000,
            'P_0': 0.99999e5, 
            'H_0': 104485. ,
            'S_0': 104.393,
            'V_0': 7.96566e-06 ,
            'Cp': [22.24, 0.0088656, -221517., 47.1998] ,
            'a_0': 0.0001024 ,
            'K_0': 61.13e+9 ,
            'Kprime_0': 6.3186 ,
            'Kdprime_0': -1.0335e-10 ,
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
            'H_0': 74206. ,
            'S_0': 100.725 ,
            'V_0': 55.845/6.98e6 ,
            'Cp': [46.024, 0., 0., 0.] ,
            'a_0': 8.2e-05 ,
            'K_0': 87.5e+9 ,
            'Kprime_0': 5.18 ,
            'Kdprime_0': -5.2/88.e+9,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)
