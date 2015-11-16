
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
            'T_einstein': 470.*0.806,
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
        self.params = {'S_0': 35.907, 
                       'V_0': 6.9359301440628439e-06, 
                       'name': 'FCC iron',
                       'H_0': 7973.0, 
                       'a_0': 4.9105508240851435e-05, 
                       'K_0': 150885687310.07358, 
                       'molar_mass': 0.055845, 
                       'equation_of_state': 'hp_tmt', 
                       'n': 1.0, 
                       'P_0': 99999.99, 
                       'formula': {'Fe': 1.0}, 
                       'T_einstein': 378.82000000000005, 
                       'Kprime_0': 5.6, 
                       'T_0': 298.15, 
                       'Cp': [52.2754, -0.000355156, 790710.86, -619.07], 
                       'Kdprime_0': -3.7114189555248337e-11}
        Mineral.__init__(self)

class hcp_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {'S_0': 31.331366709266057, 
                       'V_0': 6.7813155012469347e-06, 
                       'name': 'HCP iron', 
                       'H_0': 5785.4777535512894, 
                       'a_0': 4.7602079599957996e-05, 
                       'K_0': 162295349883.27124, 
                       'molar_mass': 0.055845, 
                       'equation_of_state': 'hp_tmt', 
                       'n': 1.0, 'P_0': 99999.99, 
                       'formula': {'Fe': 1.0}, 
                       'T_einstein': 378.82000000000005, 
                       'Kprime_0': 5.1312284277268976, 
                       'T_0': 298.15, 
                       'Cp': [52.2754, -0.000355156, 790710.86, -619.07], 
                       'Kdprime_0': -3.1616607816659349e-11}
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
            'V_0': 55.845/6.85e6 , # Hixson et al., 1990, Nasch and Manghnani; 6.85 to fit melting curve
            'Cp': [46.024, 0., 0., 0.] , # Barin
            'a_0': 6.0e-05 , # Hixson et al., 1990, Nasch and Manghnani
            'K_0': 98.e+9 ,
            'Kprime_0': 5.4 ,
            'Kdprime_0': -5.40/98.e+9,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class fcc_iron_SLB (Mineral):
    def __init__(self):
        formula='Fe'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FCC iron',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -2805. ,
            'V_0': 6.835e-6 ,
            'K_0': 165.3e9 ,
            'Kprime_0': 5.5 ,
            'Debye_0': 417.0 ,
            'grueneisen_0': 1.72 ,
            'q_0': 1. ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class hcp_iron_SLB (Mineral):
    def __init__(self):
        formula='Fe'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'HCP iron',
            'formula': formula,
            'equation_of_state': 'slb3',
            'T_0': 300.,
            'F_0': 1000. , # needs to be changed
            'V_0': 6.828e-6, # Uchida and Dewaele 6.73e-6 , # fit to Dewaele et al. (2006) (up to 100 GPa)
            'K_0': 141.e9, # Uchida and Dewaele 165.0e9 , # fit to Dewaele et al. (2006) (up to 100 GPa)
            'Kprime_0': 5.83, # fit to Uchida and Dewaele # 5.32 , # gr = 0.5*K' - 0.94 (Anderson, 2000)
            'Debye_0': 422.0 , # 422 K at 0 GPa; Sharma, 2009
            'grueneisen_0': 2.46, # fit to Uchida # 1.72 , # (Anderson, 2000)
            'q_0': 1. , # needs to be changed?
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

'''
class hcp_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {'S_0': 61.350804976190318, 'V_0': 4.5013291881820265e-06, 'name': 'HCP iron', 'H_0': 1086433.5537034615, 'a_0': 9.7932496815829863e-06, 'K_0': 897588615919.35352, 'molar_mass': 0.055845, 'equation_of_state': 'hp_tmt', 'n': 1.0, 'Kdprime_0': -4.1992187505002218e-12, 'Kprime_0': 3.3059347155761718, 'formula': {'Fe': 1.0}, 'T_einstein': 378.82000000000005, 'Cp': [52.2754, -0.000355156, 790710.86, -619.07], 'T_0': 1809.0, 'P_0': 200000000000.0}
        Mineral.__init__(self)

class fcc_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {'S_0': 64.922336564061524, 'V_0': 4.6311213292725059e-06, 'name': 'FCC iron', 'H_0': 1109977.9772064621, 'a_0': 8.8169963792607211e-06, 'K_0': 956158199251.24707, 'molar_mass': 0.055845, 'equation_of_state': 'hp_tmt', 'n': 1.0, 'Kdprime_0': -4.3701171881238569e-12, 'Kprime_0': 3.6404608215332033, 'formula': {'Fe': 1.0}, 'T_einstein': 378.82000000000005, 'Cp': [52.2754, -0.000355156, 790710.86, -619.07], 'T_0': 1809.0, 'P_0': 200000000000.0}
        Mineral.__init__(self)

class liquid_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {'S_0': 101.58903315159262, 'a_0': 1.6356321119070639e-05, 'K_0': 468545480615.50824, 'Kprime_0': 3.7509524340820311, 'T_0': 3324.8164853502135, 'T_einstein': 378.82000000000005, 'Kdprime_0': -9.0820312470540905e-12, 'V_0': 5.5443592052113084e-06, 'name': 'Liquid iron', 'H_0': 694568.860256674, 'molar_mass': 0.055845, 'equation_of_state': 'hp_tmt', 'n': 1.0, 'formula': {'Fe': 1.0}, 'Cp': [38.0, 0.0, 0.0, 0.0], 'P_0': 97000000000.0}
        Mineral.__init__(self)
'''

