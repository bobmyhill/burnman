# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

"""
Fe-Si-O database
"""

from burnman.mineral import Mineral
from burnman.solidsolution import SolidSolution
from burnman.solutionmodel import *
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass

atomic_masses=read_masses()

# Lacaze and Sundman (1991) suggest  0.5*Fe(nonmag) + 0.5*Si - 36380.6 + 2.22T
class FeSi_B20 (Mineral): # WARNING, magnetic properties screw up Barin
    def __init__(self):
        formula='Fe1.0Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FeSi B20',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -78852. , # Barin
            'S_0': 44.685 , # Barin
            'V_0': 1.359e-05 ,
            'Cp': [38.6770e+01, 0.0217569, -159.151, 0.0060376],
            'a_0': 2.811e-05 ,
            'K_0': 2.057e+11 ,
            'Kprime_0': 4.0 ,
            'Kdprime_0': -4.0/2.057e+11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses),
            'curie_temperature': [521.5, 0.0] ,
            'magnetic_moment': [1.11, 0.0] ,
            'magnetic_structural_parameter': 0.4 }
        Mineral.__init__(self)

class FeSi_B2 (Mineral): # enthalpy and entropy calculated from A2-structured Fe and Si
    def __init__(self):
        formula='Fe0.5Si0.5'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Fe0.5Si0.5 B2',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': (9149.0 + 47000.)/2. -59650/2., # includes enthalpy of ordering
            'S_0': (36.868 + 18.820 + 22.5)/2. -9.4943/2., # includes entropy of ordering
            'V_0': 1.300e-05/2. ,
            'Cp': [(21.09 + 22.826)/2., (0.0101455 + 0.003856857)/2., (-221508.+-353888.416)/2., (47.1947 + -0.0596068)/2.],
            'a_0': 3.064e-05 ,
            'K_0': 2.199e+11 ,
            'Kprime_0': 4.0 ,
            'Kdprime_0': -4.0/2.199e+11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses),
            'curie_temperature': [521.5, 0.0] ,
            'magnetic_moment': [1.11, 0.0] ,
            'magnetic_structural_parameter': 0.4 }
        Mineral.__init__(self)

'''
class FeSi_B2 (Mineral):
    def __init__(self):
        formula='Fe1.0Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FeSi B2',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -78852. , # to fit
            'S_0': 44.685 , # to fit
            'V_0': 1.300e-05 ,
            'Cp': [38.6770e+01, 0.0217569, -159.151, 0.0060376],
            'a_0': 3.064e-05 ,
            'K_0': 2.199e+11 ,
            'Kprime_0': 4.0 ,
            'Kdprime_0': -4.0/2.199e+11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses),
            'curie_temperature': [521.5, 0.0] ,
            'magnetic_moment': [1.11, 0.0] ,
            'magnetic_structural_parameter': 0.4 }
        Mineral.__init__(self)
'''

class Si_diamond_A4 (Mineral):
    def __init__(self):
        formula='Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Si A4',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 0. , # Barin
            'S_0': 18.820 , # Barin
            'V_0': 1.20588e-05 , # Hallstedt, 2007
            'Cp': [22.826, 0.003856857, -353888.416, -0.0596068], # Barin
            'a_0': 7.757256e-06 , # Fit to Roberts, 1981
            'K_0': 101.e+9 , # Hu et al., 1986 (fit to V/V0 at 11.3 GPa)
            'Kprime_0': 4.0 , # 
            'Kdprime_0': -4.0/101.e+9 , # 
            'T_einstein': 764., # Fit to Roberts, 1981 (would be 516. from 0.8*Tdebye (645 K); see wiki)
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class Si_bcc_A2 (Mineral):
    def __init__(self):
        formula='Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Si bcc A2',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 47000., # SGTE data
            'S_0': 18.820 + 22.5, # Barin, SGTE data
            'V_0': 9.1e-06 , # Hallstedt, 2007
            'Cp': [22.826, 0.003856857, -353888.416, -0.0596068], # Barin
            'a_0': 7.757256e-06 , # Fit to Roberts, 1981
            'T_einstein': 764., # Fit to Roberts, 1981
            'K_0': 50.e+9 , # ? Must destabilise BCC relative to HCP, FCC
            'Kprime_0': 6.0 , # Similar to HCP, FCC
            'Kdprime_0': -6.0/50.e+9 , # ?
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class Si_fcc_A1 (Mineral):
    def __init__(self):
        formula='Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Si fcc A1',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 51000. , # SGTE data
            'S_0': 18.820 + 21.8 , # Barin, SGTE data
            'V_0': 9.2e-06 , # Hallstedt, 2007
            'Cp': [22.826, 0.003856857, -353888.416, -0.0596068], # Barin
            'a_0': 7.757256e-06 , # Fit to Roberts, 1981
            'T_einstein': 764., # Fit to Roberts, 1981
            'K_0': 40.15e9 , # 84 = Duclos et al 
            'Kprime_0': 6.1 , # 4.22 = Duclos et al 
            'Kdprime_0': -6.1/40.15e9 , # Duclos et al 
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class Si_hcp_A3 (Mineral):
    def __init__(self):
        formula='Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Si hcp A3',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 49200., # SGTE data
            'S_0': 18.820 + 20.8, # Barin, SGTE data
            'V_0': 8.8e-06 , # Hallstedt, 2007, smaller than fcc
            'Cp': [22.826, 0.003856857, -353888.416, -0.0596068], # Barin
            'a_0': 7.757256e-06 , # Fit to Roberts, 1981
            'T_einstein': 764., # Fit to Roberts, 1981
            'K_0': 57.44e9 , # 72 = Duclos et al 
            'Kprime_0': 5.87 , # Fit to Mujica et al. # 3.9 for Duclos et al 
            'Kdprime_0': -5.87/57.44e9 , # Duclos et al 
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)
