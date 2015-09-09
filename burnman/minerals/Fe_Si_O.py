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
class FeSi_B20 (Mineral): # WARNING, no magnetic properties to avoid screwing up Barin
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FeSi B20',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -78852. , # Barin
            'S_0': 44.685 , # Barin
            'V_0': 1.359e-05 ,
            'Cp': [38.6770, 0.0217569, -156.765, 0.00461],
            'a_0': 3.285e-05 ,
            'K_0': 2.0565e+11 ,
            'Kprime_0': 4.0 ,
            'Kdprime_0': -4.0/2.0565e+11,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)


class FeSi_liquid (Mineral): 
    def __init__(self):
        formula='Fe0.5Si0.5'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Fe0.5Si0.5 liquid',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'T_0': 1683.,
            'P_0': 1.e5,
            'H_0': 75015./2. , # Barin
            'S_0': 183.619/2. , # Barin
            'V_0': 8.2e-06 ,
            'Cp': [83.680/2., 0., 0., 0.], # Barin
            'a_0': 3.057e-05 ,
            'K_0': 1.09e+11 ,
            'Kprime_0': 4.0 ,
            'Kdprime_0': -4.0/1.09e+11,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses) }
        Mineral.__init__(self)

class FeSi_B2 (Mineral): # No magnetism!!
    def __init__(self):
        formula='Fe0.5Si0.5'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Fe0.5Si0.5 B2',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -28180.3, # GJ2014 
            'S_0': 22.958,   # GJ2014
            'V_0': 6.3482e-06,
            'Cp': [38.6770/2., 0.0217569/2., -156.765/2., 0.00461/2.],
            'a_0': 3.331e-05 ,
            'K_0': 2.520e+11 ,
            'Kprime_0': 4.0,
            'Kdprime_0': -4.0/2.520e+11,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)
        
 
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
            'K_0': 57.44e9, # Mujica et al., # 72 = Duclos et al 
            'Kprime_0': 5.87, # Mujica et al. # 3.9 for Duclos et al 
            'Kdprime_0': -5.87/57.44e9 , # Duclos et al 
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class Si_liquid (Mineral):
    def __init__(self):
        formula='Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
           'name': 'Si liquid',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 48471., # Barin, JANAF
            'S_0': 44.466, # Barin, JANAF
            'V_0': 9.90e-6, # Fit to Chathoth et al., 2008; 8.6145e-06 is Hallstedt, 2007, smaller than fcc
            'Cp': [27.196, 0., 0., 0.], # Barin, JANAF
            'a_0': 3.915e-05 , # Fit to Chathoth et al., 2008
            'T_einstein': 764., # As for others
            'K_0': 85.00e9, # Fit to high pressure melting (see Cannon, 1974)
            'Kprime_0': 4., # Similar to A4
            'Kdprime_0': -4./85.e9 , # HP heuristic
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)


class FeO_solid (Mineral):
    def __init__(self):
        formula='Fe1.0O1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FeO solid',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -265450.0 ,
            'S_0': 58.0 ,
            'V_0': 1.2239e-05 , # From Simons (1980)
            'Cp': [42.63803582, 0.008971021, -260780.8155, 196.5978421], # By linear extrapolation from Fe0.9254O and hematite/3..
            'a_0': 3.39e-05 ,
            'K_0': 1.49e+11 ,
            'Kprime_0': 4.0 ,
            'Kdprime_0': -4.0/1.49e+11 ,
            'molar_mass': formula_mass(formula, atomic_masses),
            'n': sum(formula.values()),}
        Mineral.__init__(self)


class FeO_liquid (Mineral):
    def __init__(self):
        formula='Fe1.0O1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FeO liquid',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -231046.5845 ,
            'S_0': 80.49242261 ,
            'Cp': [46.12826, 0.011480597, 0., 0.],
            'V_0': 13.16e-6,
            'K_0': 128.e9 ,
            'Kprime_0': 4.,
            'Kdprime_0': -4./128.e9,
            'a_0': 3.45e-5,
            'molar_mass': formula_mass(formula, atomic_masses),
            'n': sum(formula.values()),}
        Mineral.__init__(self)

class FeO_solid_HP (Mineral):
    def __init__(self):
        formula='Fe1.0O1.0'
        formula = dictionarize_formula(formula)
        self.params = {'S_0': 144.51, 'a_0': 1.8321e-05, 'K_0': 299090000000.0, 'einstein_T': 300.11, 'Kprime_0': 3.2762, 'T_0': 1809.0, 'Kdprime_0': -1.1847e-11, 'V_0': 1.0102e-05, 'name': 'FeO solid', 'H_0': 357252.0, 'molar_mass': 0.071844, 'equation_of_state': 'hp_tmt', 'n': 2.0, 'formula': {'Fe': 1.0, 'O': 1.0}, 'Cp': [58.981, 0.0030363, 1534200.0, -247.67], 'P_0': 50000000000.0}
        Mineral.__init__(self)

class FeO_liquid_HP (Mineral):
    def __init__(self):
        formula='Fe1.0O1.0'
        formula = dictionarize_formula(formula)
        self.params = {'S_0': 165.19, 'a_0': 1.6534e-05, 'K_0': 279960000000.0, 'einstein_T': 227.82, 'Kprime_0': 3.1829, 'T_0': 1809.0, 'Kdprime_0': -1.2457e-11, 'V_0': 1.0577e-05, 'name': 'FeO liquid', 'H_0': 424324.0, 'molar_mass': 0.071844, 'equation_of_state': 'hp_tmt', 'n': 2.0, 'formula': {'Fe': 1.0, 'O': 1.0}, 'Cp': [60.545, 0.0061334, 1576000.0, -391.61], 'P_0': 50000000000.0}
        Mineral.__init__(self)
