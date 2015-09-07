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

class SiO2_liquid (Mineral):
    def __init__(self):
        formula='Si1.0O2.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'SiO2 liquid',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'T_0': 1996., # Barin
            'H_0': -781669., # Barin
            'S_0': 172.626, # Barin
            'Cp': [85.772, 0., 0., 0.], # Barin
            'V_0': 27.39e-6, # 27.3e-6 at 1673 K (Bockris et al., 1956)
            'K_0': 13.7e9 , # To fit
            'Kprime_0': 5.3, # To fit
            'Kdprime_0': -5.3/13.7e9, # To fit
            'a_0': 1.0e-05, # Lange and Carmichael, 1987
            'molar_mass': formula_mass(formula, atomic_masses),
            'n': sum(formula.values()),}
        Mineral.__init__(self)

class stv (Mineral):
    def __init__(self):
        formula='Si1.0O2.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'stv',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -885000.0 , # was -876390.0
            'S_0': 24.0 ,
            'V_0': 1.401e-05 ,
            'Cp': [68.1, 0.00601, -1978200.0, -82.1] ,
            'a_0': 1.58e-05 ,
            'K_0': 3.09e+11 ,
            'Kprime_0': 4.6 ,
            'Kdprime_0': -1.5e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)
