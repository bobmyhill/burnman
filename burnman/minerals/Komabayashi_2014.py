# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

"""
Komabayashi_2013
^^^^^^^^^^^^^

Minerals from Komabayashi 2014 and references therein.

Note that Komabayashi reports room pressure Gibbs free energies as the 
polynomial

f + g*T + h*T*lnT + i*T^2 + j/T + k*T^0.5

As G = H_ref + intCpdT - T*S_ref - T*intCpoverTdT
Cp = a + bT + cT^-2 + dT^-0.5
intCpdT = aT + 0.5bT^2 - c/T + 2dT^0.5 - the value at T_ref
-T*intCpoverTdT = -aTlnT - bT^2 + 0.5c/T + 2dT^0.5 + the value at T_ref

Thus
f = H_ref - intCpdT(T_ref) 
g = a - S_ref + T_ref*intCpoverTdT(T_ref)
h = -a
i = 0.5b - b
j = -c + 0.5c
k = 4d

H_ref = f + intCpdT(T_ref) 
S_ref = - h - g - T_ref*intCpoverTdT(T_ref)
a = -h
b = -2i
c = -2j
d = 0.25k

"""

from burnman.mineral import Mineral
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass

atomic_masses=read_masses()

class fcc_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FCC iron',
            'formula': formula,
            'equation_of_state': 'v_ag',
            'H_0': 7839.07 ,
            'S_0': 35.8 ,
            'Cp': [52.2754, -0.000355156, 790710.86, -619.07],
            'V_0': 6.82e-6,
            'K_0': 163.4e9 ,
            'Kprime_0': 5.38 ,
            'a_0': 7.e-5,
            'delta_0': 5.5,
            'kappa': 1.4,
            'T_0': 298.,
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)
        
class hcp_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'HCP iron',
            'formula': formula,
            'equation_of_state': 'v_ag',
            'H_0': 4000. ,
            'S_0': 30.28 ,
            'Cp': [52.2754, -0.000355156, 790710.86, -619.07],
            'V_0': 6.753e-6,
            'K_0': 163.4e9 ,
            'Kprime_0': 5.38 ,
            'a_0': 5.8e-5,
            'delta_0': 5.1,
            'kappa': 1.4,
            'T_0': 298.,
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)
        
class liquid_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Liquid iron',
            'formula': formula,
            'equation_of_state': 'v_ag',
            'H_0': 4707. ,
            'S_0': 17.79 ,
            'Cp': [46., 0., 0., 0.],
            'V_0': 6.88e-6,
            'K_0': 148.e9 ,
            'Kprime_0': 5.8,
            'a_0': 9.e-5,
            'delta_0': 5.1,
            'kappa': 0.56,
            'T_0': 298.,
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)
        
class FeO_solid (Mineral):
    def __init__(self):
        formula='Fe1.0O1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FeO solid',
            'formula': formula,
            'equation_of_state': 'v_ag',
            'H_0': -265054.5845 ,
            'S_0': 59.5234,
            'Cp': [46.12826, 0.011480597, 0., 0.],
            'V_0': 12.256e-6,
            'K_0': 149.e9,
            'Kprime_0': 3.83,
            'a_0': 4.5e-5,
            'delta_0': 4.25,
            'kappa': 1.4,
            'T_0': 298.,
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class FeO_liquid (Mineral):
    def __init__(self):
        formula='Fe1.0O1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FeO liquid',
            'formula': formula,
            'equation_of_state': 'v_ag',
            'H_0': -231046.5845 ,
            'S_0': 80.49242261 ,
            'Cp': [46.12826, 0.011480597, 0., 0.],
            'V_0': 13.16e-6,
            'K_0': 128.e9 ,
            'Kprime_0': 3.85,
            'a_0': 4.7e-5,
            'delta_0': 4.5,
            'kappa': 1.4,
            'T_0': 298.,
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)
