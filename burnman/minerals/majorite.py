# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

"""
majorite
Minerals from Stixrude & Lithgow-Bertelloni 2011, modified by 
Dan Frost, Tiziana Boffa Ballaran and Bob Myhill
"""

from burnman.mineral import Mineral
from burnman.solidsolution import SolidSolution
from burnman.solutionmodel import *
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass

atomic_masses=read_masses()

'''
SOLID SOLUTIONS
'''

class garnet(SolidSolution):
    def __init__(self, molar_fractions=None):
        # Name
        self.name='garnet'

        # Endmembers (garnet is symmetric)
        endmembers = [[pyrope(), '[Mg]3[Al][Al]Si3O12'], \
                          [almandine(), '[Fe]3[Al][Al]Si3O12'], \
                          [grossular(), '[Ca]3[Al][Al]Si3O12'], \
                          [mg_majorite_cubic(), '[Mg]3[Mg][Si]Si3O12']]

        # Interaction parameters [(py alm), (py gr), (py maj)], [(alm gr), (alm maj)], [(gr maj)]
        enthalpy_interaction=[[[2117., 695.], [9834., 21627.], [0., 0.]],[[6773., 873.],[0., 0.]],[[0., 0.]]]
        volume_interaction=[[[0.07e-5, 0.], [0.058e-5, 0.012e-5], [0., 0.]],[[0.03e-5, 0.],[0., 0.]],[[0., 0.]]]
        entropy_interaction=[[[0., 0.], [5.78, 5.78], [0., 0.]],[[1.69, 1.69],[0., 0.]],[[0., 0.]]]

        # Published values are on a 4-oxygen (1-cation) basis
        for interaction in [enthalpy_interaction, volume_interaction, entropy_interaction]:
            for i in range(len(interaction)):
                for j in range(len(interaction[i])):
                    for k in range(len(interaction[i][j])):
                        interaction[i][j][k]*=3.

        burnman.SolidSolution.__init__(self, endmembers, \
                                           burnman.solutionmodel.SubregularSolution(endmembers, enthalpy_interaction, volume_interaction, entropy_interaction), molar_fractions)

class ideal_garnet(SolidSolution):
    def __init__(self, molar_fractions=None):
        # Name
        self.name='ideal garnet'

        # Endmembers (garnet is symmetric)
        endmembers = [[pyrope(), '[Mg]3[Al][Al]Si3O12'], \
                          [almandine(), '[Fe]3[Al][Al]Si3O12'], \
                          [grossular(), '[Ca]3[Al][Al]Si3O12'], \
                          [mg_majorite_cubic(), '[Mg]3[Mg][Si]Si3O12']]

        burnman.SolidSolution.__init__(self, endmembers, IdealSolution(endmembers), molar_fractions)

"""
ENDMEMBERS
"""

class pyrope (Mineral):
    def __init__(self):
        formula='Mg3Al2Si3O12'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Pyrope',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -5936000.0 , # unchanged
            'V_0': 0.00011308 , #
            'K_0': 1.71e+11 , #
            'Kprime_0': 4.2 , #
            'Debye_0': 804.0 , #
            'grueneisen_0': 1.01 , #
            'q_0': 1.4 , #
            'G_0': 94.e9 , #
            'Gprime_0': 1.4 , #
            'eta_s_0': 1.2 , #
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        self.uncertainties = {
            'err_F_0': 10000.0 ,
            'err_V_0': 0.0 ,
            'err_K_0': 2000000000.0 ,
            'err_K_prime_0': 0.3 ,
            'err_Debye_0': 4.0 ,
            'err_grueneisen_0': 0.06 ,
            'err_q_0': 0.5 ,
            'err_G_0': 2000000000.0 ,
            'err_Gprime_0': 0.2 ,
            'err_eta_s_0': 0.3 }
        Mineral.__init__(self)

class almandine (Mineral):
    def __init__(self):
        formula='Fe3Al2Si3O12'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Almandine',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -4935000.0 , # unchanged
            'V_0': 0.00011543 , # 
            'K_0': 1.75e+11 , # 
            'Kprime_0': 3.7 , # 
            'Debye_0': 741.0 , # 
            'grueneisen_0': 1.06 , # 
            'q_0': 1.4 , # 
            'G_0': 96.e9 , # 
            'Gprime_0': 1.1 , # 
            'eta_s_0': 2.1 , #
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        self.uncertainties = {
            'err_F_0': 29000.0 ,
            'err_V_0': 0.0 ,
            'err_K_0': 2000000000.0 ,
            'err_K_prime_0': 0.2 ,
            'err_Debye_0': 5.0 ,
            'err_grueneisen_0': 0.06 ,
            'err_q_0': 1.0 ,
            'err_G_0': 1000000000.0 ,
            'err_Gprime_0': 0.1 ,
            'err_eta_s_0': 1.0 }
        Mineral.__init__(self)

class grossular (Mineral):
    def __init__(self):
        formula='Ca3Al2Si3O12'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Grossular',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -6278000.0 , # unchanged
            'V_0': 0.00012512 , # 
            'K_0': 1.67e+11 , # 
            'Kprime_0': 3.9 , # 
            'Debye_0': 823.0 , # 
            'grueneisen_0': 1.05 , # 
            'q_0': 1.9 , # 
            'G_0': 1.09e+11 , # 
            'Gprime_0': 1.2 , # 
            'eta_s_0': 2.4 , # 
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        self.uncertainties = {
            'err_F_0': 11000.0 ,
            'err_V_0': 0.0 ,
            'err_K_0': 1000000000.0 ,
            'err_K_prime_0': 0.2 ,
            'err_Debye_0': 2.0 ,
            'err_grueneisen_0': 0.06 ,
            'err_q_0': 0.2 ,
            'err_G_0': 4000000000.0 ,
            'err_Gprime_0': 0.1 ,
            'err_eta_s_0': 0.1 }
        Mineral.__init__(self)

class mg_majorite_cubic (Mineral):
    def __init__(self):
        formula='Mg4Si4O12'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Cubic Mg majorite',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -5691000.0 , # unchanged
            'V_0': 0.00011397 , # 
            'K_0': 1.60e+11 , # 
            'Kprime_0': 5.6 , # 
            'Debye_0': 779.0 , # 
            'grueneisen_0': 0.98 , # 
            'q_0': 1.5 , # 
            'G_0': 86.e9 , # 
            'Gprime_0': 1.4 , # 
            'eta_s_0': 1.4 , # 
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        self.uncertainties = {
            'err_F_0': 10000.0 ,
            'err_V_0': 0.0 ,
            'err_K_0': 3000000000.0 ,
            'err_K_prime_0': 0.3 ,
            'err_Debye_0': 4.0 ,
            'err_grueneisen_0': 0.07 ,
            'err_q_0': 0.5 ,
            'err_G_0': 2000000000.0 ,
            'err_Gprime_0': 0.2 ,
            'err_eta_s_0': 0.3 }
        Mineral.__init__(self)

# Garnet group
py = pyrope
al = almandine
gr = grossular
mgmj = mg_majorite_cubic
