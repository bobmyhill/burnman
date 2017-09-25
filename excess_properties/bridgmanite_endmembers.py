from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman

formula = burnman.processchemistry.dictionarize_formula('MgSiO3')
mgsio3 = burnman.Mineral(params={'name': 'MgSiO$_3$',
                                 'P_0': 1.e5,
                                 'T_0': 300.,
                                 'equation_of_state': 'bm3',
                                 'V_0': 24.445e-6,
                                 'K_0': 253.e9,
                                 'Kprime_0': 3.90,
                                 'molar_mass': burnman.processchemistry.formula_mass(formula),
                                 'n': 5.,
                                 'formula': formula})

formula = burnman.processchemistry.dictionarize_formula('FeSiO3')
fesio3 = burnman.Mineral(params={'name': 'FeSiO$_3$',
                                 'P_0': 1.e5,
                                 'T_0': 300.,
                                 'equation_of_state': 'bm3',
                                 'V_0': 24.88e-6,
                                 'K_0': 251.e9,
                                 'Kprime_0': 3.90,
                                 'molar_mass': burnman.processchemistry.formula_mass(formula),
                                 'n': 5.,
                                 'formula': formula})

formula = burnman.processchemistry.dictionarize_formula('AlAlO3')
alalo3 = burnman.Mineral(params={'name': 'AlAlO$_3$',
                                 'P_0': 1.e5,
                                 'T_0': 300.,
                                 'equation_of_state': 'bm3',
                                 'V_0': 25.91e-6,
                                 'K_0': 223.e9,
                                 'Kprime_0': 4.03,
                                 'molar_mass': burnman.processchemistry.formula_mass(formula),
                                 'n': 5.,
                                 'formula': formula})

# Caracas, 2010
formula = burnman.processchemistry.dictionarize_formula('FeAlO3')
fealo3 = burnman.Mineral(params={'name': 'FeAlO$_3$',
                                 'P_0': 1.e5,
                                 'T_0': 300.,
                                 'equation_of_state': 'bm3',
                                 'V_0': 27.68e-6, # static + 0.07 cm^3/mol
                                 'K_0': 207.e9, # static - 0.07 cm^3/mol
                                 'Kprime_0': 3.73,
                                 'molar_mass': burnman.processchemistry.formula_mass(formula),
                                 'n': 5.,
                                 'formula': formula})
