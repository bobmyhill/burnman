from __future__ import absolute_import
# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.

"""
Pyrope-Grossular "ideal" solution (where ideality is in Helmholtz free energy)
"""

import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from excess_modelling import *

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman.minerals import KMFBZ_2017, SLB_2011, HP_2011_ds62

from scipy.optimize import brentq
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'

formula = {'Mg': 1., 'Si': 1., 'O': 3.}
MgSiO3 = burnman.Mineral(params={'name': 'MgSiO$_3$ (GGA, CC2005)', 
                                 'equation_of_state': 'bm3',
                                 'V_0': 40.78/1.e30*burnman.constants.Avogadro,
                                 'K_0': 232.e9,
                                 'Kprime_0': 3.86,
                                 'molar_mass': burnman.processchemistry.formula_mass(formula),
                                 'n': 5.,
                                 'formula': formula})

formula = {'Fe': 1., 'Si': 1., 'O': 3.}
FeSiO3 = burnman.Mineral(params={'name': 'FeSiO$_3$ (GGA, CC2005)', 
                                 'equation_of_state': 'bm3',
                                 'V_0': 44.12/1.e30*burnman.constants.Avogadro,
                                 'K_0': 237.e9,
                                 'Kprime_0': 4.16,
                                 'molar_mass': burnman.processchemistry.formula_mass(formula),
                                 'n': 5.,
                                 'formula': formula})

formula = {'Al': 1., 'Al': 1., 'O': 3.}
AlAlO3 = burnman.Mineral(params={'name': 'AlAlO$_3$ (GGA, CC2005)', 
                                 'equation_of_state': 'bm3',
                                 'V_0': 43.22/1.e30*burnman.constants.Avogadro,
                                 'K_0': 202.e9,
                                 'Kprime_0': 3.95,
                                 'molar_mass': burnman.processchemistry.formula_mass(formula),
                                 'n': 5.,
                                 'formula': formula})

P_xs = 0.e9
for m in [MgSiO3, FeSiO3, AlAlO3]:
    m.set_state(P_xs, 300.)
    print('{0}: {1} cm^3/mol, {2} GPa'.format(m.name, m.V*1.e6, m.K_T/1.e9))
