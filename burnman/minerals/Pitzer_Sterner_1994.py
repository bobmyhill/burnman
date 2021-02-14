# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
Pitzer_Sterner_1994
"""

from __future__ import absolute_import
import numpy as np

from ..mineral import Mineral
from ..processchemistry import dictionarize_formula, formula_mass


class H2O_Pitzer_Sterner (Mineral):

    def __init__(self):
        formula = 'H2O'
        formula = dictionarize_formula(formula)
        self.params = {'name': 'H2O',
                       'equation_of_state': 'pitzer-sterner',
                       'formula': formula,
                       'F_0': -249677.75163034006,  # Fit to Barin at 1.e5 Pa, 2000 K
                       'Cv_0': 8.31446*2.25,  # Eyeball fit on Barin heat capacity
                       'Debye_0': 3800, # Eyeball fit on Barin entropy
                       'Debye_n': 1.2, # Eyeball fit on Barin entropy
                       'c_coeffs': np.array([[0., 0., +0.24657688e6, +0.51359951e+2, 0., 0.],
                                             [0., 0., +0.58638965e+0, -0.28646939e-2, +0.31375577e-4, 0.],
                                             [0., 0., -0.62783840e+1, +0.14791599e-1, +0.35779579E-3, +0.15432925e-7],
                                             [0., 0., 0., -0.42719875e+0, -0.16325155e-4, 0.],
                                             [0., 0., +0.56654978e+4, -0.16580167e+2, +0.76560762e-1 , 0.],
                                             [0., 0., 0., +0.10917883e+0, 0., 0.],
                                             [+0.38878656e+13, -0.13494878e+9, +0.30916564e+6, +0.75591105e+1, 0., 0.],
                                             [0., 0., -0.65537898e+5, +0.18810675e+3, 0., 0.],
                                             [-0.14182435e+14, +0.18165390e+9, -0.19769068e+6, -0.23530318e+2, 0., 0.],
                                             [0., 0., +0.92093375e+5, +0.12246777e+3, 0., 0.]]),
                       'n': sum(formula.values()),
                       'molar_mass': formula_mass(formula)}

        Mineral.__init__(self)
