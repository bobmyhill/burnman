# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""

isotropic
---------

"""
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tools import print_table_for_mineral_constants

import burnman_path  # adds the local burnman directory to the path
import burnman

from anisotropicmineral import AnisotropicMineral

assert burnman_path  # silence pyflakes warning

per = burnman.minerals.SLB_2011.periclase()
per.set_state(1.e5, 300.)
a = np.cbrt(per.params['V_0'])
cell_parameters = np.array([a, a, a, 90, 90, 90])

beta_RT = per.beta_T

"""
In an isotropic mineral, there are two independent parameters.

S11, S44
S12 = S11 - S44/2

In addition, we have
3 S11 + 6 S12 = beta_T
9 S11 - 3 S44 = beta_T

In addition, if G is defined, we have:
G = 15 / (12 S11 + 9 S44 - 12 S12 )
  = 1 / S44 (Reuss bound)
"""

S44 = 1./per.G
S11 = (per.beta_T + 3.*S44)/9.
S12 = S11 - S44/2.

constants = np.zeros((6, 6, 2, 1))
constants[:3, :3, 1, 0] = S12 / per.beta_T

for i in range(3):
    constants[i, i, 1, 0] = S11 / per.beta_T
    constants[i+3, i+3, 1, 0] = S44 / per.beta_T

P = 1.e9
T = 1000.
per = AnisotropicMineral(per, cell_parameters, constants)
per.set_state(P, T)

print(per.grueneisen_tensor)

per = burnman.minerals.SLB_2011.periclase()
per.set_state(P, T)

print(per.grueneisen_parameter)
