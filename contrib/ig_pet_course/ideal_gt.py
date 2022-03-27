# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
ideal_gt
--------
"""

from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt

import burnman
from burnman import equilibrate
from burnman.minerals import HGP_2018_ds633


py = HGP_2018_ds633.py()
alm = HGP_2018_ds633.alm()
gr = HGP_2018_ds633.gr()
andr = HGP_2018_ds633.andr()

# A super simple an-ab melt model (following White et al., 2007)
gt = burnman.SolidSolution(name='ideal gt',
                           solution_type='ideal',
                           endmembers=[[py, '[Mg]3[Al]2Si3O12'],
                                       [alm, '[Fe]3[Al]2Si3O12'],
                                       [gr, '[Ca]3[Al]2Si3O12'],
                                       [andr, '[Ca]3[Fe]2Si3O12']])


gt.set_composition([0.6, 0.3, -0.1, 0.2])

gt.set_state(1.e5, 800.)

print(gt.partial_gibbs[0] - py.gibbs)
print(burnman.constants.gas_constant*800.*(3.*np.log(0.6) + 2.*np.log(0.8)))
