# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
Pyrope-Grossular "ideal" solution (where ideality is in Helmholtz free energy)
"""
from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from excess_modelling import *

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import minerals

from scipy.optimize import brentq, curve_fit

py = minerals.HP_2011_ds62.py()
alm = minerals.HP_2011_ds62.alm()
gr = minerals.HP_2011_ds62.gr()
andr = minerals.HP_2011_ds62.andr()

pressures = np.linspace(1.e5, 30.e9, 101)
temperatures = np.array([300.]*len(pressures))

for m in [py, alm, gr, andr]:
    plt.plot(pressures, m.evaluate(['V'], pressures, temperatures)[0], label=m.name)

plt.legend(loc='best')
plt.show()

