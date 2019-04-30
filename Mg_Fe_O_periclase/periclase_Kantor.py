# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import Mineral, minerals
from burnman.processchemistry import dictionarize_formula, formula_mass

per = minerals.SLB_2011.periclase()

pressures = np.linspace(1.e5, 1.e10, 101)
temperatures = np.empty_like(pressures)

def delta_entropy(T, P, S0, m):
    m.set_state(P, T)
    return S0 - m.S

for T0 in [300., 800., 1300.]:
    per.set_state(1.e5, T0)
    S0 = per.S
    for i, P in enumerate(pressures):
        temperatures[i] = fsolve(delta_entropy, T0, args=(P, S0, per))[0]

    plt.plot(pressures/1.e9, temperatures, label='{0} K'.format(T0))
plt.xlabel('Pressure (GPa)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.show()

