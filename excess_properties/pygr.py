# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.

from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import minerals

from burnman.solidsolution import SolidSolution


class pyrope_grossular(SolidSolution):

    def __init__(self, molar_fractions=None):
        self.name = 'garnet'
        self.type = 'full_subregular'
        self.n_atoms = 20.
        self.T_0 = 300.
        self.P_0 = 1.e5
        self.endmembers = [[minerals.SLB_2011.py(), '[Mg]3Al2Si3O12'],
                           [minerals.SLB_2011.gr(), '[Ca]3Al2Si3O12']]
        self.energy_interaction = [[[0., 0.]]]
        self.volume_interaction = [[[1.2e-6, 1.2e-6]]]
        SolidSolution.__init__(self, molar_fractions)



pygr = pyrope_grossular()
pygr.set_composition([0.5, 0.5])

temperatures = [300., 1000., 2000.]
pressures = np.linspace(1.e5, 100.e9, 31)
Vex = np.empty_like(pressures)
for T in temperatures:
    for i, P in enumerate(pressures):
        pygr.set_state(P, T)
        Vex[i] = pygr.excess_volume
    plt.plot(pressures/1.e9, Vex, label=str(T)+' K')


plt.legend(loc='lower right')
plt.xlabel('Pressure (GPa)')
plt.show()

pressures = [1.e5, 20.e9, 100.e9]
temperatures = np.linspace(300., 2000., 31)
Sxs = np.empty_like(temperatures)

for P in pressures:
    for i, T in enumerate(temperatures):
        pygr.set_state(P, T)
        Sxs[i] = pygr.excess_entropy
        
    plt.plot(temperatures, Sxs, label=str(P/1.e9)+' GPa')


plt.legend(loc='lower right')
plt.xlabel('Temperature (K)')
plt.show()
