# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
fo_Cp
-----
"""

from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import burnman
from burnman import equilibrate
from burnman.minerals import HGP_2018_ds633, SLB_2011


fo_SLB = SLB_2011.fo()
fo_HP = HGP_2018_ds633.fo()

Gillet_et_al_1991 = np.array([[300., 119.2],
                              [600., 158.2],
                              [900., 171.4],
                              [1200., 180.4],
                              [1500., 188.2],
                              [1800., 195.7],
                              [2000., 199.6]])

temperatures = np.linspace(10., 2000., 101)
pressures = 1.e5 * np.ones(101)
Cp_SLB = fo_SLB.evaluate(['C_p'], pressures, temperatures)[0]
Cp_HP = fo_HP.evaluate(['C_p'], pressures, temperatures)[0]

fig = plt.figure(figsize=(6, 4))
ax = [fig.add_subplot(1, 1, i) for i in range(1, 2)]


ln, = ax[0].plot(temperatures, Cp_HP, label='Polynomial approximation (Holland et al., 2018)')

ax[0].scatter(Gillet_et_al_1991.T[0], Gillet_et_al_1991.T[1], label='Experimental (Gillet et al., 1991)',
              color=ln.get_color())

ax[0].plot(temperatures, Cp_SLB, label='Debye model (Stixrude and Lithgow-Bertelloni, 2011)')

ax[0].legend()
ax[0].set_xlim(0.,)
ax[0].set_ylim(0.,200.)

ax[0].set_xlabel('Temperature (K)')
ax[0].set_ylabel('$C_p$ (J/K/mol)')
fig.set_tight_layout(True)
fig.savefig('figures/fo_Cp.pdf')
plt.show()
