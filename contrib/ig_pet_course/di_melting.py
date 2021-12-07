# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
di_melting
----------
"""

from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import burnman
from burnman import equilibrate
from burnman.minerals import HGP_2018_ds633



di = HGP_2018_ds633.di()
diL = HGP_2018_ds633.diL()

assemblage = burnman.Composite([di, diL])

composition = diL.formula
assemblage = burnman.Composite([di, diL])
pressures = np.linspace(1.e5, 5.e9, 101)
equality_constraints = [('phase_fraction', (di, np.array([0.0]))),
                        ('P', pressures)]

sols, prm = equilibrate(composition, assemblage, equality_constraints)

temperatures = np.array([sol.assemblage.temperature for sol in sols])


PGPa = 2.5

fig = plt.figure(figsize=(8, 4))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

ax[0].plot(temperatures, pressures/1.e9)

temperatures = np.linspace(1600., 2200., 101)
pressures = temperatures*0. + PGPa*1e9
ax[0].plot(temperatures, pressures/1.e9, linestyle=':')

di_gibbs = di.evaluate(['gibbs'], pressures, temperatures)[0]
diL_gibbs = diL.evaluate(['gibbs'], pressures, temperatures)[0]
ax[1].plot(temperatures, di_gibbs/1.e3, label='solid di')
ax[1].plot(temperatures, diL_gibbs/1.e3, label='liquid di')

ax[0].text(1700., 3.5, 'solid')
ax[0].text(2000., 1.5, 'liquid')
ax[1].text(1700., di_gibbs[-1]/1.e3, f'{PGPa} GPa')

for i in range(2):
    ax[i].set_xlim(1600., 2200.)
    ax[i].set_xlabel('Temperature (K)')

ax[0].set_ylim(0, 5)
ax[0].set_ylabel('Pressure (GPa)')
ax[1].set_ylabel('Gibbs (kJ/mol)')
ax[1].legend()

fig.set_tight_layout(True)
fig.savefig('figures/unary_melting_gibbs.pdf')
plt.show()



S_sol = np.array([sol.assemblage.phases[0].S for sol in sols])
S_liq = np.array([sol.assemblage.phases[1].S for sol in sols])
V_sol = np.array([sol.assemblage.phases[0].V for sol in sols])
V_liq = np.array([sol.assemblage.phases[1].V for sol in sols])

dTdP0 = (V_sol[0] - V_liq[0]) / (S_sol[0] - S_liq[0])


fig = plt.figure(figsize=(12, 4))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]
ax[0].plot(pressures/1.e9, temperatures)
ax[1].plot(pressures/1.e9, S_liq - S_sol)
ax[2].plot(pressures/1.e9, (V_liq - V_sol)*1.e6)

ax[0].plot(pressures/1.e9, temperatures[0] + (dTdP0 * pressures), linestyle=':')

for i in range(3):
    ax[i].set_xlim(0.,)
    ax[i].set_ylim(0.,)
    ax[i].set_xlabel('Pressure (GPa)')


ax[1].set_ylim(0.,100.)

ax[0].set_ylabel('Temperature (K)')
ax[1].set_ylabel('Entropy (J/K/mol)')
ax[2].set_ylabel('Volume (cm$^3$/mol)')

ax[0].set_ylim(1500.,)

fig.set_tight_layout(True)
fig.savefig('figures/di_melting.pdf')
plt.show()
