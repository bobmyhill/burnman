# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
fo_melting
----------
"""

from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import burnman
from burnman import equilibrate
from burnman.minerals import HGP_2018_ds633



fo = HGP_2018_ds633.fo()
foL = HGP_2018_ds633.foL()

assemblage = burnman.Composite([fo, foL])
assemblage.set_state(10.e9, 2000.)
composition = foL.formula

pressures = np.array([2.e9, 4.e9, 6.e9, 8.e9])
equality_constraints = [('phase_fraction', (foL, 0.0)),
                        ('P', pressures[::-1])]
sols, prm = equilibrate(composition, assemblage, equality_constraints)
S_paths = [sol.assemblage.S for sol in sols]


figs = [plt.figure(figsize=(3, 3)) for i in range(3)]
ax = [figs[i].add_subplot(1, 1, 1) for i in range(3)]

colors = ['red', 'orange', 'blue', 'purple']

for i, S_path in enumerate(S_paths):
    assemblage.set_state(10.e9, 2500.)
    equality_constraints = [('phase_fraction', (foL, 0.0)),
                            ('S', S_path)]
    sol, prm = equilibrate(composition, assemblage, equality_constraints)
    print(sol.assemblage.temperature)
    print(sol.assemblage.pressure/1.e9)

    path_pressures = np.linspace(10.e9, sol.assemblage.pressure, 11)
    equality_constraints = [('P', path_pressures),
                            ('S', S_path)]

    assemblage = burnman.Composite([fo])
    assemblage.set_state(10.e9, 2500.)
    sols, prm = equilibrate(composition, assemblage, equality_constraints)

    path_temperatures = [sol.assemblage.temperature for sol in sols]
    path_entropies = [sol.assemblage.S for sol in sols]


    path_pressures_2 = np.linspace(sols[-1].assemblage.pressure, 0., 11)
    assemblage = burnman.Composite([fo, foL])
    assemblage.set_state(10.e9, 2500.)
    equality_constraints = [('P', path_pressures_2),
                            ('S', S_path)]
    sols, prm = equilibrate(composition, assemblage, equality_constraints)

    path_temperatures_2 = [sol.assemblage.temperature for sol in sols]
    path_entropies_2 = [sol.assemblage.S for sol in sols]
    path_liq_fractions_2 = [sol.assemblage.molar_fractions[1] for sol in sols]


    assemblage = burnman.Composite([fo, foL])
    assemblage.set_state(10.e9, 2500.)
    pressures = np.linspace(1.e5, 10.e9, 101)
    equality_constraints = [('phase_fraction', (foL, np.array([0.0]))),
                            ('P', pressures)]

    sols, prm = equilibrate(composition, assemblage, equality_constraints)

    temperatures = np.array([sol.assemblage.temperature for sol in sols])
    S_sol = np.array([sol.assemblage.phases[0].S for sol in sols])
    S_liq = np.array([sol.assemblage.phases[1].S for sol in sols])
    V_sol = np.array([sol.assemblage.phases[0].V for sol in sols])
    V_liq = np.array([sol.assemblage.phases[1].V for sol in sols])

    dTdP0 = (V_sol[0] - V_liq[0]) / (S_sol[0] - S_liq[0])


    ax[0].plot(path_temperatures, path_pressures/1.e9, linestyle='--', color=colors[i], linewidth=2)
    ax[0].plot(path_temperatures_2, path_pressures_2/1.e9, linestyle='--', color=colors[i], linewidth=2)

    ax[1].plot(path_entropies, path_pressures/1.e9, linestyle='--', color=colors[i], linewidth=2)
    ax[1].plot(path_entropies_2, path_pressures_2/1.e9, linestyle='--', color=colors[i], linewidth=2)

    ax[2].plot(path_pressures*0., path_pressures/1.e9, linestyle='--', color=colors[i], linewidth=2)
    ax[2].plot(path_liq_fractions_2, path_pressures_2/1.e9, linestyle='--', color=colors[i], linewidth=2)

    ax[0].plot(temperatures, pressures/1.e9, color='black', zorder=-1)
    ax[1].plot(S_sol, pressures/1.e9, color='black')
    ax[1].plot(S_liq, pressures/1.e9, color='black')


for i in range(3):
    ax[i].set_ylabel('Pressure (GPa)')
    ax[i].set_ylim(10, 0)

ax[0].set_xlabel('Temperature (K)')
ax[1].set_xlabel('Entropy (J/K/mol)')
ax[2].set_xlabel('Molar fraction liquid')

ax[0].set_xlim(2100., 2600.)
ax[0].text(2250., 6., 'solid',
           horizontalalignment='center', verticalalignment='center')
ax[0].text(2500., 3., 'liquid',
          horizontalalignment='center', verticalalignment='center')

ax[1].set_xlim(400., 480.)
ax[1].text(410., 8., 'solid',
           horizontalalignment='center', verticalalignment='center')
ax[1].text(455., 5., 'solid+liquid',
          horizontalalignment='center', verticalalignment='center')

for i, fig in enumerate(figs):
    fig.set_tight_layout(True)
    fig.savefig(f'figures/fo_melting_path_{i}.pdf')

plt.show()

