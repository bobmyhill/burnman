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


figs = [plt.figure(figsize=(3, 3)) for i in range(3)]
ax = [figs[i].add_subplot(1, 1, 1) for i in range(3)]

for S_path in [525., 550., 575., 600.]:

    equality_constraints = [('phase_fraction', (diL, 0.0)),
                            ('S', S_path)]
    sol, prm = equilibrate(composition, assemblage, equality_constraints)
    print(sol.assemblage.temperature)
    print(sol.assemblage.pressure)

    path_pressures = np.linspace(10.e9, sol.assemblage.pressure, 11)
    equality_constraints = [('P', path_pressures),
                            ('S', S_path)]

    assemblage = burnman.Composite([di])
    sols, prm = equilibrate(composition, assemblage, equality_constraints)

    path_temperatures = [sol.assemblage.temperature for sol in sols]
    path_entropies = [sol.assemblage.S for sol in sols]


    path_pressures_2 = np.linspace(sols[-1].assemblage.pressure, 0., 11)
    assemblage = burnman.Composite([di, diL])
    equality_constraints = [('P', path_pressures_2),
                            ('S', S_path)]
    sols, prm = equilibrate(composition, assemblage, equality_constraints)

    path_temperatures_2 = [sol.assemblage.temperature for sol in sols]
    path_entropies_2 = [sol.assemblage.S for sol in sols]
    path_liq_fractions_2 = [sol.assemblage.molar_fractions[1] for sol in sols]


    assemblage = burnman.Composite([di, diL])
    pressures = np.linspace(1.e5, 10.e9, 101)
    equality_constraints = [('phase_fraction', (diL, np.array([0.0]))),
                            ('P', pressures)]

    sols, prm = equilibrate(composition, assemblage, equality_constraints)


    temperatures = np.array([sol.assemblage.temperature for sol in sols])
    S_sol = np.array([sol.assemblage.phases[0].S for sol in sols])
    S_liq = np.array([sol.assemblage.phases[1].S for sol in sols])
    V_sol = np.array([sol.assemblage.phases[0].V for sol in sols])
    V_liq = np.array([sol.assemblage.phases[1].V for sol in sols])

    dTdP0 = (V_sol[0] - V_liq[0]) / (S_sol[0] - S_liq[0])


    ax[0].plot(path_temperatures, path_pressures/1.e9, linestyle='--', color='red', linewidth=2)
    ax[0].plot(path_temperatures_2, path_pressures_2/1.e9, linestyle='--', color='red', linewidth=2)

    ax[1].plot(path_entropies, path_pressures/1.e9, linestyle='--', color='red', linewidth=2)
    ax[1].plot(path_entropies_2, path_pressures_2/1.e9, linestyle='--', color='red', linewidth=2)

    ax[2].plot(path_pressures*0., path_pressures/1.e9, linestyle='--', color='red', linewidth=2)
    ax[2].plot(path_liq_fractions_2, path_pressures_2/1.e9, linestyle='--', color='red', linewidth=2)

    ax[0].plot(temperatures, pressures/1.e9, color='black')
    ax[1].plot(S_sol, pressures/1.e9, color='black')
    ax[1].plot(S_liq, pressures/1.e9, color='black')


for i in range(3):
    ax[i].set_ylabel('Pressure (GPa)')
    ax[i].set_ylim(10, 0)

ax[0].set_xlabel('Temperature (K)')
ax[1].set_xlabel('Entropy (J/K/mol)')
ax[2].set_xlabel('Molar fraction liquid')

ax[0].text(1900., 6., 'solid',
           horizontalalignment='center', verticalalignment='center')
ax[0].text(2200., 3., 'liquid',
          horizontalalignment='center', verticalalignment='center')

ax[1].text(560., 8., 'solid',
           horizontalalignment='center', verticalalignment='center')
ax[1].text(675., 1., 'liquid',
          horizontalalignment='center', verticalalignment='center')

for i, fig in enumerate(figs):
    fig.set_tight_layout(True)
    fig.savefig(f'figures/di_melting_path_{i}.pdf')

plt.show()

