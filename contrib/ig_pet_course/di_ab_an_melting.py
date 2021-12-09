# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
an_di_melting
-------------
"""

from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt

import burnman
from burnman import equilibrate
from burnman.minerals import HGP_2018_ds633
from burnman.minerals import JH_2015
from collections import Counter
import ternary


diL = HGP_2018_ds633.diL()
anL = HGP_2018_ds633.anL()
abL = HGP_2018_ds633.abL()
di = HGP_2018_ds633.di()
plag = JH_2015.plagioclase()

# A super simple ideal model
liq = burnman.SolidSolution(name='di-an-ab liquid',
                            solution_type='symmetric',
                            endmembers=[[diL, '[Mg]1'],
                                        [anL, '[Ca]1'],
                                        [abL, '[Na]1']],
                            energy_interaction=[[-15.e3, 0.],
                                                [-15.e3]])

xs = np.linspace(0.01, 0.99, 11)
eut_temperatures = np.empty_like(xs)
lines = []
for i, x in enumerate(xs):
    plag.set_composition([x, 1. - x])

    total = sum(plag.formula.values())
    fplag = Counter({k: v / total for k, v in plag.formula.items()})

    total = sum(di.formula.values())
    fdi = Counter({k: -v / total for k, v in di.formula.items()})

    d = fplag.copy()
    d.update(fdi)
    free_compositional_vectors = [d]

    assemblage = burnman.Composite([di, plag, liq])
    assemblage.set_state(1.e5, 1473.15)

    f = 0.3
    liq.set_composition([f, x*(1.-f), (1. - x)*(1.-f)])
    composition = liq.formula
    assemblage = burnman.Composite([liq, di, plag])
    equality_constraints = [('phase_fraction', (di, np.array([0.0]))),
                            ('phase_fraction', (plag, np.array([0.0]))),
                            ('P', 1.e5)]

    sol, prm = equilibrate(composition, assemblage, equality_constraints,
                           free_compositional_vectors,
                           verbose=False)
    
    eut_temperatures[i] = sol.assemblage.temperature

    xan, xab = sol.assemblage.phases[2].molar_fractions

    line = [sol.assemblage.phases[0].molar_fractions,
            [1., 0., 0.],
            [0., xan, xab],
            sol.assemblage.phases[0].molar_fractions]

    line = np.array(line)
    line[:, [0, 1]] = line[:, [1, 0]]
    lines.append(line)

lines = np.array(lines)


# step 2, plot liquid line of descent for batch crystallisation
liq.set_composition([0.1, 0.55, 0.35])
composition = liq.formula
assemblage = burnman.Composite([plag, liq])
equality_constraints = [('phase_fraction', (plag, np.array([0.0]))),
                        ('P', 1.e5)]

sol, prm = equilibrate(composition, assemblage, equality_constraints,
                       verbose=False)
T_liq = sol.assemblage.temperature
X_liq = liq.molar_fractions

assemblage = burnman.Composite([di, plag, liq])
equality_constraints = [('phase_fraction', (di, np.array([0.0]))),
                        ('P', 1.e5)]

sol, prm = equilibrate(composition, assemblage, equality_constraints,
                       verbose=False)
T_eut = sol.assemblage.temperature
X_eut = liq.molar_fractions

assemblage = burnman.Composite([di, plag, liq])
equality_constraints = [('phase_fraction', (liq, np.array([0.0]))),
                        ('P', 1.e5)]

sol, prm = equilibrate(composition, assemblage, equality_constraints,
                       verbose=False)
T_sol = sol.assemblage.temperature
X_sol = liq.molar_fractions

temperatures = np.linspace(T_liq, T_eut, 11)
assemblage = burnman.Composite([plag, liq])
equality_constraints = [('T', temperatures),
                        ('P', 1.e5)]

sols, prm = equilibrate(composition, assemblage, equality_constraints,
                        verbose=False)
liq_compositions1 = np.array([sol.assemblage.phases[1].molar_fractions
                              for sol in sols])
liq_compositions1[:, [0, 1]] = liq_compositions1[:, [1, 0]]

temperatures = np.linspace(T_eut, T_sol, 11)
assemblage = burnman.Composite([di, plag, liq])
equality_constraints = [('T', temperatures),
                        ('P', 1.e5)]

sols, prm = equilibrate(composition, assemblage, equality_constraints,
                        verbose=False)
liq_compositions2 = np.array([sol.assemblage.phases[2].molar_fractions
                              for sol in sols])
liq_compositions2[:, [0, 1]] = liq_compositions2[:, [1, 0]]


# Step 3: Fractional crystallisation
dT = 1.
liq.set_composition(X_liq)
composition = liq.formula

T = T_liq
process = True

liq_compositions_frac = [liq.molar_fractions]

while process:
    print(f'{T:.0f} K', end='\r', flush=True)

    assemblage = burnman.Composite([di, plag, liq])
    equality_constraints = [('phase_fraction', (di, np.array([0.0]))),
                            ('P', 1.e5)]
    sol, prm = equilibrate(composition, assemblage,
                           equality_constraints,
                           verbose=False)

    T_eut_frac = sol.assemblage.temperature

    if T > T_eut_frac + dT:
        T -= dT
    else:
        T = T_eut_frac
        process = False
    assemblage = burnman.Composite([plag, liq])
    equality_constraints = [('T', T), ('P', 1.e5)]
    sol, prm = equilibrate(composition, assemblage,
                           equality_constraints,
                           verbose=False)
    liq_compositions_frac.append(liq.molar_fractions)
    composition = liq.formula

T_eut_frac = T_eut_frac
X_eut_frac = liq_compositions_frac[-1]

process = True

while process:
    print(f'{T:.0f} K', end='\r', flush=True)
    assemblage = burnman.Composite([di, plag, liq])
    equality_constraints = [('phase_fraction', (liq, np.array([0.0]))),
                            ('P', 1.e5)]
    sol, prm = equilibrate(composition, assemblage,
                           equality_constraints,
                           verbose=False)

    T_sol_frac = sol.assemblage.temperature

    if T > T_sol_frac + dT:
        T -= dT
    else:
        T = T_sol_frac
        process = False
    assemblage = burnman.Composite([di, plag, liq])
    equality_constraints = [('T', T), ('P', 1.e5)]
    sol, prm = equilibrate(composition, assemblage,
                           equality_constraints,
                           verbose=False)
    liq_compositions_frac.append(liq.molar_fractions)
    composition = liq.formula


T_sol_frac = T_sol_frac
X_sol_frac = liq_compositions_frac[-1]


liq_compositions_frac = np.array(liq_compositions_frac)
liq_compositions_frac[:, [0, 1]] = liq_compositions_frac[:, [1, 0]]

figure = plt.figure()
ax = [figure.add_subplot(1, 1, 1)]
tax = ternary.TernaryAxesSubplot(ax=ax[0])

tax.boundary()
fontsize = 14
offset = 0.30
tax.top_corner_label("di", fontsize=fontsize, offset=0.2)
tax.left_corner_label("ab", fontsize=fontsize, offset=offset)
tax.right_corner_label("an", fontsize=fontsize, offset=offset)

tax.plot(lines[:, 0, :], linewidth=1.0, label="eutectic")

# Plot the data
for i, line in enumerate(lines[2:5]):
    tax.plot(line, linewidth=1.0,
             color='grey', label=f"eutectic coexistence at {eut_temperatures[i+2]:.0f} K")


tax.plot(liq_compositions1, linewidth=2.0,
         color='blue', label="liquid line of descent (batch)")
tax.plot(liq_compositions2, linewidth=2.0,
         color='blue')
tax.plot(liq_compositions_frac, linewidth=2.0,
         color='red', label="liquid line of descent (frac.)")

tax.ticks(axis='lbr', multiple=0.2, linewidth=1,
          offset=0.025, tick_formats="%.1f")

tax.scatter([[X_liq[1], X_liq[0], X_liq[2]]], color='red',
            label=f'bulk (liquidus at {T_liq:.0f} K)')
tax.scatter([[X_eut[1], X_eut[0], X_eut[2]]], marker='+', color='blue',
            label=f'eutectic liquid (batch) at {T_eut:.0f} K')
tax.scatter([[X_eut_frac[1], X_eut_frac[0], X_eut_frac[2]]], marker='+', color='red',
            label=f'eutectic liquid (frac.) at {T_eut_frac:.0f} K')
tax.scatter([[X_sol[1], X_sol[0], X_sol[2]]], marker='*', color='blue',
            label=f'last liquid (batch) at {T_sol:.0f} K')
tax.scatter([[X_sol_frac[1], X_sol_frac[0], X_sol_frac[2]]],
            marker='*', color='red',
            label=f'last liquid (frac.) at {T_sol_frac:.0f} K')

tax.legend()

tax.get_axes().axis('off')
tax.clear_matplotlib_ticks()
tax.get_axes().set_aspect(1)
tax._redraw_labels()
figure.savefig('figures/di_ab_an_melting.pdf')
tax.show()
