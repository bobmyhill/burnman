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

T_contours_di = np.linspace(1150.+273.15, 1350.+273.15, 5)
T_contours_plag = np.linspace(1150.+273.15, 1500.+273.15, 8)
xs = np.linspace(0.00001, 0.99999, 501)

X_liq_contours_di = np.nan * np.empty((len(T_contours_di), len(xs), 3))
X_liq_contours_plag = np.nan * np.empty((len(T_contours_plag), len(xs), 3))

eut_temperatures = np.empty_like(xs)
lines = []
for i, x in enumerate(xs):
    print(i)
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

    X_liq = liq.molar_fractions
    X_plag = plag.molar_fractions
    
    assemblage = burnman.Composite([liq, di])
    for j, T in enumerate(T_contours_di):
        equality_constraints = [('phase_fraction', (di, np.array([0.0]))),
                                ('T', T),
                                ('P', 1.e5)]

        sol, prm = equilibrate(composition, assemblage, equality_constraints,
                                free_compositional_vectors,
                                verbose=False)
        if sol.success and liq.molar_fractions[0] > X_liq[0]:
            X_liq_contours_di[j, i, 0] = sol.assemblage.phases[0].molar_fractions[1]
            X_liq_contours_di[j, i, 1] = sol.assemblage.phases[0].molar_fractions[0]
            X_liq_contours_di[j, i, 2] = sol.assemblage.phases[0].molar_fractions[2]
            
    assemblage = burnman.Composite([liq, plag])
    for j, T in enumerate(T_contours_plag):
        # liq.set_composition([0.1, 0.1, 0.8])
        # plag.set_composition([0.1, 0.9])
        try:
            equality_constraints = [('phase_fraction', (plag, np.array([0.]))),
                                    ('T', T),
                                    ('P', 1.e5)]

            sol, prm = equilibrate(composition, assemblage, equality_constraints,
                                   free_compositional_vectors,
                                   verbose=False)
            
            if sol.success and liq.molar_fractions[0] < X_liq[0]:
                X_liq_contours_plag[j, i, 0] = sol.assemblage.phases[0].molar_fractions[1]
                X_liq_contours_plag[j, i, 1] = sol.assemblage.phases[0].molar_fractions[0]
                X_liq_contours_plag[j, i, 2] = sol.assemblage.phases[0].molar_fractions[2]
        except:
            pass

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
X_first_plag = [plag.molar_fractions[0], 0., plag.molar_fractions[1]]

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

plag_compositions2 = np.zeros_like(liq_compositions2)
plag_compositions2[:, 0] = [sol.assemblage.phases[1].molar_fractions[0]
                            for sol in sols]
plag_compositions2[:, 2] = [sol.assemblage.phases[1].molar_fractions[1]
                            for sol in sols]

lines_batch = np.zeros((len(plag_compositions2), 4, 3))
lines_batch[:, 0, 1] = 1.  # diopside
lines_batch[:, 1, :] = liq_compositions2
lines_batch[:, 2, :] = plag_compositions2
lines_batch[:, 3, 1] = 1.  # diopside

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
plag_compositions_frac = [np.array([plag.molar_fractions[0], 0.,
                                    plag.molar_fractions[1]])]

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
    plag_compositions_frac.append(np.array([plag.molar_fractions[0], 0.,
                                            plag.molar_fractions[1]]))
    composition = liq.formula


plag_compositions_frac = np.array(plag_compositions_frac)
liq_compositions_frac = np.array(liq_compositions_frac)
liq_compositions_frac[:, [0, 1]] = liq_compositions_frac[:, [1, 0]]

T_sol_frac = T_sol_frac
X_sol_frac = liq_compositions_frac[-1]

lines_frac = np.zeros((len(plag_compositions_frac), 4, 3))
lines_frac[:, 0, 1] = 1.  # diopside
lines_frac[:, 1, :] = liq_compositions_frac[-len(plag_compositions_frac):, :]
lines_frac[:, 2, :] = plag_compositions_frac
lines_frac[:, 3, 1] = 1.  # diopside


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
# for i, line in enumerate(lines[2:5]):
#    tax.plot(line, linewidth=1.0,
#             color='grey', label=f"eutectic coexistence at {eut_temperatures[i+2]:.0f} K")

for i in range(len(T_contours_di)):
    tax.plot(X_liq_contours_di[i], linewidth=1.0,
             color='purple')
for i in range(len(T_contours_plag)):
    tax.plot(X_liq_contours_plag[i], linewidth=1.0,
             color='purple')
    
tax.plot(lines_batch[0], linewidth=1.0,
         color='grey', label=f"eutectic coexistence at {T_eut:.0f} K")
tax.plot(lines_batch[-1], linewidth=1.0,
         color='grey', label=f"eutectic coexistence at {T_sol:.0f} K")
tax.plot(lines_frac[0], linewidth=1.0,
         color='grey', label=f"eutectic coexistence at {T_eut_frac:.0f} K")

tax.scatter([[X_liq[1], X_liq[0], X_liq[2]]], color='red',
            label=f'bulk (liquidus at {T_liq:.0f} K)')
tax.plot([X_first_plag, liq_compositions1[0]],
         color='orange', label=f"plag-liq coexistence at {T_liq:.0f} K)")

tax.plot(liq_compositions1, linewidth=2.0,
         color='blue', label="liquid line of descent (batch)")
tax.plot(liq_compositions2, linewidth=2.0,
         color='blue')
tax.plot(liq_compositions_frac, linewidth=2.0,
         color='red', label="liquid line of descent (frac.)")

tax.ticks(axis='lbr', multiple=0.2, linewidth=1,
          offset=0.025, tick_formats="%.1f")

tax.scatter([[X_eut[1], X_eut[0], X_eut[2]]], marker='+', color='blue',
            label=f'eutectic liquid (batch) at {T_eut:.0f} K')
tax.scatter([[X_eut_frac[1], X_eut_frac[0], X_eut_frac[2]]], marker='+', color='red',
            label=f'eutectic liquid (frac.) at {T_eut_frac:.0f} K')
tax.scatter([[X_sol[1], X_sol[0], X_sol[2]]], marker='*', color='blue',
            label=f'last liquid (batch) at {T_sol:.0f} K')
tax.scatter([X_sol_frac], marker='*', color='red',
            label=f'last liquid (frac.) at {T_sol_frac:.0f} K')

leg = tax.legend(bbox_to_anchor=(1., 1.), bbox_transform=ax[0].transAxes, prop={'size': 8})
figure.set_tight_layout(True)
tax.get_axes().axis('off')
tax.clear_matplotlib_ticks()
tax.get_axes().set_aspect(1)
tax._redraw_labels()
figure.savefig('figures/di_ab_an_melting.pdf')
tax.show()
