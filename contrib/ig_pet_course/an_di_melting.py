# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
an_di_melting
-------------
"""

from __future__ import absolute_import
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

import burnman
from burnman import equilibrate
from burnman.minerals import HGP_2018_ds633
from collections import Counter


diL = HGP_2018_ds633.diL()
anL = HGP_2018_ds633.anL()
di = HGP_2018_ds633.di()
an = HGP_2018_ds633.an()

# A super simple ideal model
liq = burnman.SolidSolution(name='ideal anL-diL',
                            solution_type='symmetric',
                            endmembers=[[diL, '[Mg]1'],
                                        [anL, '[Ca]1']],
                            energy_interaction=[[-15.e3]])

total = sum(di.formula.values())
fdi = Counter({k: v / total for k, v in di.formula.items()})
total = sum(an.formula.values())
fan = Counter({k: -v / total for k, v in an.formula.items()})

d = fan.copy()
d.update(fdi)
free_compositional_vectors = [d]

assemblage = burnman.Composite([di, an])
assemblage.set_state(1.e5, 1473.15)
di.H_ref = di.H
an.H_ref = an.H

liq.set_composition([0.4, 0.6])
composition = liq.formula
assemblage = burnman.Composite([liq, di, an])
equality_constraints = [('phase_fraction', (di, np.array([0.0]))),
                        ('phase_fraction', (an, np.array([0.0]))),
                        ('P', 1.e5)]

sol, prm = equilibrate(composition, assemblage, equality_constraints,
                       free_compositional_vectors,
                       verbose=False)

x_eutectic = sol.assemblage.phases[0].molar_fractions[1]
T_C_eutectic = sol.assemblage.temperature - 273.15


fig = plt.figure(figsize=(8, 6))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

figpd = plt.figure(figsize=(6, 4))
axpd = [figpd.add_subplot(1, 1, 1)]

figsp = plt.figure(figsize=(8, 3))
axsp = [figsp.add_subplot(1, 2, i) for i in range(1, 3)]

ax[0].plot([0., 1.], [T_C_eutectic, T_C_eutectic], color='k')
axpd[0].text(0.88, T_C_eutectic + 5, f'{T_C_eutectic:.0f}$^{{\\circ}}$C')
axpd[0].plot([0., 1.], [T_C_eutectic, T_C_eutectic], color='k')
axsp[0].plot([0., 1.], [T_C_eutectic, T_C_eutectic], color='k')

ax[2].plot([0., x_eutectic, 1., 0.],
           np.array([di.H - di.H_ref,
                     liq.H - (1. - x_eutectic)*di.H_ref - x_eutectic*an.H_ref,
                     an.H - an.H_ref, di.H - di.H_ref])/1.e3, color='k')
ax[3].plot([0., x_eutectic, 1., 0.], [di.S, liq.S, an.S, di.S], color='k')

n_x = 101
xs = np.linspace(0.0001, x_eutectic, n_x)
Ts = np.empty(n_x)
Ss = np.empty(n_x)
Hs = np.empty(n_x)

for i, x in enumerate(xs):
    liq.set_composition([1.-x, x])
    composition = liq.formula
    assemblage = burnman.Composite([liq, di])
    equality_constraints = [('phase_fraction', (di, np.array([0.0]))),
                            ('P', 1.e5)]

    sol, prm = equilibrate(composition, assemblage, equality_constraints)
    Ts[i] = sol.assemblage.temperature
    an.set_state(1.e5, sol.assemblage.temperature)
    Hs[i] = liq.H - (1. - x)*di.H_ref - x*an.H_ref
    Ss[i] = liq.S

ax[0].plot(xs, Ts-273.15, color='k')
axpd[0].plot(xs, Ts-273.15, color='k')
axsp[0].plot(xs, Ts-273.15, color='k')

ax[2].plot(xs, Hs/1.e3, color='k')
ax[3].plot(xs, Ss, color='k')

xs = np.linspace(x_eutectic, 0.9999, n_x)
Ts = np.empty(n_x)
Ss = np.empty(n_x)
Hs = np.empty(n_x)
for i, x in enumerate(xs):
    liq.set_composition([1.-x, x])
    composition = liq.formula
    assemblage = burnman.Composite([liq, an])
    equality_constraints = [('phase_fraction', (an, np.array([0.0]))),
                            ('P', 1.e5)]

    sol, prm = equilibrate(composition, assemblage, equality_constraints)
    Ts[i] = sol.assemblage.temperature
    di.set_state(1.e5, sol.assemblage.temperature)
    Hs[i] = liq.H - (1. - x)*di.H_ref - x*an.H_ref
    Ss[i] = liq.S


ax[0].plot(xs, Ts-273.15, color='k')
axpd[0].plot(xs, Ts-273.15, color='k')
axsp[0].plot(xs, Ts-273.15, color='k')
ax[2].plot(xs, Hs/1.e3, color='k')
ax[3].plot(xs, Ss, color='k')

for i in range(4):
    ax[i].set_xlabel('p(an)')
    ax[i].set_xlim(0, 1)

axpd[0].set_xlabel('p(an)')
axpd[0].set_xlim(0, 1)
axsp[0].set_xlabel('p(an)')
axsp[0].set_xlim(0, 1)
axsp[1].set_xlabel('p(an)')
axsp[1].set_xlim(0, 1)


ax[0].set_ylabel('Temperature ($^{\\circ}$C)')
axpd[0].set_ylabel('Temperature ($^{\\circ}$C)')
axsp[0].set_ylabel('Temperature ($^{\\circ}$C)')
ax[1].set_ylabel('$\Delta \mathcal{G}_{asmblg-sol}$ (kJ/mol)')
axsp[1].set_ylabel('$\Delta \mathcal{G}_{asmblg-sol}$ (kJ/mol)')
ax[2].set_ylabel('$\mathcal{H} - \mathcal{H}_{1200C}$ (kJ/mol)')
ax[3].set_ylabel('Entropy (J/K/mol)')
ax[0].set_ylim(1150, 1650)
axpd[0].set_ylim(1150, 1650)
axsp[0].set_ylim(1150, 1650)

ax[1].set_ylim(-30, 30)
axsp[1].set_ylim(-30, 30)

for a in [ax, axpd, axsp]:
    a[0].text(0.5, 1200., 'di + an', horizontalalignment='center',
              verticalalignment='center')
    a[0].text(0.1, 1325., 'di + L', horizontalalignment='center',
              verticalalignment='center')
    a[0].text(0.4, 1475., 'L', horizontalalignment='center',
              verticalalignment='center')
    a[0].text(0.8, 1375., 'L + an', horizontalalignment='center',
              verticalalignment='center')

ax[2].text(0.5, 0., 'di + an', horizontalalignment='center',
           verticalalignment='center')
ax[2].text(0.5, 70., 'di + an + L', horizontalalignment='center',
           verticalalignment='center')
ax[2].text(0.1, 120., 'di + L', horizontalalignment='center',
           verticalalignment='center')
ax[2].text(0.3, 220., 'L', horizontalalignment='center',
           verticalalignment='center')
ax[2].text(0.8, 150., 'L + an', horizontalalignment='center',
           verticalalignment='center')

ax[3].text(0.7, 550., 'di + an', horizontalalignment='center',
           verticalalignment='center')
ax[3].text(0.5, 630., 'di + an + L', horizontalalignment='center',
           verticalalignment='center')
ax[3].text(0.1, 600., 'di + L', horizontalalignment='center',
           verticalalignment='center')
ax[3].text(0.3, 720., 'L', horizontalalignment='center',
           verticalalignment='center')
ax[3].text(0.8, 710., 'L + an', horizontalalignment='center',
           verticalalignment='center')

fig.set_tight_layout(True)
figsp.set_tight_layout(True)


xs = np.linspace(0., 1., n_x)
n_TCs = 5
gibbs = np.empty((n_TCs, n_x))
gibbs_eqm = np.empty((n_TCs, 2))
entropy = np.empty((n_TCs, n_x))
entropy_liq_eqm = np.empty((n_TCs, 2))
entropy_sol_eqm = np.empty((n_TCs, 2))

enthalpy = np.empty((n_TCs, n_x))
enthalpy_liq_eqm = np.empty((n_TCs, 2))
enthalpy_sol_eqm = np.empty((n_TCs, 2))
xs_eqm = np.empty((n_TCs, 2))
T_Cs = np.linspace(1200., 1600., n_TCs)
for j, T_C in enumerate(T_Cs):
    for i, x in enumerate(xs):
        liq.set_composition([1.-x, x])
        assemblage = burnman.Composite([liq, di, an])
        assemblage.set_state(1.e5, T_C + 273.15)

        gibbs[j, i] = liq.gibbs - (1. - x)*di.gibbs - x*an.gibbs
        enthalpy[j, i] = liq.H - (1. - x)*di.H_ref - x*an.H_ref
        entropy[j, i] = liq.S

    ax[0].plot([0.,  1.], [T_C, T_C])
    axsp[0].plot([0.,  1.], [T_C, T_C])

    liq.set_composition([0.4, 0.6])
    composition = liq.formula

    for i, ph in enumerate([di, an]):
        assemblage = burnman.Composite([liq, ph])
        equality_constraints = [('phase_fraction', (ph, np.array([0.0]))),
                                ('T', T_C+273.15),
                                ('P', 1.e5)]

        sol, prm = equilibrate(composition, assemblage, equality_constraints,
                               free_compositional_vectors,
                               verbose=False)

        assemblage = burnman.Composite([liq, di, an])
        assemblage.set_state(1.e5, T_C + 273.15)
        xs_eqm[j, i] = liq.molar_fractions[1]
        gibbs_eqm[j, i] = (liq.gibbs - (1. - xs_eqm[j, i])*di.gibbs
                           - xs_eqm[j, i]*an.gibbs)
        enthalpy_liq_eqm[j, i] = (liq.H - (1. - xs_eqm[j, i])*di.H_ref
                                  - xs_eqm[j, i]*an.H_ref)
        enthalpy_sol_eqm[j, i] = ph.H - ph.H_ref
        entropy_liq_eqm[j, i] = liq.S
        entropy_sol_eqm[j, i] = ph.S

    ln = ax[1].plot(xs, gibbs[j]/1.e3, label=f'T: {T_Cs[j]} C',
                    linestyle=':')
    ln = axsp[1].plot(xs, gibbs[j]/1.e3, label=f'T: {T_Cs[j]} C',
                      linestyle=':')
    xs_min = []
    gibbs_min = []
    entropy_min = []
    enthalpy_min = []
    if gibbs[j, 0] > 0. and gibbs_eqm[j, 0] < 0.:
        xs_min = [0., xs_eqm[j, 0]]
        gibbs_min = [0., gibbs_eqm[j, 0]]
        entropy_min = [entropy_sol_eqm[j, 0], entropy_liq_eqm[j, 0]]
        enthalpy_min = [enthalpy_sol_eqm[j, 0], enthalpy_liq_eqm[j, 0]]

        mask = np.all(((xs < xs_eqm[j, 1]), (xs > xs_eqm[j, 0])), axis=0)

        xs_min.extend(list(xs[mask]))
        gibbs_min.extend(list(gibbs[j][mask]))
        entropy_min.extend(list(entropy[j][mask]))
        enthalpy_min.extend(list(enthalpy[j][mask]))

        xs_min.extend([xs_eqm[j, 1], 1.])
        gibbs_min.extend([gibbs_eqm[j, 1], 0.])
        entropy_min.extend([entropy_liq_eqm[j, 1], entropy_sol_eqm[j, 1]])
        enthalpy_min.extend([enthalpy_liq_eqm[j, 1], enthalpy_sol_eqm[j, 1]])

    elif xs_eqm[j, 0] < xs_eqm[j, 1]:
        if gibbs[j, -1] > 0. and gibbs_eqm[j, -1] < 0.:
            mask = xs < xs_eqm[j, 1]
            xs_min = list(xs[mask])
            gibbs_min = list(gibbs[j][mask])
            entropy_min = list(entropy[j][mask])
            enthalpy_min = list(enthalpy[j][mask])

            xs_min.extend([xs_eqm[j, 1], 1.])
            gibbs_min.extend([gibbs_eqm[j, 1], 0.])
            entropy_min.extend([entropy_liq_eqm[j, 1], entropy_sol_eqm[j, 1]])
            enthalpy_min.extend([enthalpy_liq_eqm[j, 1],
                                 enthalpy_sol_eqm[j, 1]])
        else:
            xs_min = xs
            gibbs_min = gibbs[j]
            entropy_min = entropy[j]
            enthalpy_min = enthalpy[j]

    else:
        xs_min = [0., 1.]
        gibbs_min = [0., 0.]
        entropy_min = [entropy_sol_eqm[j, 0], entropy_sol_eqm[j, 1]]
        enthalpy_min = [enthalpy_sol_eqm[j, 0], enthalpy_sol_eqm[j, 1]]

    ax[1].plot(xs_min, np.array(gibbs_min)/1.e3, color=ln[0].get_color())
    axsp[1].plot(xs_min, np.array(gibbs_min)/1.e3, color=ln[0].get_color())
    try:
        ax[2].plot(xs_min, np.array(enthalpy_min)/1.e3,
                   color=ln[0].get_color())
        ax[3].plot(xs_min, entropy_min, color=ln[0].get_color())
    except ValueError:
        pass
    if xs_eqm[j, 0] > 0.0001 and gibbs_eqm[j, 0] < 0.:
        ax[0].scatter([xs_eqm[j, 0]], [T_Cs[j]], color=ln[0].get_color())
        axsp[0].scatter([xs_eqm[j, 0]], [T_Cs[j]], color=ln[0].get_color())
        ax[1].scatter([xs_eqm[j, 0]], [gibbs_eqm[j, 0]/1.e3],
                      color=ln[0].get_color())
        axsp[1].scatter([xs_eqm[j, 0]], [gibbs_eqm[j, 0]/1.e3],
                        color=ln[0].get_color())

    if xs_eqm[j, 1] < 0.9999 and gibbs_eqm[j, 0] < 0.:
        ax[0].scatter([xs_eqm[j, 1]], [T_Cs[j]], color=ln[0].get_color())
        axsp[0].scatter([xs_eqm[j, 1]], [T_Cs[j]], color=ln[0].get_color())
        ax[1].scatter([xs_eqm[j, 1]], [gibbs_eqm[j, 1]/1.e3],
                      color=ln[0].get_color())
        axsp[1].scatter([xs_eqm[j, 1]], [gibbs_eqm[j, 1]/1.e3],
                        color=ln[0].get_color())

    figsp.savefig(f'figures/an_di_melting_step_{j}.pdf')


fig.savefig('figures/an_di_melting.pdf')


figpd.savefig('figures/an_di_melting_phase_diagram.pdf')


axpd[0].set_xlim(0., 0.3)
axpd[0].set_ylim(1300., 1400.)
axpd[0].text(0.2, 1380., 'L', horizontalalignment='center', verticalalignment='center')
figpd.savefig('figures/di_rich_an_di_melting.pdf')


plt.show()
