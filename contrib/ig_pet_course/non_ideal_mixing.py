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
from copy import copy

py = HGP_2018_ds633.py()
gr1 = HGP_2018_ds633.gr()

gr_params = copy(gr1.params)


# Fudge Ggr to make plot look more readable:
gr_params['S_0'] += py.params['S_0'] - gr1.params['S_0']
gr1.set_state(1.e5, 1000.)
py.set_state(1.e5, 1000.)
gr_params['H_0'] += py.gibbs - gr1.gibbs + 15.e3
gr = burnman.Mineral(gr_params)


# A one-site model
gt = burnman.SolidSolution(name='py-gr',
                           solution_type='symmetric',
                           endmembers=[[py, '[Mg]1'],
                                       [gr, '[Ca]1']],
                           energy_interaction=[[18.e3]])
gt_ideal = burnman.SolidSolution(name='py-gr',
                                 solution_type='ideal',
                                 endmembers=[[py, '[Mg]1'],
                                             [gr, '[Ca]1']])

fig = plt.figure(figsize=(11, 3))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

n_x = 101
xs = np.linspace(0.0001, 0.99999, n_x)

G_mech = np.empty(n_x)
G_ideal = np.empty(n_x)
G_tot = np.empty(n_x)

for i, T in enumerate([800., 1000., 1200.]):

    gt.set_state(1.e5, T)
    Gpy = gt.endmembers[0][0].gibbs
    Ggr = gt.endmembers[1][0].gibbs
    G_mech = Gpy + xs*(Ggr - Gpy)

    for j, x in enumerate(xs):
        gt.set_state(1.e5, T)
        gt_ideal.set_state(1.e5, T)
        gt.set_composition([1.-x, x])
        gt_ideal.set_composition([1.-x, x])
        G_tot[j] = gt.gibbs
        G_ideal[j] = gt_ideal.gibbs

    ax[i].plot(xs, G_mech/1.e3, label='mech')
    ax[i].plot(xs, G_ideal/1.e3, label='ideal')
    ax[i].plot(xs, (G_mech + G_tot - G_ideal)/1.e3,
               label='non-ideal')
    ax[i].plot(xs, G_tot/1.e3, label='total')

    gt.set_composition([0.5, 0.5])
    composition = gt.formula
    gt1 = copy(gt)
    gt2 = copy(gt)
    gt1.set_composition([0.01, 0.99])
    gt2.set_composition([0.99, 0.01])
    assemblage = burnman.Composite([gt1, gt2])
    equality_constraints = [['P', 1.e5],
                            ['T', T]]
    sol, prm = equilibrate(composition,
                           assemblage,
                           equality_constraints)
    mus = sol.assemblage.phases[0].partial_gibbs
    xxs = [sol.assemblage.phases[0].molar_fractions[1],
           sol.assemblage.phases[1].molar_fractions[1]]
    Gxs = np.array([sol.assemblage.phases[0].gibbs,
                    sol.assemblage.phases[1].gibbs])
    ax[i].plot([0., 1.], mus/1.e3, label='$\\mu$-tieline')
    ax[i].scatter(xxs, Gxs/1.e3, color='red')

    Gav = (Gpy + Ggr)/2./1000.
    ax[i].set_xlim(0., 1.)
    ax[i].set_ylim(Gav - 8., Gav + 8.)
    ax[i].text(0.05, Gav - 7, f'{T:.0f} K')
    ax[i].set_xlabel('Composition')
    ax[i].set_ylabel('$\\mathcal{{G}}$ (kJ/mol)')

ax[2].legend(ncol=2)
fig.set_tight_layout(True)

fig.savefig('figures/non_ideal_mixing.pdf')

plt.show()
