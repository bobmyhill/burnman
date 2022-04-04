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

fig = plt.figure(figsize=(4, 3))
ax = [fig.add_subplot(1, 1, 1)]

n_x = 101
xs = np.linspace(0.0001, 0.99999, n_x)

G_mech = np.empty(n_x)
G_ideal = np.empty(n_x)
G_tot = np.empty(n_x)

for T in [0.1, 500., 1000., 1500.]:

    gt.set_state(1.e5, T)
    Gpy = gt.endmembers[0][0].gibbs
    Ggr = gt.endmembers[1][0].gibbs
    G_mech = Gpy + xs*(Ggr - Gpy)

    for j, x in enumerate(xs):
        gt_ideal.set_state(1.e5, T)
        gt_ideal.set_composition([1.-x, x])
        G_ideal[j] = gt_ideal.excess_gibbs

    ax[0].plot(xs, (G_ideal)/1.e3, label=f'{T:.0f} K')

ax[0].set_xlim(0., 1.)
ax[0].set_xlabel('Composition')
ax[0].set_ylabel('-$TS_{{conf}}$ (kJ/mol)')

ax[0].legend()

fig.set_tight_layout(True)

fig.savefig('figures/ideal_mixing_energies.pdf')

plt.show()
