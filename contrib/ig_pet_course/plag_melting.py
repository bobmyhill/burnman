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


abL = HGP_2018_ds633.abL()
anL = HGP_2018_ds633.anL()
ab = HGP_2018_ds633.ab()
an = HGP_2018_ds633.an()

# A super simple an-ab melt model (following White et al., 2007)
liq = burnman.SolidSolution(name='ideal anL-abL',
                            solution_type='ideal',
                            endmembers=[[anL, '[An]1'],
                                        [abL, '[Ab]1']])
plag = JH_2015.plagioclase()

# anorthite is the first endmember
assemblage = burnman.Composite([plag, liq])
assemblage.set_state(1.e5, 1473.15)

xans = np.linspace(0.999, 0.001, 101)
xanls = np.empty_like(xans)
Ts = np.empty_like(xans)

for i, x in enumerate(xans):
    plag.set_composition([x, 1.-x])
    if i < 10:
        liq.set_composition([x, 1.-x])
    composition = plag.formula
    equality_constraints = [('phase_fraction', (liq, np.array([0.0]))),
                            ('P', 1.e5)]

    sol, prm = equilibrate(composition, assemblage, equality_constraints,
                           verbose=False)

    xanls[i] = sol.assemblage.phases[1].molar_fractions[0]
    Ts[i] = sol.assemblage.temperature

fig = plt.figure(figsize=(8, 3))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
ax[0].plot(xans, Ts, label='plag composition')
ax[0].plot(xanls, Ts, label='melt composition')


xans = np.linspace(1., 0., 101)
Gs = np.empty((2, len(xans)))

P = 1.e5
T = 1600.
for i, ph in enumerate([plag, liq]):
    for j, x in enumerate(xans):
        ph.set_composition([x, 1.-x])
        ph.set_state(P, T)
        Gs[i, j] = ph.gibbs/1.e3

ax[1].plot(xans, Gs[0] - Gs[0, 0]*xans - Gs[0, -1]*(1. - xans), label='plag')
ax[1].plot(xans, Gs[1] - Gs[0, 0]*xans - Gs[0, -1]*(1. - xans), label='melt')

plag.set_composition([0.3, 0.7])
composition = plag.formula
equality_constraints = [('T', T),
                        ('P', P)]

sol, prm = equilibrate(composition, assemblage, equality_constraints,
                       verbose=False)

mu_ab, mu_an = sol.assemblage.chemical_potential([ab.formula, an.formula])

x_an, x_anl = [sol.assemblage.phases[i].molar_fractions[0]
               for i in range(2)]


ax[0].plot([x_an, x_an], [1300., 1900.], linestyle='--', color='grey')
ax[0].plot([x_anl, x_anl], [1300., 1900.], linestyle='--', color='grey')
ax[0].set_ylim(1300., 1900.)

ax[1].plot([x_an, x_an], [-15., 15.], linestyle='--', color='grey')
ax[1].plot([x_anl, x_anl], [-15., 15.], linestyle='--', color='grey')
ax[1].set_ylim(-15, 15)

ax[1].plot([0., 1.], [mu_ab/1.e3 - Gs[0, -1], mu_an/1.e3 - Gs[0, 0]],
           linestyle=':', color='k', label='equilibrium')

for i in range(2):
    ax[i].legend()
    ax[i].set_xlabel('$x_{{an}}$ (mol%)')
    ax[i].set_xlim(0., 1.)

ax[0].plot([0., 1.], [1600., 1600.], linestyle=':', color='k',
           label='T section')
ax[0].text(0.1, 1700., 'melt')
ax[0].text(0.32, 1620., 'melt+plag')
ax[0].text(0.8, 1500.,  'plag')
ax[0].set_ylabel('Temperature (K)')
ax[1].set_ylabel('$\\mathcal{{G}}$ (soln - mech.mix.plag.mbrs; kJ/mol)')

fig.set_tight_layout(True)
fig.savefig('figures/plagioclase_melting.pdf')
plt.show()
