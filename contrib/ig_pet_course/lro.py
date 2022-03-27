# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
ideal_gt
--------
"""

from __future__ import absolute_import
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import burnman
from burnman import equilibrate
from burnman.minerals import HGP_2018_ds633


di = HGP_2018_ds633.di()
jd = HGP_2018_ds633.jd()

dj = burnman.CombinedMineral([di, jd], [0.5, 0.5], [-6.e3/2., 0., 0.])

# A super simple an-ab melt model (following White et al., 2007)
omph = burnman.SolidSolution(name='di-jd',
                             solution_type='symmetric',
                             endmembers=[[di, '[Ca]0.5[Ca]0.5[Mg]0.5[Mg]0.5Si2O6'],
                                         [jd, '[Na]0.5[Na]0.5[Al]0.5[Al]0.5Si2O6'],
                                         [dj, '[Ca]0.5[Na]0.5[Al]0.5[Mg]0.5Si3O12']],
                             energy_interaction=[[26.e3, 16.e3],
                                                 [16.e3]])

assemblage = burnman.Composite([omph])
temperatures = np.linspace(473.15, 1273.15, 5)

xs = np.linspace(0.01, 0.99, 91)
Ss = np.empty((len(xs), len(temperatures))) * np.nan
for i, x in enumerate(xs):
    disord = np.array([1. - x, x, 0.])
    if x < 0.5:
        ord = np.array([1. - 2.*x, 0., 2.*x])
    else:
        ord = np.array([0., 2.*x - 1., 2.*(1. - x)])

    print(ord*0.99 + disord*0.01)
    omph.set_composition(ord*0.99 + disord*0.01)
    try:
        sols = equilibrate(assemblage.formula, assemblage,
                           equality_constraints=[['T', temperatures],
                                                 ['P', 1.e5]])

        Ss[i] = [sol.assemblage.phases[0].excess_entropy for sol in sols[0]]
    except:
        pass

for i, T in enumerate(temperatures):
    plt.plot(xs, Ss[:, i], label=f'{T-273.15:.0f}$^{{\circ}}$C')


omph2 = deepcopy(omph)

c = 0.376
assemblage = burnman.Composite([omph, omph2], [1. - 2.*c, c*2.])

temperatures = np.linspace(1073.15, 273.15, 1001)

omph.set_composition([0.98, 0.01, 0.01])
omph2.set_composition([0.01, 0.01, 0.98])

sols = equilibrate(assemblage.formula, assemblage,
                   equality_constraints=[['T', temperatures],
                                         ['P', 1.e5]])

Ts = [sol.assemblage.temperature for sol in sols[0] if sol.success]
Ss = [sol.assemblage.phases[0].excess_entropy for sol in sols[0] if sol.success]
Ss2 = [sol.assemblage.phases[1].excess_entropy for sol in sols[0] if sol.success]


xs = np.array([sol.assemblage.phases[0].formula['Na'] for sol in sols[0] if sol.success])
xs2 = np.array([sol.assemblage.phases[1].formula['Na'] for sol in sols[0] if sol.success])

first = True
for i, T in enumerate(Ts):
    if (T - 273.15) % 200. < 0.1:
        if first:
            plt.plot([xs[i], xs2[i]], [Ss[i], Ss2[i]], c='gray', linestyle=':', label='tielines')
            first=False
        else:
            plt.plot([xs[i], xs2[i]], [Ss[i], Ss2[i]], c='gray', linestyle=':')
        plt.plot([1. - xs[i], 1. - xs2[i]], [Ss[i], Ss2[i]], c='gray', linestyle=':')


xs_comb = np.concatenate((xs[::-1], xs2, 1. - xs2[::-1], 1. - xs))
Ss_comb = np.concatenate((Ss[::-1], Ss2, Ss2[::-1], Ss))
plt.plot(xs_comb, Ss_comb, label='solvus', c='black')
plt.fill_between(xs_comb, Ss_comb*0., Ss_comb, color='purple', alpha=0.1)

plt.xlabel('x (di)')
plt.ylabel('Entropy of mixing (J/K/mol)')

plt.legend()

plt.savefig('jd_di_mixing.pdf')
plt.show()
