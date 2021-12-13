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
from burnman.minerals import Pitzer_Sterner_1994
from collections import Counter
from burnman.tools.eos import check_eos_consistency

abL = HGP_2018_ds633.abL()
h2oL = HGP_2018_ds633.h2oL()
ab = HGP_2018_ds633.ab()
h2o = Pitzer_Sterner_1994.H2O_Pitzer_Sterner() # apparently not working... not in main branch either, so that's ok.


h2o.set_state(1.e9+0.5, 1001.)
print(h2o.V)

h2o.set_state(1.e9, 1000.5)
print(h2o.S)

h2o.set_state(1.e9, 1000)
G0 = h2o.gibbs
h2o.set_state(1.e9, 1001)
G1 = h2o.gibbs
h2o.set_state(1.e9+1., 1001)
G2 = h2o.gibbs

print(G0 - G1)
print((G2 - G1)/1.)


# A super simple symmetric model as in Holland and Powell 1998
liq = burnman.SolidSolution(name='ideal abL-H2OL',
                            solution_type='symmetric',
                            endmembers=[[abL, '[Na]1'],
                                        [h2oL, '[H]1']],
                            energy_interaction=[[-9.4e3]])


pressures = np.linspace(1.2e9, 1.2e6, 101)
assemblage = burnman.Composite([ab, liq, h2o])


P = 1.e5
T = 1346.

assemblage.set_state(P, T)

print(h2oL.V)
print(h2o.V)
print(h2oL.S)
print(h2o.S)

xs = np.linspace(0., 1., 101)
Gs = np.empty_like(xs)

for i, x in enumerate(xs):
    liq.set_composition([1.-x, x])
    Gs[i] = liq.gibbs

plt.plot(xs, Gs - xs*Gs[-1] - (1. - xs)*Gs[0])
plt.plot([0., 1.], [ab.gibbs - Gs[0], h2o.gibbs - Gs[-1]])
plt.show()
exit()

for i, P in enumerate(pressures):
    liq.set_composition([0.01, 0.99])
    assemblage.set_state(P, 800.)
    composition = liq.formula

    equality_constraints = [('phase_fraction', (liq, np.array([0.0]))),
                            ('P', P)]

    sols, prm = equilibrate(composition, assemblage, equality_constraints,
                            verbose=True)
    print(sols.assemblage.temperature)
#print([sol.assemblage.temperature for sol in sols])