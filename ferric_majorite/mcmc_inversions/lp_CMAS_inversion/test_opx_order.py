from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
# from scipy.optimize import minimize, fsolve

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))
import burnman
from burnman.solidsolution import SolidSolution as Solution
from burnman.solutionbases import transform_solution_to_new_basis


# Orthopyroxene endmembers
oen = burnman.minerals.HHPH_2013.en()
ofs = burnman.minerals.HHPH_2013.fs()

for f in [7.e3, 15.e3]:
    x = f / 4. + 2.25e3
    Etweak = f / 4. - 8.35e3
    ofm = burnman.CombinedMineral([oen, ofs], [0.5, 0.5], [Etweak, 0., 0.],
                                  name='ofm')

    opx = Solution(name='orthopyroxene with order-disorder',
                   solution_type='asymmetric',
                   alphas=[1., 1., 1.],
                   endmembers=[[oen, '[Mg][Mg][Si]1/2Si3/2O6'],
                               [ofs, '[Fe][Fe][Si]1/2Si3/2O6'],
                               [ofm, '[Fe][Mg][Si]1/2Si3/2O6']],
                   energy_interaction=[[f, x],
                                       [x]])


    assemblage = burnman.Composite([opx], [1.])
    opx.set_composition([0.1, 0.1, 0.8])

    temperatures = np.linspace(100., 1500., 101)
    equality_constraints = equality_constraints = [('T', temperatures),
                                                   ('P', 1.e5)]
    assemblage.set_state(1.e5, 300)
    sols, prm = burnman.equilibrate(assemblage.formula, assemblage,
                                    equality_constraints,
                                    initial_state_from_assemblage=True,
                                    initial_composition_from_assemblage=True,
                                    store_iterates=False,
                                    store_assemblage=True)

    x_ofm = [sol.assemblage.phases[0].molar_fractions[2] for sol in sols]
    plt.plot(temperatures, x_ofm)
plt.show()
