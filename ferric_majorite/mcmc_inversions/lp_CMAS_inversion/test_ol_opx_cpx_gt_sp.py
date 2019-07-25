from __future__ import absolute_import
from __future__ import print_function

import platform
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from scipy.optimize import fsolve
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))

import burnman
from input_dataset import create_minerals
from fitting_functions import equilibrium_order


mineral_dataset = create_minerals()
endmembers = mineral_dataset['endmembers']
solutions = mineral_dataset['solutions']
child_solutions = mineral_dataset['child_solutions']

phases = [endmembers['fo'], endmembers['sp']]
phases.append(child_solutions['oen_mgts_odi'])
phases.append(child_solutions['di_cen_cats'])
phases.append(child_solutions['py_gr_gt'])


temperatures = np.linspace(1350., 1650., 21)
assemblage = burnman.Composite([phases])


child_solutions['oen_mgts_odi'].set_composition([0.74, 0.20, 0.06])
child_solutions['di_cen_cats'].set_composition([0.33, 0.46, 0.21])
child_solutions['py_gr_gt'].set_composition([0.87, 0.13])

child_solutions['oen_mgts_odi'].guess = np.array([0.74, 0.20, 0.06])
child_solutions['di_cen_cats'].guess = np.array([0.33, 0.46, 0.21])
child_solutions['py_gr_gt'].guess = np.array([0.87, 0.13])

assemblage = burnman.Composite(phases, [0.80, 0.01, 0.09, 0.09, 0.01])
assemblage.set_state(2.4e9, 1200)
equality_constraints = [('T', temperatures),
                        ('phase_proportion',
                         (endmembers['sp'], np.array([0.0])))]

print(assemblage)
sols, prm = burnman.equilibrate(assemblage.formula, assemblage,
                                equality_constraints,
                                initial_state_from_assemblage=True,
                                initial_composition_from_assemblage=True,
                                store_iterates=False,
                                store_assemblage=True)

pressures = [sol.pressure for sol in sols]

plt.plot(temperatures-273.15, pressures)
plt.xlabel('Temperatures (C)')
plt.ylabel('Q')
plt.show()
