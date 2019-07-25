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


opx = child_solutions['mg_fe_opx']
opx.set_composition([0.1, 0.1, 0.8])

temperatures = np.linspace(850., 1350., 101)
Q = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    opx.set_state(1.e5, T)
    equilibrium_order(opx)
    Q[i] = opx.molar_fractions[2]


plt.plot(temperatures-273.15, Q)
plt.xlabel('Temperatures (C)')
plt.ylabel('Q')
plt.show()
