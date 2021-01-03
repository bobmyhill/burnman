from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from input_dataset import create_minerals
from fitting_functions import equilibrium_order


mineral_dataset = create_minerals()
endmembers = mineral_dataset['endmembers']
solutions = mineral_dataset['solutions']


opx = solutions['opx']
opx.set_composition([0.1, 0.1, 0., 0., 0.8])

temperatures = np.linspace(850., 1350., 101)
Q = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    opx.set_state(1.e5, T)
    equilibrium_order(opx)
    Q[i] = opx.molar_fractions[4]


plt.plot(temperatures-273.15, Q)
plt.xlabel('Temperatures (C)')
plt.ylabel('Q')
plt.show()
