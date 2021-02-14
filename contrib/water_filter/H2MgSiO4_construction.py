from __future__ import absolute_import
from __future__ import print_function
import os.path
import sys
sys.path.insert(1, os.path.abspath('../..'))

import burnman
from burnman.minerals import SLB_2011, HP_2011_ds62
from burnman import constants
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import matplotlib.pyplot as plt

print('If we use any of these, we must change HSC->SUP convention for Holland and Powell '
      'endmembers to make consistent with SLB database')

Mg2SiO4 = SLB_2011.forsterite()
H2MgSiO4 = burnman.CombinedMineral([HP_2011_ds62.phA(), SLB_2011.mg_wadsleyite(), SLB_2011.hp_clinoenstatite()],
                                   [1./3., -5./3., 1.])

#H2MgSiO4 = burnman.CombinedMineral([HP_2011_ds62.br(), SLB_2011.periclase(), SLB_2011.hp_clinoenstatite()],
#                                   [1., -1., 0.5])

#H2MgSiO4 = burnman.CombinedMineral([HP_2011_ds62.br(), SLB_2011.forsterite(), SLB_2011.hp_clinoenstatite()],
#                                   [1., -1., 1.])

#H2MgSiO4 = burnman.CombinedMineral([HP_2011_ds62.phA(), SLB_2011.forsterite(), HP_2011_ds62.br(), SLB_2011.periclase(), SLB_2011.hp_clinoenstatite()],
#                                   [1./6., -5./6., 1./2., -1./2., 0.75])

H2MgSiO4.set_state(1.e5, 298.15)
print(H2MgSiO4.V)
print(H2MgSiO4.S)
print(H2MgSiO4.formula)
print(H2MgSiO4.K_T/1.e9)
K1 = H2MgSiO4.K_T

dP = 1000.
H2MgSiO4.set_state(1.e5+dP, 298.15)
K2 = H2MgSiO4.K_T

print((K2 - K1)/dP)


fig = plt.figure()
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
pressures = np.linspace(1.e9, 60.e9, 101)
for T in [300., 2000.]:
    temperatures = pressures*0. + T
    Vs1, Ss1 = Mg2SiO4.evaluate(['V', 'S'], pressures, temperatures)
    Vs2, Ss2 = H2MgSiO4.evaluate(['V', 'S'], pressures, temperatures)

    ax[0].plot(pressures/1.e9, Vs1, label='Mg2SiO4')
    ax[0].plot(pressures/1.e9, Vs2, label='H2MgSiO4')
    ax[1].plot(pressures/1.e9, Ss1, label='Mg2SiO4')
    ax[1].plot(pressures/1.e9, Ss2, label='H2MgSiO4')
plt.legend()
plt.show()
