from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tools import print_table_for_mineral_constants

import burnman

from burnman import AnisotropicMineral
from burnman.tools.unitcell import molar_volume_from_unit_cell_volume
from burnman.tools.eos import check_eos_consistency

stv = burnman.minerals.SLB_2011.stishovite()
print(stv.property_modifiers)
stv.property_modifiers = [['landau', {'Tc_0': -800., 'S_D': 0.12, 'V_D': 1.e-8}]]
check_eos_consistency(stv, 56.e9, 300., verbose=True)


stv2 = burnman.minerals.SLB_2011.stishovite()
stv2.property_modifiers = []

for T in [50., 100., 150.]:
    pressures = np.linspace(10.e9, 30.e9, 101)
    temperatures = T + 0.*pressures
    plt.plot(pressures/1.e9, stv.evaluate(['V'], pressures, temperatures)[0]) # - stv2.evaluate(['V'], pressures, temperatures)[0])

plt.show()
exit()


stv = burnman.minerals.SLB_2011.stishovite()
print(stv.property_modifiers)
stv.property_modifiers = []

pstv = burnman.minerals.SLB_2011.stishovite()
pstv.property_modifiers = [['linear', {'delta_E': 1.e-8*55.e9 - 0.12*300.,
                                       'delta_S': -0.12,
                                       'delta_V': -1.e-8}]] # S, V made 10* bigger

stv.set_state(1.e5, 300.)
print(stv.V)

pressures = np.linspace(30.e9, 130.e9, 101)
temperatures = 300. + 0.*pressures
#plt.plot(pressures/1.e9, stv.evaluate(['gibbs'], pressures, temperatures)[0])
#plt.plot(pressures/1.e9, pstv.evaluate(['gibbs'], pressures, temperatures)[0])

plt.plot(pressures/1.e9, stv.evaluate(['gibbs'], pressures, temperatures)[0] - pstv.evaluate(['gibbs'], pressures, temperatures)[0])
plt.show()

exit()

a = np.cbrt(per.params['V_0'])
cell_parameters = np.array([a, a, a, 90, 90, 90])

beta_RT = per.beta_T

per_data = np.loadtxt('data/isentropic_stiffness_tensor_periclase.dat')


AnisotropicMineral(per, cell_parameters, constants)
