from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import minerals


x_NaCl = 0.2
theta_NaCl = 288.

x_KCl = 0.8
theta_KCl = 235.


theta_NaKCl_ideal = np.power(theta_NaCl, x_NaCl)*np.power(theta_KCl, x_KCl)
theta_NaKCl = theta_NaKCl_ideal - 7.0

print(theta_NaKCl_ideal/theta_NaKCl)


temperatures = np.linspace(1., 300., 300)
Cps = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    Cps[i] = burnman.eos.debye.heat_capacity_v(T, theta_NaKCl, 2.) - (x_NaCl*burnman.eos.debye.heat_capacity_v(T, theta_NaCl, 2.) + x_KCl*burnman.eos.debye.heat_capacity_v(T, theta_KCl, 2.))

plt.plot(temperatures, Cps)
plt.show()



halite = minerals.HP_2011_ds62.hlt()
sylvite = minerals.HP_2011_ds62.syv()

halite.set_state(1.e5, 300.)
sylvite.set_state(1.e5, 300.)


V_excess = 1.5e-7
V_ideal = halite.V*0.5 + sylvite.V*0.5
print('V/V_ideal', (V_ideal + V_excess) / V_ideal)

print('K/K_ideal', np.power(V_ideal/(V_ideal + V_excess) , 7./3.))

print('S_xs', 3.*2.*burnman.constants.gas_constant*np.log((V_ideal + V_excess) / V_ideal))


theta_NaKCl = np.power((theta_NaCl*theta_KCl), 0.5)*V_ideal/(V_ideal + V_excess)

temperatures = np.linspace(1., 300., 300)
Cps = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    Cps[i] = burnman.eos.debye.heat_capacity_v(T, theta_NaKCl, 2.) - (0.5*burnman.eos.debye.heat_capacity_v(T, theta_NaCl, 2.) + 0.5*burnman.eos.debye.heat_capacity_v(T, theta_KCl, 2.))

plt.plot(temperatures, Cps)
plt.show()


# Vex (py-gr)
# V/Videal ~ 1.01
print('Py-Gr')
pyrope = minerals.HP_2011_ds62.py()
grossular = minerals.HP_2011_ds62.gr()

pyrope.set_state(1.e5, 300.)
grossular.set_state(1.e5, 300.)

V_excess = 1.0e-6 # see Du et al., 2015; Ganguly et al., 1993
V_ideal = pyrope.V*0.5 + grossular.V*0.5
print('V/V_ideal', (V_ideal + V_excess) / V_ideal)

print('K/K_ideal', np.power(V_ideal/(V_ideal + V_excess) , 7./3.))

print('S_xs', 3.*pyrope.params['n']*burnman.constants.gas_constant*np.log((V_ideal + V_excess) / V_ideal))
# Dachs and Geiger (2006) show excess entropies of mixing up to 4-5 J/K/mol at 300 K!!
