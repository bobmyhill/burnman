from __future__ import absolute_import
from __future__ import print_function
# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.

import matplotlib.pyplot as plt
import numpy as np

import argparse
import os.path
import sys
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman

dS = 0.
dV = 0.
SiO2_liquid = burnman.CombinedMineral([burnman.SesameMineral('SiO2_sesame_301', 'SiO2')],
                                      [1.0],
                                      [0., dS, dV])

'''
# Check gridding
SiO2_liquid.set_state(44.06e9, 1044.)
print(SiO2_liquid.rho, '4480')
exit()
'''

stv = burnman.minerals.HP_2011_ds62.stv()
coe = burnman.minerals.HP_2011_ds62.coe()
coe = burnman.minerals.SLB_2011.coesite()

'''
P = 13.7e9
T = 3073.
stv.set_state(P, T)
SiO2_liquid.set_state(P, T)
gibbs_diff = stv.gibbs - SiO2_liquid.gibbs

SiO2_liquid = burnman.CombinedMineral([burnman.SesameMineral('SiO2_sesame_301', 'SiO2')],
                                      [1.0],
                                      [gibbs_diff, dS, dV])

stv.set_state(13.7e9, 3073.)
SiO2_liquid.set_state(13.7e9, 3073.)


pressures = np.linspace(3.e9, 15.e9, 101)
temperatures = np.empty_like(pressures)
Tguess = 3073.
for i, pressure in enumerate(pressures):
    temperatures[i] = burnman.tools.equilibrium_temperature([coe, SiO2_liquid],
                                                            [1.0, -1.0], pressure, Tguess)
    Tguess = temperatures[i]
plt.plot(pressures/1.e9, temperatures)


pressures = np.linspace(11.e9, 200.e9, 101)
temperatures = np.empty_like(pressures)
for i, pressure in enumerate(pressures):
    temperatures[i] = burnman.tools.equilibrium_temperature([stv, SiO2_liquid],
                                                            [1.0, -1.0], pressure, 3073.)

plt.plot(pressures/1.e9, temperatures)
plt.show()
exit()
'''


prop = 'rho'

pressures = np.linspace(1.e5, 100.e9, 101)
for temperature in [3073.]:
    temperatures = [temperature] * len(pressures)
    values = SiO2_liquid.evaluate([prop], pressures, temperatures)[0]
    coe_values = coe.evaluate([prop], pressures, temperatures)[0]
    
    plt.plot(pressures/1.e9, values, label='liq @ {0} K'.format(temperature))
    plt.plot(pressures/1.e9, coe_values, linestyle=':', label='coe @ {0} K'.format(temperature))

    
plt.legend(loc='upper left')
plt.xlabel('Pressures (GPa)')
plt.ylabel(prop)
plt.show()

temperatures = np.linspace(2000., 4000., 101)
for pressure in [13.7e9]:
    pressures = [pressure] * len(temperatures)
    values = SiO2_liquid.evaluate([prop], pressures, temperatures)[0]
    coe_values = coe.evaluate([prop], pressures, temperatures)[0]
    
    plt.plot(temperatures, values, label='liq @ {0} GPa'.format(pressure/1.e9))
    plt.plot(temperatures, coe_values, linestyle=':', label='coe @ {0} GPa'.format(pressure/1.e9))
plt.legend(loc='upper left')
plt.xlabel('Temperatures (K)')
plt.ylabel(prop)
plt.show()
