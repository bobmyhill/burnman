import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from eos.solid_EoS import *
from minerals.deKoker_solids import *
from burnman.minerals import DKS_2013_liquids
from burnman import constants
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
pressure=14.e9 # Pa
actual_temperature=3120. # K
temperature = fsolve(find_temperature, 2000., args=(pressure))[0]
'''

pressure = 14.e9 # Pa
temperature = 3000. # K
stv_SLB=burnman.minerals.SLB_2011.stishovite()
stv=stishovite()

stv_SLB.set_state(pressure, temperature)
stv_delta_G=gibbs_solid(pressure, temperature, stv.params) - stv_SLB.gibbs

print stv_delta_G

print stv_SLB.gibbs, gibbs_solid(pressure, temperature, stv.params)-stv_delta_G, \
    gibbs_solid(pressure, temperature, stv.params)-stv_delta_G-stv_SLB.gibbs

pressure = 14.e9 # Pa
temperature = 4500. # K
stv_SLB.set_state(pressure, temperature)
print stv_SLB.gibbs, gibbs_solid(pressure, temperature, stv.params)-stv_delta_G, \
    (gibbs_solid(pressure, temperature, stv.params)-stv_delta_G-stv_SLB.gibbs)/1500.

pressure = 14.e9 # Pa
temperature = 1500. # K
stv_SLB.set_state(pressure, temperature)
print stv_SLB.gibbs, gibbs_solid(pressure, temperature, stv.params)-stv_delta_G, \
    (gibbs_solid(pressure, temperature, stv.params)-stv_delta_G-stv_SLB.gibbs)/-1500.

# MELTS
def find_temperature(temperature, pressure, solid, liquid):
    liquid.set_state(pressure, temperature[0])
    return gibbs_solid(pressure, temperature[0], solid.params) - liquid.gibbs

print 'STISHOVITE'
stv=stishovite()
SiO2_liq=DKS_2013_liquids.SiO2_liquid()
pressure = 14.e9 # Pa
T_melt = fsolve(find_temperature, 5000., args=(pressure, stv, SiO2_liq))[0]

temperatures_SiO2 = np.linspace(1400., T_melt, 101)
entropy_melting_SiO2 = np.empty_like(temperatures_SiO2)
for i, temperature in enumerate(temperatures_SiO2):
    SiO2_liq.set_state(pressure, temperature)
    V_solid=volume_solid(pressure, temperature, stv.params)
    entropy_melting_SiO2[i] = SiO2_liq.S - entropy_solid(V_solid, temperature, stv.params)

print 'PERICLASE'
per=periclase()
MgO_liq=DKS_2013_liquids.MgO_liquid()
pressure = 14.e9 # Pa
T_melt = fsolve(find_temperature, 5000., args=(pressure, per, MgO_liq))[0]
temperatures_MgO = np.linspace(1400., T_melt, 101)
entropy_melting_MgO = np.empty_like(temperatures_MgO)
for i, temperature in enumerate(temperatures_MgO):
    MgO_liq.set_state(pressure, temperature)
    V_solid=volume_solid(pressure, temperature, per.params)
    entropy_melting_MgO[i] = MgO_liq.S - entropy_solid(V_solid, temperature, per.params)



plt.plot(temperatures_SiO2, entropy_melting_SiO2, label='SiO2')
plt.plot(temperatures_MgO, entropy_melting_MgO, label='MgO')

plt.plot(temperatures_SiO2[100], entropy_melting_SiO2[100], marker='o', linestyle='None', label='SiO2')
plt.plot(temperatures_MgO[100], entropy_melting_MgO[100], marker='o', linestyle='None', label='MgO')

plt.xlabel('Temperature (K)')
plt.ylabel('Entropy of melting (J/K/mol)')
plt.legend(loc='upper right')
plt.xlim(1400, 4300)
plt.show()

