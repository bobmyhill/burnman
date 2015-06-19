import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    DKS_2013_liquids, \
    DKS_2013_solids, \
    SLB_2011
from burnman import constants
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def find_temperature(temperature, pressure, solid, liquid):
    liquid.set_state(pressure, temperature[0])
    solid.set_state(pressure, temperature[0])
    return solid.gibbs - liquid.gibbs


'''
Find stishovite delta gibbs and delta entropy
'''

print 'Corrections to solids'
print '.....................'

phases=[[SLB_2011.stishovite(), DKS_2013_solids.stishovite()],
        [SLB_2011.periclase(), DKS_2013_solids.periclase()]]

for SLB_phase, MD_phase in phases:
    print SLB_phase.params['name']
    
    pressure = 14.e9 # Pa
    temperature_0 = 3000. # K

    SLB_phase.set_state(pressure, temperature_0)
    MD_phase.set_state(pressure, temperature_0)
    stv_delta_G=MD_phase.gibbs - SLB_phase.gibbs

    print 'delta_gibbs:', stv_delta_G

    temperatures=[1500, 2000, 2500, 3000, 3500, 4000, 4500]

    for temperature in temperatures:
        SLB_phase.set_state(pressure, temperature)
        MD_phase.set_state(pressure, temperature)
        print temperature, MD_phase.S - SLB_phase.S
    print ''

print ''


print 'Liquid entropy of melting'
print '.........................'


stv=DKS_2013_solids.stishovite()
SiO2_liq=DKS_2013_liquids.SiO2_liquid()
pressure = 14.e9 # Pa
T_melt = fsolve(find_temperature, 5000., args=(pressure, stv, SiO2_liq))[0]
print 'Stishovite T_melt:', T_melt, 'K'


temperatures_SiO2 = np.linspace(1400., T_melt, 101)
entropy_melting_SiO2 = np.empty_like(temperatures_SiO2)
for i, temperature in enumerate(temperatures_SiO2):
    SiO2_liq.set_state(pressure, temperature)
    stv.set_state(pressure, temperature)
    entropy_melting_SiO2[i] = SiO2_liq.S - stv.S


per=DKS_2013_solids.periclase()
MgO_liq=DKS_2013_liquids.MgO_liquid()
pressure = 14.e9 # Pa
T_melt = fsolve(find_temperature, 5000., args=(pressure, per, MgO_liq))[0]

print 'Periclase T_melt:', T_melt, 'K'


temperatures_MgO = np.linspace(1400., T_melt, 101)
entropy_melting_MgO = np.empty_like(temperatures_MgO)
for i, temperature in enumerate(temperatures_MgO):
    MgO_liq.set_state(pressure, temperature)
    per.set_state(pressure, temperature)
    entropy_melting_MgO[i] = MgO_liq.S - per.S



plt.plot(temperatures_SiO2, entropy_melting_SiO2, label='SiO2')
plt.plot(temperatures_MgO, entropy_melting_MgO, label='MgO')

plt.plot(temperatures_SiO2[100], entropy_melting_SiO2[100], marker='o', linestyle='None', label='SiO2')
plt.plot(temperatures_MgO[100], entropy_melting_MgO[100], marker='o', linestyle='None', label='MgO')

plt.xlabel('Temperature (K)')
plt.ylabel('Entropy of melting (J/K/mol)')
plt.legend(loc='upper right')
plt.xlim(1400, 4300)
plt.show()

