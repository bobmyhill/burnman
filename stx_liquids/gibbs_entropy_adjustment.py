import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    DKS_2013_liquids, \
    DKS_2013_liquids_tweaked, \
    DKS_2013_solids, \
    SLB_2011, \
    HP_2011_ds62
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
stv=SLB_2011.stishovite()
coe=SLB_2011.coesite()
SiO2_liq=DKS_2013_liquids_tweaked.SiO2_liquid()
SiO2_liq_alt=DKS_2013_liquids_tweaked.SiO2_liquid_alt()

pressure = 13.7e9 # Pa
T_melt = fsolve(find_temperature, 5000., args=(pressure, stv, SiO2_liq))[0]
print 'Stishovite T_melt (13.7 GPa):', T_melt, 'K, should be 3073.15 K'
T_melt = fsolve(find_temperature, 5000., args=(pressure, stv, SiO2_liq_alt))[0]
print 'Stishovite T_melt (13.7 GPa):', T_melt, 'K, should be 3073.15 K'

pressures = np.linspace(10.e9, 14.e9, 5)
temperatures = np.empty_like(pressures)
temperatures_alt = np.empty_like(pressures)
for i, pressure in enumerate(pressures):    
    temperatures[i] = fsolve(find_temperature, 2000., args=(pressure, stv, SiO2_liq))
    temperatures_alt[i] = fsolve(find_temperature, 2000., args=(pressure, stv, SiO2_liq_alt))

plt.plot(pressures, temperatures, label='stv')
plt.plot(pressures, temperatures_alt, label='stv (alt)')

for i, pressure in enumerate(pressures):    
    temperatures[i] = fsolve(find_temperature, 2000., args=(pressure, coe, SiO2_liq))
    temperatures_alt[i] = fsolve(find_temperature, 2000., args=(pressure, coe, SiO2_liq_alt))

plt.plot(pressures, temperatures, label='coe')
plt.plot(pressures, temperatures_alt, label='coe (alt)')
plt.show()
'''

def find_temperature_mul(temperature, pressure, solid, liquid, factor):
    liquid.set_state(pressure, temperature[0])
    solid.set_state(pressure, temperature[0])
    return solid.gibbs - (liquid.gibbs)*factor


'''
Find stishovite delta gibbs and delta entropy
'''

print 'Corrections to solids'
print '.....................'

phases=[[HP_2011_ds62.fo(), SLB_2011.forsterite()],
        [HP_2011_ds62.stv(), DKS_2013_solids.stishovite()],
        [SLB_2011.stishovite(), DKS_2013_solids.stishovite()],
        [SLB_2011.periclase(), DKS_2013_solids.periclase()],
        [SLB_2011.mg_perovskite(), DKS_2013_solids.perovskite()]]

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
        print temperature, MD_phase.S - SLB_phase.S, MD_phase.gibbs - SLB_phase.gibbs
    print ''

print ''


# check coesite-stishovite
coe=SLB_2011.coesite()
stv=DKS_2013_solids.stishovite()

pressure_c_s = 13.8e9 # Pa
temperature_c_s = 2800. + 273.15 # K

stv.set_state(pressure_c_s, temperature_c_s)
coe.set_state(pressure_c_s, temperature_c_s)
correction = stv.gibbs - coe.gibbs
print 'corr', correction


def find_stv_coe_temperature(T, P):
    stv.set_state(P, T[0])
    coe.set_state(P, T[0])
    return stv.gibbs - coe.gibbs - correction

pressures=np.linspace(8.e9, 17.e9, 10)
for pressure in pressures:
    print pressure, fsolve(find_stv_coe_temperature, 2000.,  args=(pressure))

stv=SLB_2011.stishovite()
correction = 0.
print 'corr:', correction
pressures=np.linspace(8.e9, 17.e9, 10)
for pressure in pressures:
    print pressure, fsolve(find_stv_coe_temperature, 2000.,  args=(pressure))

print 'coesite  - SiO2_liquid volumes'
pressure = 10.e9 # Pa
temperature = 3120. # K
SiO2_liq=DKS_2013_liquids.SiO2_liquid()
coe.set_state(pressure, temperature)
SiO2_liq.set_state(pressure, temperature)

print 'Volumes', coe.V, SiO2_liq.V

print ''
print 'Temperatures of melting'
print '.........................'

fo=SLB_2011.forsterite()
fo_liq=DKS_2013_liquids_tweaked.Mg2SiO4_liquid()
pressure = 1.e5 # Pa
temperature = 2163. # K
fo_liq.set_state(pressure, temperature)
fo.set_state(pressure, temperature)
print fo_liq.gibbs - fo.gibbs

pressure = 14.e9
T_melt = fsolve(find_temperature, 3000., args=(pressure, fo, fo_liq))[0]
print 'Forsterite T_melt (14 GPa):', T_melt, 'K, should be 2580 K'


cen=SLB_2011.hp_clinoenstatite()
oen=SLB_2011.enstatite()
en_liq=DKS_2013_liquids_tweaked.MgSiO3_liquid()
pressure = 0.e9 # Pa
temperature = 1850. # K
en_liq.set_state(pressure, temperature)
oen.set_state(pressure, temperature)
print en_liq.gibbs - oen.gibbs/2.


pressure = 0.e9
T_melt = fsolve(find_temperature_mul, 2000., args=(pressure, oen, en_liq, 2.))[0]
print 'Orthoenstatite T_melt (0 GPa):', T_melt, 'K, should be 1850 K'
pressure = 13.3e9
T_melt = fsolve(find_temperature_mul, 3000., args=(pressure, cen, en_liq, 2.))[0]
print 'Clinoenstatite T_melt (13.3 GPa):', T_melt, 'K, should be 2563 K'

# 3430 with 0
# 3540 with 20

# dS/dV = dP/dT


pv=DKS_2013_solids.perovskite()
pv_liq=DKS_2013_liquids.MgSiO3_liquid()
pressure = 25.e9 # Pa
T_melt = fsolve(find_temperature, 5000., args=(pressure, pv, pv_liq))[0]
print 'Perovskite T_melt (25 GPa):', T_melt, 'K, should be 2900 K'

#stv=DKS_2013_solids.stishovite()
#SiO2_liq=DKS_2013_liquids.SiO2_liquid()

stv=SLB_2011.stishovite()
SiO2_liq=DKS_2013_liquids_tweaked.SiO2_liquid()
SiO2_liq_alt=DKS_2013_liquids_tweaked.SiO2_liquid_alt()
pressure = 13.7e9 # Pa
T_melt = fsolve(find_temperature, 5000., args=(pressure, stv, SiO2_liq))[0]
print 'Stishovite T_melt (13.7 GPa):', T_melt, 'K, should be 3073.15 K'
T_melt = fsolve(find_temperature, 5000., args=(pressure, stv, SiO2_liq_alt))[0]
print 'Stishovite T_melt (13.7 GPa):', T_melt, 'K, should be 3073.15 K'


    
#per=DKS_2013_solids.periclase()
#MgO_liq=DKS_2013_liquids.MgO_liquid()

per=SLB_2011.periclase()
MgO_liq=DKS_2013_liquids_tweaked.MgO_liquid()
pressure = 1.e5 # Pa
T_melt = fsolve(find_temperature, 3000., args=(pressure, per, MgO_liq))[0]

print 'Periclase T_melt (0 GPa):', T_melt, 'K, should be 3070 K'

print ''
print 'Liquid entropy of melting'
print '.........................'

pressure = 14.e9 # Pa

temperatures_SiO2 = np.linspace(1400., T_melt, 101)
entropy_melting_SiO2 = np.empty_like(temperatures_SiO2)
for i, temperature in enumerate(temperatures_SiO2):
    SiO2_liq.set_state(pressure, temperature)
    stv.set_state(pressure, temperature)
    entropy_melting_SiO2[i] = SiO2_liq.S - stv.S


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



temperatures_SiO2 = np.linspace(1400., 3100., 101)
dTdP_SiO2_alt = np.empty_like(temperatures_SiO2)
dTdP_SiO2 = np.empty_like(temperatures_SiO2)
for i, temperature in enumerate(temperatures_SiO2):
    SiO2_liq.set_state(pressure, temperature)
    SiO2_liq_alt.set_state(pressure, temperature)
    stv.set_state(pressure, temperature)
    dTdP_SiO2[i] = (SiO2_liq.V - stv.V) / (SiO2_liq.S - stv.S)
    dTdP_SiO2_alt[i] = (SiO2_liq_alt.V - stv.V) / (SiO2_liq_alt.S - stv.S)

plt.plot(temperatures_SiO2, dTdP_SiO2, label='SiO2')
plt.plot(temperatures_SiO2, dTdP_SiO2_alt, label='SiO2')
plt.show()


