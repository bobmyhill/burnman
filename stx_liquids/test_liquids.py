import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from eos.liquid_EoS import _pressure, _electronic_excitation_entropy, helmholtz_free_energy
from minerals.deKoker_liquids import *
from burnman import constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




SiO2_liq=SiO2_liquid()
MgO_liq=MgO_liquid()

# see page 15 of supplementary material for SiO2 volume-pressure and volume-energy benchmark
# For a volume of 11 cm^3/mol and temperature of 7000 K, pressure should be ca. 170 GPa  
# Energy (E, not F or G) should be ca. -1560 kJ/mol
temperature=7000. # K
volume=11e-6 # m^3/mol
print _pressure(volume, temperature, SiO2_liq.params)/1e9

temperature=3000. # K
volume=0.2780000000E+02*1e-6 # m^3/mol
print _pressure(volume, temperature, SiO2_liq.params)/1e9


fig1 = mpimg.imread('figures/SiO2_liquid_PVT.png')
plt.imshow(fig1, extent=[9, 30, -10, 220], aspect='auto')

temperatures=np.linspace(2000., 7000., 6)
volumes=np.linspace(9e-6, 30e-6, 101)
pressures=np.empty_like(volumes)
for temperature in temperatures:
    for i, volume in enumerate(volumes):
        pressures[i]=_pressure(volume, temperature, SiO2_liq.params)/1e9
    plt.plot(volumes*1e6, pressures, linewidth=2, label=str(temperature)+'K')

plt.legend(loc='upper right')
plt.xlim(9,30)
plt.ylim(-10,220)
plt.show()

fig1 = mpimg.imread('figures/SiO2_liquid_SelVT.png')
plt.imshow(fig1, extent=[9, 30, -0.03, 0.75], aspect='auto')

temperatures=np.linspace(2000., 7000., 6)
volumes=np.linspace(9e-6, 30e-6, 101)
entropies=np.empty_like(volumes)
for temperature in temperatures:
    for i, volume in enumerate(volumes):
        entropies[i]=_electronic_excitation_entropy(volume, temperature, SiO2_liq.params)
    plt.plot(volumes*1e6, entropies/3./constants.gas_constant, linewidth=2, label=str(temperature)+'K')

plt.legend(loc='upper right')
plt.xlim(9,30)
plt.show()

fig1 = mpimg.imread('figures/SiO2_liquid_EVT.png')
plt.imshow(fig1, extent=[9, 30, -2400, -1200], aspect='auto')

temperatures=np.linspace(2000., 7000., 6)
volumes=np.linspace(9e-6, 30e-6, 101)
energies=np.empty_like(volumes)
for temperature in temperatures:
    for i, volume in enumerate(volumes):
        dT=1e-4
        F = helmholtz_free_energy(volume, temperature, SiO2_liq.params)
        S = (F - helmholtz_free_energy(volume, temperature+dT, SiO2_liq.params))/dT
        energies[i]=F + temperature*S
    plt.plot(volumes*1e6, energies/1e3, linewidth=2, label=str(temperature)+'K')

plt.legend(loc='upper right')
plt.xlim(9,30)
plt.show()

# Plot MgO
fig1 = mpimg.imread('figures/MgO_liquid_PVT.png')
plt.imshow(fig1, extent=[7, 18, -6, 240], aspect='auto')

temperatures=np.linspace(2000., 7000., 6)
volumes=np.linspace(7e-6, 18e-6, 101)
pressures=np.empty_like(volumes)
for temperature in temperatures:
    for i, volume in enumerate(volumes):
        pressures[i]=_pressure(volume, temperature, MgO_liq.params)/1e9
    plt.plot(volumes*1e6, pressures, linewidth=2, label=str(temperature)+'K')

temperature = 10000. # K
for i, volume in enumerate(volumes):
    pressures[i]=_pressure(volume, temperature, MgO_liq.params)/1e9
plt.plot(volumes*1e6, pressures, linewidth=2, label=str(temperature)+'K')

    
plt.legend(loc='upper right')
plt.xlim(7,18)
plt.ylim(-6,240)
plt.show()


fig1 = mpimg.imread('figures/MgO_liquid_SelVT.png')
plt.imshow(fig1, extent=[6, 18, -0.04, 0.84], aspect='auto')

temperatures=np.linspace(2000., 7000., 6)
volumes=np.linspace(7e-6, 18e-6, 101)
entropies=np.empty_like(volumes)
for temperature in temperatures:
    for i, volume in enumerate(volumes):
        entropies[i]=_electronic_excitation_entropy(volume, temperature, MgO_liq.params)
    plt.plot(volumes*1e6, entropies/2./constants.gas_constant, linewidth=2, label=str(temperature)+'K')

temperature = 10000. # K
for i, volume in enumerate(volumes):
    entropies[i]=_electronic_excitation_entropy(volume, temperature, MgO_liq.params)
plt.plot(volumes*1e6, entropies/2./constants.gas_constant, linewidth=2, label=str(temperature)+'K')

    
plt.legend(loc='upper right')
plt.xlim(6,18)
plt.ylim(-0.04,0.84)
plt.show()


fig1 = mpimg.imread('figures/MgO_liquid_EVT.png')
plt.imshow(fig1, extent=[7, 18, -1200, -200], aspect='auto')

temperatures=np.linspace(2000., 7000., 6)
volumes=np.linspace(7e-6, 18e-6, 101)
energies=np.empty_like(volumes)
for temperature in temperatures:
    for i, volume in enumerate(volumes):
        dT=1e-4
        F = helmholtz_free_energy(volume, temperature, MgO_liq.params)
        S = (F - helmholtz_free_energy(volume, temperature+dT, MgO_liq.params))/dT
        energies[i]=F + temperature*S
    plt.plot(volumes*1e6, energies/1e3, linewidth=2, label=str(temperature)+'K')

temperature=10000. # K
for i, volume in enumerate(volumes):
    dT=1e-4
    F = helmholtz_free_energy(volume, temperature, MgO_liq.params)
    S = (F - helmholtz_free_energy(volume, temperature+dT, MgO_liq.params))/dT
    energies[i]=F + temperature*S
plt.plot(volumes*1e6, energies/1e3, linewidth=2, label=str(temperature)+'K')

plt.legend(loc='upper right')
plt.xlim(7,18)
plt.show()


