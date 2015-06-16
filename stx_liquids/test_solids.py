import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from eos.solid_EoS import *
from minerals.deKoker_solids import *
from burnman import constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
SOLIDS
'''

stv=stishovite()


fig1 = mpimg.imread('figures/stishovite_PVT.png')
plt.imshow(fig1, extent=[10, 18, -25, 175], aspect='auto')

temperatures=np.linspace(2000., 6000., 5)
volumes=np.linspace(10e-6, 18e-6, 101)
pressures=np.empty_like(volumes)
for temperature in temperatures:
    for i, volume in enumerate(volumes):
        pressures[i]=pressure_solid(volume, temperature, stv.params)
    plt.plot(volumes*1e6, pressures/1e9, linewidth=2, label=str(temperature)+'K')

plt.legend(loc='upper right')
plt.xlim(10,18)
plt.show()



fig1 = mpimg.imread('figures/stishovite_EVT.png')
plt.imshow(fig1, extent=[10, 18, -2400, -1800], aspect='auto')

temperatures=np.linspace(2000., 6000., 5)
volumes=np.linspace(10e-6, 18e-6, 101)
energies=np.empty_like(volumes)
for temperature in temperatures:
    for i, volume in enumerate(volumes):
        energies[i]=energy_solid(volume, temperature, stv.params)
    plt.plot(volumes*1e6, energies/1e3, linewidth=2, label=str(temperature)+'K')

plt.legend(loc='upper right')
plt.xlim(10,18)
plt.show()


pv=perovskite()

fig1 = mpimg.imread('figures/perovskite_PVT.png')
plt.imshow(fig1, extent=[14.5, 27.5, 0, 344], aspect='auto')

temperatures=np.linspace(2000., 6000., 5)
volumes=np.linspace(14.5e-6, 27.5e-6, 101)
pressures=np.empty_like(volumes)
for temperature in temperatures:
    for i, volume in enumerate(volumes):
        pressures[i]=pressure_solid(volume, temperature, pv.params)
    plt.plot(volumes*1e6, pressures/1e9, linewidth=2, label=str(temperature)+'K')

plt.legend(loc='upper right')
plt.xlim(14.5,27.5)
plt.show()


fig1 = mpimg.imread('figures/perovskite_EVT.png')
plt.imshow(fig1, extent=[14.5, 27.5, -3600, -2000], aspect='auto')

temperatures=np.linspace(2000., 6000., 5)
volumes=np.linspace(14.5e-6, 27.5e-6, 101)
energies=np.empty_like(volumes)
for temperature in temperatures:
    for i, volume in enumerate(volumes):
        energies[i]=energy_solid(volume, temperature, pv.params)
    plt.plot(volumes*1e6, energies/1e3, linewidth=2, label=str(temperature)+'K')

plt.legend(loc='upper right')
plt.xlim(14.5,27.5)
plt.show()

per=periclase()

fig1 = mpimg.imread('figures/periclase_PVT.png')
plt.imshow(fig1, extent=[6.5, 14, -25, 275], aspect='auto')

temperatures=np.linspace(2000., 8000., 4)
volumes=np.linspace(6.5e-6, 14e-6, 101)
pressures=np.empty_like(volumes)
for temperature in temperatures:
    for i, volume in enumerate(volumes):
        pressures[i]=pressure_solid(volume, temperature, per.params)
    plt.plot(volumes*1e6, pressures/1e9, linewidth=2, label=str(temperature)+'K')

plt.legend(loc='upper right')
plt.xlim(6.5,14)
plt.show()


fig1 = mpimg.imread('figures/periclase_EVT.png')
plt.imshow(fig1, extent=[6.5, 14, -1200, -560], aspect='auto')

temperatures=np.linspace(2000., 8000., 4)
volumes=np.linspace(6.5e-6, 14e-6, 101)
energies=np.empty_like(volumes)
for temperature in temperatures:
    for i, volume in enumerate(volumes):
        energies[i]=energy_solid(volume, temperature, per.params)
    plt.plot(volumes*1e6, energies/1e3, linewidth=2, label=str(temperature)+'K')

plt.legend(loc='upper right')
plt.xlim(6.5,14)
plt.show()

