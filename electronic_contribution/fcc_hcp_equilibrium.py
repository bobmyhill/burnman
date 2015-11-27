import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from scipy.optimize import fsolve
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
from listify_xy_file import *
from fitting_functions import *

import matplotlib.image as mpimg

from fcc_iron import fcc_iron
from hcp_iron import hcp_iron


bcc = minerals.HP_2011_ds62.iron()
fcc = fcc_iron()
hcp = hcp_iron()


# Metastable fcc <-> hcp at 1 bar is ca. 500 K
# bcc <-> hcp at 300 K is at ca. 12 GPa
print burnman.tools.equilibrium_temperature([fcc, hcp], [1.0, -1.0], 1.e5)
print burnman.tools.equilibrium_pressure([bcc, hcp], [1.0, -1.0], 298.15)/1.e9




fig1 = mpimg.imread('data/Anzellini_2013_Fe_melting.png')  # Uncomment these two lines if you want to overlay the plot on a screengrab from SLB2011
plt.imshow(fig1, extent=[0., 230., 1200., 5200.], aspect='auto')



# Find triple point
P_inv, T_inv = burnman.tools.invariant_point([fcc, hcp], [1.0, -1.0],\
                                             [bcc, hcp], [1.0, -1.0],\
                                             [10.e9, 800.])

temperatures = np.linspace(298.15, T_inv, 11)
pressures = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    pressures[i] = burnman.tools.equilibrium_pressure([bcc, hcp], [1.0, -1.0], T, 10.e9)
plt.plot(pressures/1.e9, temperatures)

pressures = np.linspace(1.e5, P_inv, 11)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = burnman.tools.equilibrium_temperature([bcc, fcc], [1.0, -1.0], P, 1000.)
plt.plot(pressures/1.e9, temperatures)

temperatures = np.linspace(T_inv, 3600., 11)
pressures = np.empty_like(temperatures)
fcc_volumes = np.empty_like(pressures)
hcp_volumes = np.empty_like(pressures)
fcc_entropies = np.empty_like(pressures)
hcp_entropies = np.empty_like(pressures)

melting_curve = listify_xy_file('data/Anzellini_2013_Fe_melting_curve.dat')
Pmelt, Tmelt, phasemelt = melting_curve
plt.plot(Pmelt, Tmelt)

for i, T in enumerate(temperatures):
    pressures[i] = burnman.tools.equilibrium_pressure([fcc, hcp], [1.0, -1.0], T, 100.e9)
    fcc_volumes[i] = fcc.V
    hcp_volumes[i] = hcp.V
    fcc_entropies[i] = fcc.S
    hcp_entropies[i] = hcp.S

plt.plot(pressures/1.e9, temperatures)
plt.plot([98., 116.], [3635., 3862.])
plt.ylim(300., 7000.)
plt.xlim(0., 350.)
plt.xlabel("Pressure (GPa)")
plt.ylabel("Temperature (K)")
plt.show()
    
plt.plot(pressures/1.e9, fcc_volumes, label='fcc')
plt.plot(pressures/1.e9, hcp_volumes, label='hcp')
plt.legend(loc="lower left")
plt.show()

plt.plot(pressures/1.e9, fcc_entropies, label='fcc')
plt.plot(pressures/1.e9, hcp_entropies, label='hcp')
plt.legend(loc="lower left")
plt.show()
