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

from fcc_iron import fcc_iron
from hcp_iron import hcp_iron

fcc = fcc_iron()
hcp = hcp_iron()

# FCC gibbs free energy is -56656 J/mol at 1200 K
G_1200 = -56656.
fcc.set_state(1.e5, 1200.)
print fcc.gibbs
fcc.params['F_0'] = fcc.params['F_0'] - (fcc.gibbs - G_1200)

# Metastable fcc <-> hcp at 1 bar is ca. 500 K
# bcc <-> hcp at 300 K is at ca. 12 GPa
fcc.set_state(1.e5, 500.)
hcp.set_state(1.e5, 500.)
hcp.params['F_0'] = hcp.params['F_0'] - (hcp.gibbs - fcc.gibbs)



temperatures = np.linspace(500., 4000., 11)
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
