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

bcc = burnman.minerals.HP_2011_ds62.iron()
    
bcc.set_state(1.e5, 300.)
hcp.params['F_0'] = bcc.helmholtz + 3800.
fcc.params['F_0'] = bcc.helmholtz + 3800.
print bcc.helmholtz + 3800.
pressures = np.linspace(0., 100.e6, 11)
fcc_volumes = np.empty_like(pressures)
hcp_volumes = np.empty_like(pressures)
fcc_entropies = np.empty_like(pressures)
hcp_entropies = np.empty_like(pressures)

for i, P in enumerate(pressures):
    print P/1.e9, burnman.tools.equilibrium_temperature([fcc, hcp], [1.0, -1.0], P, 2000.)
    print fcc.gibbs - hcp.gibbs
    fcc_volumes[i] = fcc.V
    hcp_volumes[i] = hcp.V
    fcc_entropies[i] = fcc.S
    hcp_entropies[i] = hcp.S

plt.plot(pressures/1.e9, fcc_volumes, label='fcc')
plt.plot(pressures/1.e9, hcp_volumes, label='hcp')
plt.legend(loc="lower left")
plt.show()

plt.plot(pressures/1.e9, fcc_entropies, label='fcc')
plt.plot(pressures/1.e9, hcp_entropies, label='hcp')
plt.legend(loc="lower left")
plt.show()
