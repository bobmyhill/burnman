import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import DKS_2013_solids
from burnman.minerals import DKS_2013_liquids
from burnman import constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pv = DKS_2013_solids.perovskite()
pv_liq = DKS_2013_liquids.MgSiO3_liquid()

pressures = np.linspace(20.e9, 150.e9, 31)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = burnman.tools.equilibrium_temperature([pv, pv_liq], [1.0, -1.0], P, 4000.)


plt.plot(pressures, temperatures)
plt.show()
