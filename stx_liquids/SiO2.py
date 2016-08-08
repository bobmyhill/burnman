import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import DKS_2013_solids
from burnman.minerals import DKS_2013_liquids
from burnman import constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

SiO2_liq = DKS_2013_liquids.SiO2_liquid()

T = 4000.
pressures = np.linspace(20.e9, 150.e9, 31)
volumes = np.empty_like(pressures)
for i, P in enumerate(pressures):
    SiO2_liq.set_state(P, T)
    volumes[i] = SiO2_liq.V


plt.plot(pressures/1.e9, volumes)
plt.show()
