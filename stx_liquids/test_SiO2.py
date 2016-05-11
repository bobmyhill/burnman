import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import DKS_2013_solids, DKS_2013_liquids, DKS_2013_liquids_tweaked, SLB_2011
from burnman import constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
SOLIDS
'''

stv = DKS_2013_solids.stishovite()
stv_2 = SLB_2011.stishovite()
coe = SLB_2011.coesite()
liquid = DKS_2013_liquids.SiO2_liquid()
liquid_2 = DKS_2013_liquids_tweaked.SiO2_liquid()


T = 2973.15
P = 13.e9

stv.set_state(P, T)
stv_2.set_state(P, T)
liquid.set_state(P, T)
liquid_2.set_state(P, T)

print liquid.S - liquid_2.S
print stv.S - stv_2.S

pressures = np.linspace(1.e9, 50.e9, 21)
T1 = np.empty_like(pressures)
T2 = np.empty_like(pressures)
for i, P in enumerate(pressures):
    try:
        T1[i] = burnman.tools.equilibrium_temperature([stv_2, liquid_2], [1.0, -1.0], P, 2000.)
    except:
        print()

    try:
        T2[i] = burnman.tools.equilibrium_temperature([coe, liquid_2], [1.0, -1.0], P, 2000.)
    except:
        print()
        
plt.plot(pressures/1.e9, T1)
plt.plot(pressures/1.e9, T2)
plt.show()
