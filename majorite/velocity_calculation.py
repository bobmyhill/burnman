# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

"""
calculates velocities
"""

import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals

# density = mass / volume
def Vp(mineral):
    return np.sqrt((mineral.K_T + 4./3.*mineral.G)*mineral.V/mineral.molar_mass())
def Vs(mineral):
    return np.sqrt((mineral.G)*mineral.V/mineral.molar_mass())


g = minerals.majorite.garnet()

# py, alm, gr, maj
composition = [0.24, 0.17, 0.39, 0.2]
pressure = 20.e9 # Pa
temperature = 1600. + 273.15 # K

g.set_composition(composition)
g.set_state(pressure, temperature)

#####

g2 = minerals.majorite.ideal_garnet()

# py, alm, gr, maj
composition = [0.24, 0.17, 0.39, 0.2]
pressure = 20.e9 # Pa
temperature = 1600. + 273.15 # K

g2.set_composition(composition)
g2.set_state(pressure, temperature)

print 'V_p, V_s output'
print ''
print 'Nonideal velocities (km/s):', Vp(g)/1000., Vs(g)/1000., g.V*1e6
print 'Ideal velocities (km/s):', Vp(g2)/1000., Vs(g2)/1000., g2.V*1e6


print ''
print '(Nonideal garnet has ideal mixing for majorite'
print ' - other parameters come from Ganguly et al., 1996)'
