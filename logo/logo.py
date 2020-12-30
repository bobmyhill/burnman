# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
"""
from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.interpolate import griddata
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import minerals


fo = minerals.SLB_2011.forsterite()


pressures = np.logspace(5., 12., 101)
temperatures = np.logspace(0., 3.4, 201)

'''
PP, TT = np.meshgrid(pressures, temperatures)
Vs, Cps = fo.evaluate(['V', 'molar_heat_capacity_p'], PP.ravel(), TT.ravel())
xi = temperatures
yi = np.linspace(0., burnman.constants.gas_constant*3.*7., 1001)
zi = griddata((TT.ravel(), Cps), PP.ravel(), (xi[None,:], yi[:,None]), method='cubic')
CS = plt.contourf(xi,yi,zi,cmap=plt.cm.jet)
'''

f = 1./5.
cNorm  = colors.Normalize(vmin=np.power(pressures[0], f), vmax=np.power(pressures[-1], f)*1.01)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.cm.afmhot)


n = 21. # fo.params['n'] for cropped image at 3nR

for p in pressures:
    ps = [p]*len(temperatures)
    Cps = fo.evaluate(['molar_heat_capacity_p'], ps, temperatures)[0]
    mask = [i for i, Cp in enumerate(Cps) if Cp<n*3.*burnman.constants.gas_constant] 
    plt.plot(temperatures[mask], Cps[mask], color=scalarMap.to_rgba(np.power(p, f)))


plt.savefig('u.pdf') 
plt.show()   
