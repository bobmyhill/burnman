# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import fsolve

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import Mineral, minerals
from burnman.processchemistry import dictionarize_formula, formula_mass

HS_per = minerals.SLB_2011.periclase()
LS_per = minerals.SLB_2011.periclase()

Z = 4 # number of formula units per unit cell

# These are parameters from Solomatova et al. (2016), as
# estimated for the results from Lin et al. (2005) for Fe0.17Mg0.83O
HS_per.params['V_0'] = 75.94e-30 * burnman.constants.Avogadro / Z
HS_per.params['K_0'] =  160.e9
HS_per.params['Kprime_0'] = 4.04

LS_per.params['V_0'] = 72.29e-30 * burnman.constants.Avogadro / Z
LS_per.params['K_0'] =  190.e9
LS_per.params['Kprime_0'] = 4.

'''
# These are parameters from Solomatova et al. (2016), as
# estimated for the results from Marquardt et al., (2009) for Fe0.1Mg0.9O
HS_per.params['V_0'] = 75.55e-30 * burnman.constants.Avogadro / Z
HS_per.params['K_0'] =  159.e9
HS_per.params['Kprime_0'] = 3.96

LS_per.params['V_0'] = 74.59e-30 * burnman.constants.Avogadro / Z
LS_per.params['K_0'] =  159.e9
LS_per.params['Kprime_0'] = 4.
'''


Ohnishi_Delta0 = 11800.

P0 = 1.e5
T = 300.
E0s = [6300., 8900., 12400.]  # approximate values from Keppler et al.
E0s = [Ohnishi_Delta0]  # used by Clendenen and Drickamer (1966), originally evaluated by Hush and Pryce (1958) from the thermodynamical data

pressures = np.linspace(P0, 100.e9, 101)
volumes = np.empty_like(pressures)
energies = [np.empty_like(pressures) for i in range(len(E0s))]

HS_per.set_state(P0, T)
V0 = HS_per.V

for j, E0 in enumerate(E0s):
    for i, P in enumerate(pressures):
        if P < 52.e9:
            m = HS_per
        else:
            m = LS_per
            
        m.set_state(P, T)
        volumes[i] = m.V
        energies[j][i] = E0*np.power((volumes[i]/V0), -5./3.) # using a Maxwell model




img=mpimg.imread('Keppler_et_al_2007_figure_4.png')
plt.imshow(img, extent=[0,90,0,18000], aspect='auto')

#plt.plot(pressures/1.e9, volumes*1.e6)
for E in energies:
    plt.plot(pressures/1.e9, E)
plt.show()

