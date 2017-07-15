# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.

from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import minerals, tools


# Let's instantiate pyropes from two different databases
py_SLB = minerals.SLB_2011.pyrope()
py_HP = minerals.HP_2011_ds62.py()


# We can check our equation of state consistency
# (making use of Maxwell's Relations)
tools.check_eos_consistency(py_SLB, verbose=True)
print('')
                      
# Here's how we do a single P, T evaluation
P = 1.e5 # Pa
T = 300.
py_SLB.set_state(P, T)
print('Gibbs energy of SLB pyrope at {0} GPa and {1} K: {2}'.format(P/1.e9, T, py_SLB.gibbs))

# And here's how we do lots of evaluations at different states
pressures = np.array([1.e5] * 101)
temperatures = np.linspace(300., 1300., 101)
Cp_SLB = py_SLB.evaluate(['heat_capacity_p'],
                         pressures, temperatures)[0]
Cp_HP = py_HP.evaluate(['heat_capacity_p'],
                       pressures, temperatures)[0]

# Plot heat capacities!
plt.plot(temperatures, Cp_SLB, label='SLB')
plt.plot(temperatures, Cp_HP, label='HP')
plt.legend(loc='lower right')
plt.xlabel('Temperatures (K)')
plt.ylabel('Heat capacity (J/K/mol)')
plt.show()


"""
We can create an instance of a solid solution in two ways. 
We can create a single instance using the SolidSolution constructor
(here's an example of an ideal pyrope-almandine garnet) ...
"""
g1 = burnman.SolidSolution(name = 'Ideal pyrope-almandine garnet',
                           solution_type = 'ideal',
                           endmembers = [[minerals.HP_2011_ds62.py(),
                                          '[Mg]3[Al]2Si3O12'],
                                         [minerals.HP_2011_ds62.alm(),
                                          '[Fe]3[Al]2Si3O12']],
                           molar_fractions = [0.5, 0.5])

g2 = burnman.SolidSolution(name = 'asymmetric garnet',
                           solution_type = 'asymmetric',
                           endmembers = [[minerals.HP_2011_ds62.py(),
                                          '[Mg]3[Al]2Si3O12'],
                                         [minerals.HP_2011_ds62.alm(),
                                          '[Fe]3[Al]2Si3O12'],
                                         [minerals.HP_2011_ds62.gr(),
                                          '[Ca]3[Al]2Si3O12'],
                                         [minerals.HP_2011_ds62.andr(),
                                          '[Ca]3[Fe]2Si3O12']],
                           alphas = [1.0, 1.0, 2.7, 2.7],
                           energy_interaction = [[2.5e3, 31.e3, 53.2e3],
                                                 [5.e3, 37.24e3],
                                                 [2.e3]])

g2.set_composition([0.1, 0.2, 0.3, 0.4])
g2.set_state(1.e5, 300.)

print(g2.activities)


############################
# Ol polymorph example
############################

import burnman
from burnman.minerals import SLB_2011
from burnman.mineral import Mineral
from burnman.solidsolution import SolidSolution
from burnman.solutionmodel import *
from burnman.processchemistry import *
from burnman.chemicalpotentials import *
from burnman import constants

from equilibrium_functions import *
from models import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10

from scipy import optimize


plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'

'''
Initialise solid solutions
'''


ol=mg_fe_olivine()
wd=mg_fe_wadsleyite()
rw=mg_fe_ringwoodite()

fig1 = mpimg.imread('figures/ol_polymorphs_1400C.png')
plt.imshow(fig1, extent=[0,1,6.0,20.], aspect='auto')


# Data from Dan Frost's experiments
ol_wad_data = []
ol_rw_data = []
wad_rw_data = []
for line in open('ol_polymorph_equilibria.dat'):
    content=line.strip().split()
    if content[0] != '%':
        # ol-wad
        if content[7] != '-' and content[9] != '-':
            ol_wad_data.append([float(content[2]), 0.5, float(content[3]), float(content[7]), float(content[8]), float(content[9]), float(content[10])])
        # ol-rw
        if content[7] != '-' and content[11] != '-':
            ol_rw_data.append([float(content[2]), 0.5, float(content[3]), float(content[7]), float(content[8]), float(content[11]), float(content[12])])
        # wd-rw
        if content[9] != '-' and content[11] != '-':
            wad_rw_data.append([float(content[2]), 0.5, float(content[3]), float(content[9]), float(content[10]), float(content[11]), float(content[12])])

ol_wad_data = zip(*ol_wad_data)
ol_rw_data = zip(*ol_rw_data)
wad_rw_data = zip(*wad_rw_data)



# Temperature of phase diagram
T=1673. # K

# Find invariant point
invariant=optimize.fsolve(eqm_P_xMgABC(ol, wd, rw), [15.e9, 0.2, 0.3, 0.4], args=(T))
print invariant

# Initialise arrays
XMgA_ol_wad=np.linspace(invariant[1], 0.9999, 15)
XMgA_ol_rw=np.linspace(0.0001, invariant[1], 15)
XMgA_wad_rw=np.linspace(invariant[2], 0.9999, 15)

P_ol_wad=np.empty_like(XMgA_ol_wad)
XMgB_ol_wad=np.empty_like(XMgA_ol_wad)

P_ol_rw=np.empty_like(XMgA_ol_wad)
XMgB_ol_rw=np.empty_like(XMgA_ol_wad)

P_wad_rw=np.empty_like(XMgA_ol_wad)
XMgB_wad_rw=np.empty_like(XMgA_ol_wad)


# Find transition pressures
for idx, XMgA in enumerate(XMgA_ol_wad):
    XMgB_guess=1.0-((1.0-XMgA_ol_wad[idx])*0.8)
    P_ol_wad[idx], XMgB_ol_wad[idx] = optimize.fsolve(eqm_P_xMgB(ol, wd), [5.e9, XMgB_guess], args=(T, XMgA_ol_wad[idx]))
    XMgB_guess=1.0-((1.0-XMgA_ol_rw[idx])*0.8)
    P_ol_rw[idx], XMgB_ol_rw[idx] = optimize.fsolve(eqm_P_xMgB(ol, rw), [5.e9, XMgB_guess], args=(T, XMgA_ol_rw[idx]))
    XMgB_guess=1.0-((1.0-XMgA_wad_rw[idx])*0.8)
    P_wad_rw[idx], XMgB_wad_rw[idx] = optimize.fsolve(eqm_P_xMgB(wd, rw), [5.e9, XMgB_guess], args=(T, XMgA_wad_rw[idx]))


# Plot data
plt.plot( 1.0-np.array([invariant[1], invariant[2], invariant[3]]), np.array([invariant[0], invariant[0], invariant[0]])/1.e9, color='black', linewidth=3, label='invariant')

plt.plot( 1.0-XMgA_ol_wad, P_ol_wad/1.e9, 'r-', linewidth=3, label='wad-out (ol, wad)')
plt.plot( 1.0-XMgB_ol_wad, P_ol_wad/1.e9, 'g-', linewidth=3, label='ol-out (ol, wad)')

plt.plot( 1.0-XMgA_ol_rw, P_ol_rw/1.e9, 'r-',  linewidth=3, label='rw-out (ol, rw)')
plt.plot( 1.0-XMgB_ol_rw, P_ol_rw/1.e9, 'b-',  linewidth=3, label='ol-out (ol, rw)')

plt.plot( 1.0-XMgA_wad_rw, P_wad_rw/1.e9, 'g-',  linewidth=3, label='rw-out (wad, rw)')
plt.plot( 1.0-XMgB_wad_rw, P_wad_rw/1.e9, 'b-',  linewidth=3, label='wad-out (wad, rw)')

plt.errorbar( ol_wad_data[3], ol_wad_data[0], xerr=[ol_wad_data[4], ol_wad_data[4]], yerr=[ol_wad_data[1], ol_wad_data[1]], fmt='--o', color='red', linestyle='none', label='ol')
plt.errorbar( ol_wad_data[5], ol_wad_data[0], xerr=[ol_wad_data[6], ol_wad_data[6]], yerr=[ol_wad_data[1], ol_wad_data[1]], fmt='--o', color='green', linestyle='none', label='wad')
plt.errorbar( ol_rw_data[3], ol_rw_data[0], xerr=[ol_rw_data[4], ol_rw_data[4]], yerr=[ol_rw_data[1], ol_rw_data[1]], fmt='--o', color='red', linestyle='none', label='ol')
plt.errorbar( ol_rw_data[5], ol_rw_data[0], xerr=[ol_rw_data[6], ol_rw_data[6]], yerr=[ol_rw_data[1], ol_rw_data[1]], fmt='--o', color='blue', linestyle='none', label='rw')
plt.errorbar( wad_rw_data[3], wad_rw_data[0], xerr=[wad_rw_data[4], wad_rw_data[4]], yerr=[wad_rw_data[1], wad_rw_data[1]], fmt='--o', color='green', linestyle='none', label='wad')
plt.errorbar( wad_rw_data[5], wad_rw_data[0], xerr=[wad_rw_data[6], wad_rw_data[6]], yerr=[wad_rw_data[1], wad_rw_data[1]], fmt='--o', color='blue', linestyle='none', label='rw')


plt.title('Mg2SiO4-Fe2SiO4 phase diagram')
plt.xlabel("X_Fe")
plt.ylabel("Pressure (GPa)")
plt.legend(loc='center right')
plt.show()

