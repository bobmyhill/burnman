# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.

'''
example_equilibrate
--------------------

This example demonstrates how burnman may be used to calculate the
equilibrium phase proportions and compositions for an assemblage
of a fixed bulk composition.

*Uses:*

* :doc:`mineral_database`
* :class:`burnman.composite.Composite`
* :func:`burnman.equilibrate.equilibrate`
'''
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt

import burnman
from burnman.minerals import SLB_2011
from burnman import equilibrate

"""
# Parameters from Takayuki
# mpv + wus -> fpv + per ... dG_bdg_fper = 24.1e3 + 3.5*T
dE_bdg_fper = 24.1e3
dS_bdg_fper = -3.5
dV_bdg_fper = 0.

# mrw/2 + wus -> frw/2 + per ... DG_rw_fper = 912 + 4.15*(T - 298) + 18.9*10*P # 
dE_rw_fper = 912. + (4.15 * -298.)
dS_rw_fper = -4.15
dV_rw_fper = 18.9*10*1.e-9 # in GPa 

W_bdg = 2.7e3
W_fper = 13.9e3
W_rw = 2.5e3 # on a one cation basis
"""

P_mrw_breakdown = 23.79e9
T_mrw_breakdown = 1700.


# note that the above parameters only provide
# two of the three/four required reaction energies -
# There are three components (MgO, FeO, SiO2) and
# six or seven endmembers (w/ and w/out stishovite).

per = SLB_2011.periclase() # MgO standard state
wus = SLB_2011.wuestite() # FeO standard state
mpv = SLB_2011.mg_bridgmanite() # SiO2 standard state


# 1) Create mrw (use reaction pressure at 1700 K)
mrw = SLB_2011.mg_ringwoodite()
for m in [mpv, mrw, per]:
    m.set_state(P_mrw_breakdown, T_mrw_breakdown)
print('Making mrw {0:.2f} J/mol more stable'.format(mrw.gibbs - (per.gibbs + mpv.gibbs)))
mrw.params['F_0'] -= mrw.gibbs - (per.gibbs + mpv.gibbs)

# 2) Create frw (frw = mrw + 2*wus - 2*per)
frw = SLB_2011.fe_ringwoodite()

# 3) Create fpv (fpv = mpv + wus - per)
fpv = SLB_2011.fe_bridgmanite()

# 4) Create stv (not in manuscript)
stv = SLB_2011.stishovite()

# Create the solution models
fper = burnman.SolidSolution(name = 'magnesiowustite/ferropericlase',
                             solution_type = 'symmetric',
                             endmembers = [[per, '[Mg]O'], [wus, '[Fe]O']],
                             energy_interaction = [[13.0e3]])

rw = burnman.SolidSolution(name = 'ringwoodite',
                             solution_type = 'symmetric',
                             endmembers = [[mrw, '[Mg]2SiO4'], [frw, '[Fe]2SiO4']],
                             energy_interaction = [[9.34e3]]) 

bdg = burnman.SolidSolution(name = 'bridgmanite',
                             solution_type = 'symmetric',
                             endmembers = [[mpv, '[Mg]SiO3'], [fpv, '[Fe]SiO3']],
                             energy_interaction = [[0.e3]])


x_Fe = 0.3
rw.guess = np.array([1. - x_Fe, x_Fe])
fper.guess = np.array([1. - x_Fe, x_Fe])
bdg.guess = np.array([1. - x_Fe, x_Fe])
composition = {'Fe': x_Fe*2., 'Mg': (1. - x_Fe)*2., 'Si': 1., 'O': 4.}
assemblage = burnman.Composite([rw, fper, bdg, stv])
assemblage.set_state(P_mrw_breakdown, T_mrw_breakdown)
equality_constraints = [('T', T_mrw_breakdown),
                        ('phase_proportion', (rw, np.array([0.])))]
sol, prm = equilibrate(composition, assemblage,
                       equality_constraints,
                       initial_state_from_assemblage = True,
                       store_iterates=False)

P_univariant = sol.x[0]
x_rw = sol.x[3]
x_fper = sol.x[5]
x_bdg = sol.x[7]
x_bdg_per = (sol.x[5] + sol.x[7])/2. # equal parts bdg and fper

plt.plot([x_rw, x_bdg_per, sol.x[5]],
         [P_univariant/1.e9, P_univariant/1.e9, P_univariant/1.e9], color='black')
print(x_rw, x_bdg_per)

# Create the phase diagram
x_Fes = np.linspace(0.0001, x_bdg_per, 21)

sols = []
cs = []
for x_Fe in x_Fes:
    rw.guess = np.array([1. - x_Fe, x_Fe])
    fper.guess = np.array([1. - x_Fe, x_Fe])
    bdg.guess = np.array([1. - x_Fe, x_Fe])
    composition = {'Fe': x_Fe*2., 'Mg': (1. - x_Fe)*2., 'Si': 1., 'O': 4.}
    assemblage = burnman.Composite([rw, fper, bdg])
    assemblage.set_state(P_mrw_breakdown, T_mrw_breakdown)
    equality_constraints = [('T', T_mrw_breakdown),
                        ('phase_proportion', (rw, np.array([0.])))]
    sol, prm = equilibrate(composition, assemblage,
                           equality_constraints,
                           initial_state_from_assemblage = True,
                           store_iterates=False)
    if sol.success:
        sols.append(sol.x)
        cs.append(x_Fe)

s = np.array(sols)
plt.plot(s[:,3], s[:,0]/1.e9, color='black')
plt.plot(cs, s[:,0]/1.e9, color='black')


# Also rw + fper + stv
x_Fe = 0.3
rw.guess = np.array([1. - x_rw, x_rw])
fper.guess = np.array([1. - x_fper, x_fper])
bdg.guess = np.array([1. - x_bdg, x_bdg])
composition = {'Fe': x_Fe*2., 'Mg': (1. - x_Fe)*2., 'Si': 1., 'O': 4.}
assemblage = burnman.Composite([rw, fper, stv])


pressures = np.linspace(P_univariant-1000., 23.6e9, 5)
equality_constraints = [('P', pressures),
                        ('T', T_mrw_breakdown)]
sols, prm = equilibrate(composition, assemblage,
                        equality_constraints,
                        initial_state_from_assemblage = True,
                        store_iterates=False)

s = np.array([sol.x for sol in sols if sol.success])
plt.plot(s[:,3], s[:,0]/1.e9, color='black')
plt.plot(s[:,5], s[:,0]/1.e9, color='black')  

assemblage = burnman.Composite([bdg, fper, stv])


pressures = np.linspace(P_univariant+1000., 24.1e9, 5)
equality_constraints = [('P', pressures),
                        ('T', T_mrw_breakdown)]
sols, prm = equilibrate(composition, assemblage,
                        equality_constraints,
                        initial_state_from_assemblage = True,
                        store_iterates=False)

s = np.array([sol.x for sol in sols if sol.success])
plt.plot((s[:,3]+s[:,5])/2., s[:,0]/1.e9, color='black')
plt.plot(s[:,5], s[:,0]/1.e9, color='black')  

plt.scatter([0.2237, 0.2372, 0.35], [23.93, 23.93, 23.93])

plt.xlim(0., )
plt.ylim(23.6, 24.1)
plt.xlabel('$x_{Fe}$')
plt.ylabel('P (GPa)')

plt.savefig('postspinel_binary_loop_SLB_mod.pdf')
plt.show()
