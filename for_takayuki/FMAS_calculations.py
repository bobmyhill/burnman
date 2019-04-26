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
plt.style.use('ggplot')

import burnman
from burnman.minerals import HP_2011_ds62, SLB_2011, JH_2015
from burnman import equilibrate

ordering = False # This example plots the state of order of the Jennings and Holland orthopyroxene in the simple en-fs binary at 1 bar.
aluminosilicates = False # The example plots the andalusite-sillimanite-kyanite phase diagram
gt_solvus = False # This example demonstrates the shape of the pyrope-grossular solvus
lower_mantle = True # This example calculates temperatures and assemblage properties along an isentrope in the lower mantle
upper_mantle = False # This example produces a 2D grid of the ol-opx-gt field
olivine_polymorphs = True # This example produces a P-T pseudosection for a fo90 composition 

gt = SLB_2011.garnet()
ol = SLB_2011.mg_fe_olivine()
wad = SLB_2011.mg_fe_wadsleyite()
rw = SLB_2011.mg_fe_ringwoodite()
bdg = SLB_2011.mg_fe_bridgmanite()
ppv = SLB_2011.post_perovskite()
per = SLB_2011.ferropericlase()
opx = SLB_2011.orthopyroxene()
stv = SLB_2011.stishovite()
coe = SLB_2011.coesite()
cpv = SLB_2011.ca_perovskite()

ol.guess = np.array([0.93, 0.07])
wad.guess = np.array([0.91, 0.09]) # 0.91 0.09 works for olivine polymorphs...
rw.guess = np.array([0.93, 0.07])
opx.guess = np.array([0.68, 0.08, 0.15, 0.09])
gt.guess = np.array([0.42, 0.12, 0.46, 0.0, 0.00])
bdg.guess = np.array([0.86, 0.1, 0.04]) # 
ppv.guess = np.array([0.86, 0.1, 0.01]) # bdg-in works if guess[2] = 0.
per.guess = np.array([0.9, 0.1])


from burnman import CombinedMineral, SolidSolution
class fmaj(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'ferromajorite',
                                 mineral_list = [SLB_2011.pyrope(), SLB_2011.almandine(), SLB_2011.mg_majorite()],
                                 molar_amounts = [-1., 1., 1.],
                                 free_energy_adjustment=[-21.20278e3, 0., 0.])
        
class maj_garnet(SolidSolution):

    def __init__(self, molar_fractions=None):
        self.name = 'FMS majoritic garnet'
        self.solution_type = 'symmetric'
        self.endmembers = [[SLB_2011.pyrope(), '[Mg]3[Al][Al]Si3O12'],
                           [SLB_2011.mg_majorite(), '[Mg]3[Mg][Si]Si3O12'],
                           [fmaj(), '[Fe]3[Mg][Si]Si3O12']]
        self.energy_interaction = [[21.20278e3, -0.e3+2.*0.e3+2.*21.20278e3],
                                   [0.e3]]

        SolidSolution.__init__(self, molar_fractions=molar_fractions)


maj = maj_garnet()
maj.guess = np.array([0.9, 0.01, 0.09])

gt.guess = np.array([-0.1, 0.1, 0.0, 1.0, 0.00])
bdg.guess = np.array([0.86, 0.14, 0.0]) # 

P0 = 23.e9
T0 = 1600.

# ringwoodite bulk composition
composition = {'Fe': 0.2, 'Mg': 1.8, 'Si': 1.1, 'O':(2.0 + 1.1*2.)}
assemblage = burnman.Composite([rw, maj, bdg])
equality_constraints = [('T', T0), ('phase_proportion', (bdg, np.array([0.])))]
sol, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
if sol.success:
    print(assemblage)
    plt.scatter([0.], [assemblage.pressure])
    
assemblage = burnman.Composite([rw, maj, bdg])
equality_constraints = [('T', T0), ('phase_proportion', (maj, np.array([0.])))]
sol, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
if sol.success:
    print(assemblage)
    plt.scatter([0.], [assemblage.pressure])
    
assemblage = burnman.Composite([rw, bdg, per])
equality_constraints = [('T', T0), ('phase_proportion', (per, np.array([0.])))]
sol, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
if sol.success:
    print(assemblage)
    plt.scatter([0.], [assemblage.pressure])
    
assemblage = burnman.Composite([rw, bdg, per])
equality_constraints = [('T', T0), ('phase_proportion', (rw, np.array([0.])))]
sol, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
if sol.success:
    print(assemblage)
    plt.scatter([0.], [assemblage.pressure])
exit()

# Additional silica and aluminium
for xAl in np.linspace(0.02, 0.1, 21):
    xSi = 1.1
    composition = {'Fe': 0.2, 'Mg': 1.8, 'Al': xAl, 'Si': xSi, 'O':(2.0 + xSi*2. + xAl*1.5)}
    assemblage = burnman.Composite([rw, maj, bdg])
    assemblage.set_state(25.e9, T0)
    equality_constraints = [('T', T0), ('phase_proportion', (rw, np.array([0.])))]
    sol, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    if sol.success:
        print(assemblage)
        
        plt.scatter([xAl], [assemblage.pressure])

plt.show()
        
# Additional silica and aluminium
for xAl in np.linspace(0.02, 0.2, 21):
    xSi = 1.1
    composition = {'Fe': 0.2, 'Mg': 1.8, 'Al': xAl, 'Si': xSi, 'O':(2.0 + xSi*2. + xAl*1.5)}
    assemblage = burnman.Composite([rw, maj, bdg, per])
    assemblage.set_state(25.e9, T0)
    equality_constraints = [('T', T0), ('phase_proportion', (per, np.array([0.])))]
    sol, prm = equilibrate(composition, assemblage, equality_constraints, initial_state_from_assemblage=True, store_iterates=False)
    if sol.success:
        print(assemblage)
        
        plt.scatter([xAl], [assemblage.pressure])
        
    composition = {'Fe': 0.2, 'Mg': 1.8, 'Al': xAl, 'Si': 1.5, 'O':(2.0 + 1.5*2. + xAl*1.5)}
    assemblage = burnman.Composite([rw, maj, bdg, per])
    assemblage.set_state(25.e9, T0)
    equality_constraints = [('T', T0), ('phase_proportion', (rw, np.array([0.])))]
    sol, prm = equilibrate(composition, assemblage, equality_constraints, initial_state_from_assemblage=True, store_iterates=False)
    if sol.success:
        print(assemblage)

    
        plt.scatter([xAl], [assemblage.pressure])
        
    equality_constraints = [('T', T0), ('phase_proportion', (per, np.array([0.])))]
    sol, prm = equilibrate(composition, assemblage, equality_constraints, initial_state_from_assemblage=True, initial_composition_from_assemblage=True, store_iterates=False)
    
    if sol.success:
        print(assemblage)
    
        plt.scatter([xAl], [assemblage.pressure])

    f = np.array(assemblage.molar_fractions)[0:3]
    P = assemblage.pressure
    T = assemblage.temperature
    assemblage = burnman.Composite([rw, maj, bdg])
    assemblage.set_fractions(f)
    assemblage.set_state(P, T)
    assemblage.n_moles = assemblage.n_moles
    equality_constraints = [('T', T0), ('phase_proportion', (bdg, np.array([0.])))]
    sol, prm = equilibrate(composition, assemblage, equality_constraints, initial_state_from_assemblage=True, initial_composition_from_assemblage=True, store_iterates=False)
    if sol.success:
        print(assemblage)
    
        plt.scatter([xAl], [assemblage.pressure])

    
    assemblage = burnman.Composite([maj, bdg, per])
    assemblage.set_state(25.e9, T0)
    equality_constraints = [('T', T0), ('phase_proportion', (maj, np.array([0.])))]
    sol, prm = equilibrate(composition, assemblage, equality_constraints, initial_state_from_assemblage=True, store_iterates=False)
    if sol.success:
        print(assemblage)
    
        plt.scatter([xAl], [assemblage.pressure])

plt.show()

    

'''
    S = np.array([assemblage.molar_entropy*assemblage.n_moles])
    
    
    assemblage = burnman.Composite([bdg, per, ppv, cpv])
    equality_constraints = [('S', S), ('phase_proportion', (bdg, np.array([0.])))]
    sol, prm = equilibrate(composition, assemblage, equality_constraints,
                           store_iterates=False,
                           initial_state = sol.x[0:2])
    P_bdg_in = assemblage.pressure
    
    
    equality_constraints = [('S', S), ('phase_proportion', (ppv, np.array([0.])))]
    sol, prm = equilibrate(composition, assemblage, equality_constraints,
                           store_iterates=False,
                           initial_state = sol.x[0:2])
    P_ppv_in = assemblage.pressure
    T_ppv_in = assemblage.temperature

    pressures = np.linspace(P_ppv_in, P_bdg_in, 21)
    equality_constraints = [('P', pressures), ('S', S)]
    sols1, prm1 = equilibrate(composition, assemblage, equality_constraints,
                              initial_state = [P_ppv_in, T_ppv_in],
                              initial_composition_from_assemblage = True,
                              store_iterates=False)
    p1 = np.array([sol.x for sol in sols1 if sol.success]).T

'''
