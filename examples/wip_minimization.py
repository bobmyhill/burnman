# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.

'''
example_gibbs_minimization
--------------------

This example demonstrates how burnman may be used to calculate the
equilibrium phase proportions and compositions for an assemblage
of a fixed bulk composition.

*Uses:*

* :doc:`mineral_database`
* :class:`burnman.composite.Composite`
* :func:`burnman.equilibriumassemblage.gibbs_minimizer`
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
from burnman.minerals import HP_2011_ds62, SLB_2011
from burnman import equilibrate

aluminosilicates = False
gt_solvus = False
lower_mantle = True
upper_mantle = False
olivine_polymorphs = True

if aluminosilicates:
    sillimanite = HP_2011_ds62.sill()
    andalusite = HP_2011_ds62.andalusite()
    kyanite = HP_2011_ds62.ky()
    
    composition = sillimanite.formula
    assemblage = burnman.Composite([sillimanite, andalusite, kyanite])
    equality_constraints = [('phase_proportion', (kyanite, np.array([0.0]))), ('phase_proportion', (sillimanite, np.array([0.0])))]
    sol, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    P, T = sol.x[0:2]
    print('invariant point found at {0} GPa, {1} K'.format(P/1e9, T))
    
    temperatures = np.linspace(500., 1500., 21)
    assemblage = burnman.Composite([sillimanite, andalusite])
    equality_constraints = [('T', temperatures), ('phase_proportion', (sillimanite, np.array([0.0])))]
    sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    Ps = np.array([sol.x[0] for sol in sols if sol.success])
    Ts = np.array([sol.x[1] for sol in sols if sol.success])
    plt.plot(Ts, Ps/1.e9)
    
    assemblage = burnman.Composite([sillimanite, kyanite])
    equality_constraints = [('T', temperatures), ('phase_proportion', (sillimanite, np.array([0.0])))]
    sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    Ps = np.array([sol.x[0] for sol in sols if sol.success])
    Ts = np.array([sol.x[1] for sol in sols if sol.success])
    plt.plot(Ts, Ps/1.e9)
    
    assemblage = burnman.Composite([andalusite, kyanite])
    equality_constraints = [('T', temperatures), ('phase_proportion', (andalusite, np.array([0.0])))]
    sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    Ps = np.array([sol.x[0] for sol in sols if sol.success])
    Ts = np.array([sol.x[1] for sol in sols if sol.success])
    plt.plot(Ts, Ps/1.e9)
    
    plt.show()

if gt_solvus:
    gt1 = SLB_2011.garnet()
    gt2 = SLB_2011.garnet()
    
    gt1.guess = np.array([0.99, 0.00, 0.01, 0.0, 0.00])
    gt2.guess = np.array([0.01, 0.00, 0.99, 0.0, 0.00])
    
    
    temperatures = np.linspace(200., 600., 21)
    pressure = 1.e5
    
    composition = {'Mg': 1.5, 'Ca': 1.5, 'Al': 2., 'Si': 3.0, 'O':12.0}
    
    
    
    assemblage = burnman.Composite([gt1, gt2])
    equality_constraints = [('P', pressure), ('T', temperatures)]
    sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    x1s = np.array([sol.x[3] for sol in sols if sol.success])
    x2s = np.array([sol.x[6] for sol in sols if sol.success])
    Ts = np.array([sol.x[1] for sol in sols if sol.success])
    plt.plot(x1s, Ts)
    plt.plot(x2s, Ts)
    plt.show()
    


    
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
wad.guess = np.array([0.91, 0.09])
rw.guess = np.array([0.93, 0.07])
opx.guess = np.array([0.68, 0.08, 0.15, 0.09])
gt.guess = np.array([0.42, 0.12, 0.46, 0.0, 0.00])
bdg.guess = np.array([0.86, 0.1, 0.04]) # 
ppv.guess = np.array([0.86, 0.1, 0.01]) # bdg-in works if guess[2] = 0.
per.guess = np.array([0.9, 0.1])

if lower_mantle:
    P0 = 25.e9
    T0 = 1600.
    
    composition = {'Fe': 0.2, 'Mg': 2.0, 'Si': 1.9, 'Ca': 0.2, 'Al': 0.4, 'O':6.8}
    assemblage = burnman.Composite([bdg, per, cpv])
    equality_constraints = [('P', P0), ('T', T0)]
    sol, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    
    S = np.array([assemblage.molar_entropy*assemblage.n_moles])
    
    
    assemblage = burnman.Composite([bdg, per, ppv, cpv])
    equality_constraints = [('S', S), ('phase_proportion', (bdg, np.array([0.])))]
    sol, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    bdg_in = assemblage.pressure
    
    
    equality_constraints = [('S', S), ('phase_proportion', (ppv, np.array([0.])))]
    sol, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    ppv_in = assemblage.pressure
    
    
    pressures = np.linspace(ppv_in, bdg_in, 21)
    equality_constraints = [('P', pressures), ('S', S)]
    sols1, prm1 = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    
    p1 = np.array([sol.x for sol in sols1 if sol.success]).T
    
    assemblage = burnman.Composite([bdg, per, cpv])
    pressures = np.linspace(25.e9, ppv_in, 21)
    equality_constraints = [('P', pressures), ('S', S)]
    sols2, prm2 = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    
    p2 = np.array([sol.x for sol in sols2 if sol.success]).T
    
    assemblage = burnman.Composite([ppv, per, cpv])
    pressures = np.linspace(bdg_in, 140.e9, 21)
    equality_constraints = [('P', pressures), ('S', S)]
    sols3, prm3 = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    
    p3 = np.array([sol.x for sol in sols3 if sol.success]).T
    
    
    
    fig = plt.figure()
    ax = [fig.add_subplot(2, 3, i) for i in range(1, 7)]
    phases = [bdg, per, cpv, ppv]
    colors = ['red', 'green', 'blue', 'orange']
    for p, prm in [(p1, prm1), (p2, prm2), (p3, prm3)]:
        
        pressures, temperatures = p[[0, 1],:]
        ax[0].plot(pressures/1.e9, temperatures, color='black')
        
        for i, phase in enumerate(phases):
            try:
                idx = prm.parameter_names.index('x({0})'.format(phase.name))
                x_phase = p[idx,:]
                ax[i+1].plot(pressures/1.e9, x_phase, color=colors[i])
                ax[i+1].set_ylim(0,2)
                ax[i+1].set_xlim(0,140)
            except:
                pass

    plt.show()

if upper_mantle:
    temperatures = np.linspace(800., 1500., 8)
    pressures = np.linspace(1.e9, 14.e9, 11)
    
    composition = {'Fe': 0.2, 'Mg': 2.0, 'Si': 1.9, 'Ca': 0.2, 'Al': 0.4, 'O':6.8}
    assemblage = burnman.Composite([ol, opx, gt])
    #equality_constraints = [('phase_proportion', (opx, np.array([0.0]))), ('T', temperatures)]
    equality_constraints = [('P', pressures), ('T', temperatures)]
    sol, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    

if olivine_polymorphs:
    temperatures = np.linspace(573., 1773., 8)
    pressures = np.linspace(1.e9, 14.e9, 11)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    
    composition = {'Fe': 0.2, 'Mg': 1.8, 'Si': 1.0, 'O':4.0}
    assemblage = burnman.Composite([ol, wad])
    equality_constraints = [('T', temperatures), ('phase_proportion', (wad, np.array([0.0])))]
    sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    Ps = np.array([sol.x[0] for sol in sols if sol.success])
    Ts = np.array([sol.x[1] for sol in sols if sol.success])
    ax.plot(Ts, Ps/1.e9)
    
    equality_constraints = [('T', temperatures), ('phase_proportion', (wad, np.array([1.0])))]
    sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    Ps = np.array([sol.x[0] for sol in sols if sol.success])
    Ts = np.array([sol.x[1] for sol in sols if sol.success])
    ax.plot(Ts, Ps/1.e9)
    
    
    
    assemblage = burnman.Composite([wad, rw])
    equality_constraints = [('T', temperatures), ('phase_proportion', (rw, np.array([0.0])))]
    sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    Ps = np.array([sol.x[0] for sol in sols if sol.success])
    Ts = np.array([sol.x[1] for sol in sols if sol.success])
    ax.plot(Ts, Ps/1.e9)
    
    equality_constraints = [('T', temperatures), ('phase_proportion', (rw, np.array([1.0])))]
    sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    Ps = np.array([sol.x[0] for sol in sols if sol.success])
    Ts = np.array([sol.x[1] for sol in sols if sol.success])
    ax.plot(Ts, Ps/1.e9)
    
    
    
    assemblage = burnman.Composite([rw, bdg, per])
    equality_constraints = [('T', temperatures), ('phase_proportion', (rw, np.array([0.0])))]
    sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    Ps = np.array([sol.x[0] for sol in sols if sol.success])
    Ts = np.array([sol.x[1] for sol in sols if sol.success])
    ax.plot(Ts, Ps/1.e9)
    
    equality_constraints = [('T', temperatures), ('phase_proportion', (rw, np.array([1.0])))]
    sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    Ps = np.array([sol.x[0] for sol in sols if sol.success])
    Ts = np.array([sol.x[1] for sol in sols if sol.success])
    ax.plot(Ts, Ps/1.e9)
    
    plt.show()


'''
S =  assemblage.molar_entropy*assemblage.n_moles
#assemblage.set_state(P, T)
parameters = get_parameters(assemblage, indices)

PTX = []
pressures = np.linspace(13.81994e9, 13.81998e9, 15)
for P in pressures:
    equality_constraints = [('P', P), ('S', S)]
    sol = damped_newton_solve(F = lambda x: F(x, assemblage, equality_constraints),
                              J = lambda x: jacobian(x, assemblage, equality_constraints),
                              lambda_bounds = lambda dx, x: lambda_bounds(dx, x, indices),
                              guess = parameters,
                              constraints = lambda x: C(x, constraint_matrix, constraint_vector),
                              tol=1.e-3) # no need for a particularly high tolerance
    #sol = fsolve(F, x0 = parameters, args = (assemblage, P, 300.), full_output=True)
    #sol = root(F, x0 = parameters, args = (assemblage, P, 300.))
    #print('\n{0}'.format(sol[-1]))
    #if sol[-2] == 1:
    
    print(sol.text)
    if sol.success:
        print('\nFinal assemblage:')
        print(assemblage)

        PTX.append([sol.x[0], sol.x[1], assemblage.phases[assemblage.phases.index(gt)].molar_fractions])
        
        dx = np.zeros_like(sol.x)
        dx[0] = pressures[1] - pressures[0]
        parameters = sol.x + np.dot(dxidxj(sol.x, assemblage, equality_constraints), dx) # next guess based on the Jacobian J and change in parameters dx
        constraint_values = C(parameters, constraint_matrix, constraint_vector)
        if any(constraint_values > 0.):
            print(sol.x[0:2])
            print('The following phases might be exhausted before the next step')
            # Check to see if it's one of the phase amounts going to zero
            print([assemblage.phases[i].name for i, v in enumerate(parameters[proportion_indices]) if v<0.])
            print('Terminating')
            break
    else:
        print(C(sol.x, constraint_matrix, constraint_vector))


parameters = get_parameters(assemblage, indices)
equality_constraints = [('T', T), phase_proportion_constraint(opx, assemblage, indices, 0.0)]
sol = damped_newton_solve(F = lambda x: F(x, assemblage, equality_constraints),
                          J = lambda x: jacobian(x, assemblage, equality_constraints),
                          lambda_bounds = lambda dx, x: lambda_bounds(dx, x, indices),
                          guess = parameters,
                          constraints = lambda x: C(x, constraint_matrix, constraint_vector),
                          tol=1.e-3)
if sol.success:
    print(sol.text)
    print('\nFinal assemblage:')
    print(assemblage)
else:
    print(C(sol.x, constraint_matrix, constraint_vector))


assemblage.set_state(1.e9, 1000.)
parameters = get_parameters(assemblage, indices)
equality_constraints = [('T', T), phase_proportion_constraint(opx, assemblage, indices, 0.0)]
sol = damped_newton_solve(F = lambda x: F(x, assemblage, equality_constraints),
                          J = lambda x: jacobian(x, assemblage, equality_constraints),
                          lambda_bounds = lambda dx, x: lambda_bounds(dx, x, indices),
                          guess = parameters,
                          constraints = lambda x: C(x, constraint_matrix, constraint_vector),
                          tol=1.e-3)
if sol.success:
    print(sol.text)
    print('\nFinal assemblage:')
    print(assemblage)
else:
    print(C(sol.x, constraint_matrix, constraint_vector))
'''


