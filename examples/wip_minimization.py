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

'''
sillimanite = HP_2011_ds62.sill()
andalusite = HP_2011_ds62.andalusite()
kyanite = HP_2011_ds62.ky()

composition = sillimanite.formula
assemblage = burnman.Composite([sillimanite, andalusite, kyanite])
equality_constraints = [('phase_proportion', (kyanite, np.array([0.0]))), ('phase_proportion', (sillimanite, np.array([0.0])))]
sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
P, T = sols[0][0].x[0:2]
print(P/1e9, T)

temperatures = np.linspace(500., 1500., 21)
assemblage = burnman.Composite([sillimanite, andalusite])
equality_constraints = [('T', temperatures), ('phase_proportion', (sillimanite, np.array([0.0])))]
sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
Ps = np.array([sol[0].x[0] for sol in sols if sol[0].success])
Ts = np.array([sol[0].x[1] for sol in sols if sol[0].success])
plt.plot(Ts, Ps/1.e9)

assemblage = burnman.Composite([sillimanite, kyanite])
equality_constraints = [('T', temperatures), ('phase_proportion', (sillimanite, np.array([0.0])))]
sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
Ps = np.array([sol[0].x[0] for sol in sols if sol[0].success])
Ts = np.array([sol[0].x[1] for sol in sols if sol[0].success])
plt.plot(Ts, Ps/1.e9)

assemblage = burnman.Composite([andalusite, kyanite])
equality_constraints = [('T', temperatures), ('phase_proportion', (andalusite, np.array([0.0])))]
sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
Ps = np.array([sol[0].x[0] for sol in sols if sol[0].success])
Ts = np.array([sol[0].x[1] for sol in sols if sol[0].success])
plt.plot(Ts, Ps/1.e9)

plt.show()
'''


gt1 = SLB_2011.garnet()
gt2 = SLB_2011.garnet()

gt1.guess = np.array([0.99, 0.00, 0.01, 0.0, 0.00])
gt2.guess = np.array([0.01, 0.00, 0.99, 0.0, 0.00])


temperatures = np.linspace(200., 600., 21)
pressures = np.array([1.e5])

composition = {'Mg': 1.5, 'Ca': 1.5, 'Al': 2., 'Si': 3.0, 'O':12.0}

'''
assemblage = burnman.Composite([gt1, gt2])
equality_constraints = [('P', pressures), ('phase_proportion', (gt2, np.array([0.0])))]
sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
print(sols[0][0])
x1s = np.array([sol.x[3] for sol in sols[0] if sol.success])
x2s = np.array([sol.x[6] for sol in sols[0] if sol.success])
Ts = np.array([sol.x[1] for sol in sols[0] if sol.success])
plt.plot(x1s, Ts)
plt.plot(x2s, Ts)
plt.show()
'''

assemblage = burnman.Composite([gt1, gt2])
equality_constraints = [('P', pressures), ('T', temperatures)]
sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
x1s = np.array([sol.x[3] for sol in sols[0] if sol.success])
x2s = np.array([sol.x[6] for sol in sols[0] if sol.success])
Ts = np.array([sol.x[1] for sol in sols[0] if sol.success])
plt.plot(x1s, Ts)
plt.plot(x2s, Ts)
plt.show()


    
gt = SLB_2011.garnet()
ol = SLB_2011.mg_fe_olivine()
wad = SLB_2011.mg_fe_wadsleyite()
rw = SLB_2011.mg_fe_ringwoodite()
bdg = SLB_2011.mg_fe_bridgmanite()
per = SLB_2011.ferropericlase()
opx = SLB_2011.orthopyroxene()
stv = SLB_2011.stishovite()
coe = SLB_2011.coesite()

ol.guess = np.array([0.93, 0.07])
wad.guess = np.array([0.91, 0.09])
rw.guess = np.array([0.93, 0.07])
opx.guess = np.array([0.68, 0.08, 0.15, 0.09])
gt.guess = np.array([0.42, 0.12, 0.46, 0.0, 0.00])
bdg.guess = np.array([0.9, 0.1, 0.0])
per.guess = np.array([0.9, 0.1])


P = 2.e9
T = 1200.


# Better compositional choice? weighted nnls
#Ax = b, where A = stoichiometric matrix (converted into site-endmembers), and b = bulk composition (weight heavily)
#Ax = b, where A = compositional constraints (if any) and b = 0 (weight heavily)
#Ax = b, where 

#endmember proportions
#(A1 + A2) / (A1 + A2 + A3) = p_i
#i.e. A = 1-p, 1-p, -p , b=0



temperatures = np.linspace(800., 1500., 8)
pressures = np.linspace(1.e9, 14.e9, 11)

'''
composition = {'Fe': 0.2, 'Mg': 2.0, 'Si': 1.9, 'Ca': 0.2, 'Al': 0.4, 'O':6.8}
assemblage = burnman.Composite([ol, opx, gt])
#equality_constraints = [('phase_proportion', (opx, np.array([0.0]))), ('T', temperatures)]
equality_constraints = [('P', pressures), ('T', temperatures)]
sol, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
'''



composition = {'Fe': 0.2, 'Mg': 1.8, 'Si': 1.0, 'O':4.0}
assemblage = burnman.Composite([ol, wad])
equality_constraints = [('T', temperatures), ('phase_proportion', (wad, np.array([0.0])))]
sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
pressures = np.array([sol[0].x[0] for sol in sols])
plt.plot(temperatures, pressures/1.e9)


equality_constraints = [('T', temperatures), ('phase_proportion', (wad, np.array([1.0])))]
sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
pressures = np.array([sol[0].x[0] for sol in sols])
plt.plot(temperatures, pressures/1.e9)



assemblage = burnman.Composite([wad, rw])
equality_constraints = [('T', temperatures), ('phase_proportion', (rw, np.array([0.0])))]
sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
pressures = np.array([sol[0].x[0] for sol in sols])
plt.plot(temperatures, pressures/1.e9)

equality_constraints = [('T', temperatures), ('phase_proportion', (rw, np.array([1.0])))]
sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
pressures = np.array([sol[0].x[0] for sol in sols])
plt.plot(temperatures, pressures/1.e9)



assemblage = burnman.Composite([rw, bdg, per])
equality_constraints = [('T', temperatures), ('phase_proportion', (rw, np.array([0.0])))]
sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
pressures = np.array([sol[0].x[0] for sol in sols])
plt.plot(temperatures, pressures/1.e9)

equality_constraints = [('T', temperatures), ('phase_proportion', (rw, np.array([1.0])))]
sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
pressures = np.array([sol[0].x[0] for sol in sols])
plt.plot(temperatures, pressures/1.e9)

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


