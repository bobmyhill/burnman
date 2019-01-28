# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""

"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))


import burnman
from burnman.minerals import JH_2015
from burnman.solutionbases import transform_solution_to_new_basis, feasible_solution_in_component_space, feasible_site_occupancies_from_charge_balance, dependent_endmember_site_occupancies, dependent_endmember_sums, site_occupancies_to_strings, independent_row_indices
from burnman.processanalyses import compute_and_set_phase_compositions, assemblage_affinity_misfit


# bridgmanite in FMASO system: Mg Fe Fe3+ Al3+ | Fe3+ Al3+ Si
bridgmanite_array = feasible_site_occupancies_from_charge_balance([[2,2,3,3], [3,3,4]], 6, as_fractions=False)
print(bridgmanite_array)

# majorite from Holland et al. (2013)  Mg Fe Ca Na | Mg Fe Al Si
majorite_array = feasible_site_occupancies_from_charge_balance([[6, 6, 6, 3], [4, 4, 6, 8]], 12, as_fractions=False)
print(majorite_array[independent_row_indices(majorite_array)])

# Clinopyroxene from Jennings and Holland
endmember_sums = dependent_endmember_sums(JH_2015.clinopyroxene().solution_model,
                                          as_fractions=False)
print(endmember_sums)

site_occupancies = dependent_endmember_site_occupancies(JH_2015.clinopyroxene().solution_model,
                                                        as_fractions=False)
print(site_occupancies_to_strings(JH_2015.clinopyroxene().solution_model,
                                  site_occupancies))


gt = JH_2015.garnet()
new_gt = transform_solution_to_new_basis(gt, [[0, 1, -1, 1, 0]],
                                         endmember_names = ['skiagite'],
                                         solution_name='andr-sk')

gt = JH_2015.garnet()
new_gt = transform_solution_to_new_basis(gt, [[0, 0, 0, 1, 0],
                                              [0, 1, -1, 1, 0]],
                                         endmember_names = ['andradite', 'skiagite'],
                                         solution_name='andr-sk')

cpx = JH_2015.clinopyroxene()
new_cpx = transform_solution_to_new_basis(cpx, [[1, 1, 0, 0, 0, 0, 0, -1],
                                                [0, 0, 1, 0, 0, 0, 0, 0]],
                                          solution_name='fdi-cats')


P = 1.e9
T = 1000.
cpx.set_composition([0.5, 0.5, 0.5, 0, 0, 0, 0, -0.5])
new_cpx.set_composition([0.5, 0.5])
cpx.set_state(P, T)
new_cpx.set_state(P, T)
print(burnman.processchemistry.formula_to_string(cpx.formula))
print(burnman.processchemistry.formula_to_string(new_cpx.formula))
print(cpx.gibbs, new_cpx.gibbs)
print(cpx.S, new_cpx.S)
print(cpx.V, new_cpx.V)


"""
# Check if starting composition is inside hull

def in_hull(points, x):
    # grabbed from https://stackoverflow.com/a/43564754
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]

    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

def single_solution(points, x):
    # grabbed from https://stackoverflow.com/a/43564754
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]

    sol = nnls(A, b)
    return sol


def feasibility_mask(points, x, vital_min=1.e-8):
    # somewhat faster than imposing bounds
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]

    
    A_ub = np.ones((1,n_points))
    b_ub = 1. - vital_min

    mask = []
    for i in range(n_points):
        A_ub[:,i] = 0
        lp = linprog(c, A_eq=A, b_eq=b, A_ub=A_ub, b_ub=b_ub)
        mask.append(lp.success)
        A_ub[:,i] = 1
    return mask

n_points = 100
n_dim = 5
Z = np.random.rand(n_points,n_dim)
x = np.random.rand(n_dim)
x = (Z[0] + Z[1] + Z[2])/3.
print(in_hull(Z, x))
print(single_solution(Z, x))

dx = np.sqrt(3.)
Z = np.array([[0.+dx, 1.],
              [0.+dx, -1.],
              [0.5+dx, 0.],
              [-0.5+dx, 0.]])
x = np.array([-0.250000000001+dx, 0.5])
print(in_hull(Z, x))
x = np.array([0.2500000000001+dx, 0.5])
print(in_hull(Z, x))
x = np.array([-0.25000000001+dx, 0.5])
print(in_hull(Z, x))
x = np.array([0.250000000001+dx, 0.5])
print(in_hull(Z, x))


x = np.array([-0.25+dx, 0.5])
print(feasibility_mask(Z, x))
"""

