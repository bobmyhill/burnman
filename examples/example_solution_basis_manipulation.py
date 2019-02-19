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
from burnman.solutionbases import transform_solution_to_new_basis, feasible_solution_in_component_space, feasible_endmember_occupancies_from_charge_balance, independent_endmember_occupancies_from_charge_balance, dependent_endmember_site_occupancies, dependent_endmember_sums, site_occupancies_to_strings, generate_complete_basis
from burnman.processanalyses import compute_and_set_phase_compositions, assemblage_affinity_misfit


# bridgmanite in FMASO system: Mg Fe Fe3+ Al3+ | Fe3+ Al3+ Si
bridgmanite_array = feasible_endmember_occupancies_from_charge_balance([[2,2,3,3], [3,3,4]], 6, as_fractions=False)
print(bridgmanite_array)

# Two site
simple_majorite_array = feasible_endmember_occupancies_from_charge_balance([[2, 3, 4], [2, 3, 4]], 6, as_fractions=False)
print(simple_majorite_array)

# Two site
simple_majorite_array = independent_endmember_occupancies_from_charge_balance([[2, 3, 4], [2, 3, 4]], 6, as_fractions=False)
print(simple_majorite_array)

# Four site majorite, independent endmembers
complex_majorite_array = independent_endmember_occupancies_from_charge_balance([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], 6, as_fractions=False)
print(complex_majorite_array)


# NCFMASO garnet
# Multiplicity 3     | Multiplicity 2     | total charge
# Ca, Fe2+, Mg2+, Na | Mg2+, Fe2+ Fe3+, Al3+, Si4+   | 12
# SWAPPED SITES SO THAT NA ONLY IN ONE ENDMEMBER
NCFMASO_majorite_array = independent_endmember_occupancies_from_charge_balance([[4, 4, 6, 6, 8], [6, 6, 6, 3]], 12, as_fractions=False)
print(len(NCFMASO_majorite_array))
print(NCFMASO_majorite_array)


# NCFMASO garnet
# Multiplicity 3     | Multiplicity 1     | Multiplicity 1     | total charge
# Na, Ca, Fe2+, Mg2+ | Mg2+, Fe2+ Fe3+, Al3+, Si4+   | Mg2+, Fe2+ Fe3+, Al3+, Si4+   | 12
NCFMASO_majorite_array = independent_endmember_occupancies_from_charge_balance([[3, 6, 6, 6], [2, 2, 3, 3, 4], [2, 2, 3, 3, 4]], 12, as_fractions=False)
print(len(NCFMASO_majorite_array))


# Spinel from Holland et al., 2018
# Mg Fe Fe3 Al |  Mg, Fe, Fe3, Al, Cr, Ti
spinel_array = independent_endmember_occupancies_from_charge_balance([[2, 2, 3, 3], [4, 4, 6, 6, 6, 8]], 8, as_fractions=False)
print(spinel_array)


# Opx from Holland et al., 2018
#       M1                       M2               T
#       Mg  Fe  Al  Fe3 Cr  Ti | Mg  Fe  Ca  Na | Si  Al
opx_array = independent_endmember_occupancies_from_charge_balance([[2, 2, 3, 3, 3, 4], [2, 2, 2, 1], [4, 3]], 8, as_fractions=False)
print(opx_array)

# Camph from Diener and Powell, 2012
# A*1    M13*3   M2*2            M4*2          T1*4
# v Na | Mg Fe | Mg Fe Al Fe3  | Ca Mg Fe Na | Si Al
camph_array = independent_endmember_occupancies_from_charge_balance([[0, 1],
                                                                     [6, 6],
                                                                     [4, 4, 6, 6],
                                                                     [4, 4, 4, 2],
                                                                     [16, 12]], 30., as_fractions=False)
print(camph_array)
n_published_members = 9
n_mbrs = len(camph_array)
n_missing_members = n_mbrs - n_published_members


camph_partial_basis = np.array([[1,0,1,0,1,  0,0,  0,1,0,0,0,  1,  0  ], # tr
                                [1,0,1,0,0,  0,1,  0,1,0,0,0,  0.5,0.5], # ts
                                [0,1,1,0,0.5,0,0.5,0,1,0,0,0,  0.5,0.5], # parg
                                [1,0,1,0,0,  0,1,  0,0,0,0,1,  1,  0  ], # gl
                                [1,0,1,0,1,  0,0,  0,0,1,0,0,  1,  0  ], # cumm
                                [1,0,0,1,0,  1,0,  0,0,0,1,0,  1,  0  ], # grun
                                [1,0,1,0,0,  1,0,  0,0,0,1,0,  1,  0  ], # a
                                [1,0,0,1,1,  0,0,  0,0,0,1,0,  1,  0  ], # b
                                [1,0,1,0,0,  0,0,  1,0,0,0,1,  1,  0  ]]) # mrb


print('{0} endmembers: note that the published model has {1} endmembers'.format(n_mbrs, n_published_members))

# Camph from Green et al., 2016; Holland et al., 2018
# A*1      M13*3   M2*2               M4*2          T1*4    V*2
# v Na K | Mg Fe | Mg Fe Al Fe3 Ti  | Ca Mg Fe Na | Si Al | OH O
camph_array = independent_endmember_occupancies_from_charge_balance([[0, 1, 1],
                                                                     [6, 6],
                                                                     [4, 4, 6, 6, 8],
                                                                     [4, 4, 4, 2],
                                                                     [16, 12],
                                                                     [-2, -4]], 28., as_fractions=False)
print(camph_array)
n_published_members = 11
n_mbrs = len(camph_array)
n_missing_members = n_mbrs - n_published_members

camph_partial_basis = np.array([[1,0,0,1,0,1,0,0,0,0,1,0,0,0,1,0,1,0], # tr
                                [1,0,0,1,0,0,0,1,0,0,1,0,0,0,0.5,0.5,1,0], # ts
                                [0,1,0,1,0,0.5,0,0.5,0,0,1,0,0,0,0.5,0.5,1,0], # parg
                                [1,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,1,0], # gl
                                [1,0,0,1,0,1,0,0,0,0,0,1,0,0,1,0,1,0], # cumm
                                [1,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0], # grun
                                [1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,1,0], # a
                                [1,0,0,0,1,1,0,0,0,0,0,0,1,0,1,0,1,0], # b
                                [1,0,0,1,0,0,0,0,1,0,0,0,0,1,1,0,1,0], # mrb
                                [0,0,1,1,0,0.5,0,0.5,0,0,1,0,0,0,0.5,0.5,1,0], # kprg
                                [1,0,0,1,0,0,0,0,0,1,1,0,0,0,0.5,0.5,0,1]]) # tts



print('{0} endmembers: note that the published model has {1} endmembers'.format(n_mbrs, n_published_members))
print('The following vector(s) complete the basis:')
print(generate_complete_basis(camph_partial_basis, camph_array)[-n_missing_members:])

# majorite from Holland et al. (2013)  Mg Fe Ca Na | Mg Fe Al Si
majorite_array = independent_endmember_occupancies_from_charge_balance([[6, 6, 6, 3], [4, 4, 6, 8]], 12, as_fractions=False)
print(majorite_array)

# Clinopyroxene from Jennings and Holland
cpx = JH_2015.clinopyroxene()
endmember_sums = dependent_endmember_sums(cpx.solution_model,
                                          as_fractions=False)
print(endmember_sums)
site_occupancies = dependent_endmember_site_occupancies(cpx.solution_model,
                                                        as_fractions=False)
print(site_occupancies_to_strings(cpx.solution_model.sites,
                                  cpx.solution_model.site_multiplicities,
                                  site_occupancies))


cpx.set_state(1.e5, 1000.)
cpx.set_composition([0.5, 0.5, 0.5, 0, 0, 0, 0, -0.5])
print(cpx.gibbs)
new_cpx = transform_solution_to_new_basis(cpx, [[1, 1, 0, 0, 0, 0, 0, -1],
                                                [0, 0, 1, 0, 0, 0, 0, 0]],
                                          solution_name='fdi-cats')

new_cpx.set_state(1.e5, 1000.)
new_cpx.set_composition([0.5, 0.5])
print(new_cpx.gibbs)


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

