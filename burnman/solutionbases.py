# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2019 by the BurnMan team, released under the GNU
# GPL v2 or later.

from __future__ import absolute_import

import cdd
import numpy as np
from sympy import Matrix, nsimplify
from fractions import Fraction
from scipy.optimize import linprog
from .processchemistry import dictionarize_formula, compositional_array, formula_to_string, site_occupancies_to_strings
from . import CombinedMineral, SolidSolution

def feasible_site_occupancies_from_charge_balance(charges, charge_total, as_fractions=False):
    n_sites = len(charges)
    all_charges = np.concatenate(charges)
    n_site_elements = len(all_charges)
    equalities = np.empty((n_sites+1, n_site_elements+1))
    equalities[:-1,0] = -1
    i = 0
    for i_site, site_charges in enumerate(charges):
        equalities[i_site,1:] = [1 if (j>=i and j<i+len(site_charges)) else 0 for j in range(n_site_elements)]
        i+=len(site_charges)
        
    equalities[-1,0] = -charge_total
    equalities[-1,1:] = all_charges
    polytope = cdd.Matrix(equalities, linear=True, number_type='fraction')
    
    polytope.extend(np.concatenate((np.zeros((len(equalities[0])-1, 1)),
                                    np.identity(len(equalities[0])-1)),
                                   axis=1),
                    linear=False)
    P = cdd.Polyhedron(polytope)
    if as_fractions:
        V = np.array([map(Fraction, v) for v in P.get_generators()])
    else:
        V = np.array([map(float, v) for v in P.get_generators()])
    vertices = V[:,1:]/V[:,0,np.newaxis]
    
    return vertices

def dependent_endmember_site_occupancies(solution_model, as_fractions=False):
    n_sites = len(solution_model.sites)
    nullspace = np.array(Matrix(solution_model.endmember_occupancies).nullspace(),
                         dtype=np.float)
    
    equalities = np.zeros((len(nullspace)+1, solution_model.n_occupancies+1))
    equalities[0,0] = -n_sites
    equalities[0,1:] = 1

    if len(nullspace) > 0:
        equalities[1:,1:] = nullspace
    
    polytope = cdd.Matrix(equalities, linear=True, number_type='fraction')
    
    polytope.extend(np.concatenate((np.zeros((len(equalities[0])-1, 1)),
                                    np.identity(len(equalities[0])-1)),
                                   axis=1),
                    linear=False)
    P = cdd.Polyhedron(polytope)
    if as_fractions:
        V = np.array([map(Fraction, v) for v in P.get_generators()])
    else:
        V = np.array([map(float, v) for v in P.get_generators()])
    vertices = V[:,1:]/V[:,0,np.newaxis]
    return vertices

def dependent_endmember_sums(solution_model, as_fractions=False):
    vertices = dependent_endmember_site_occupancies(solution_model,
                                                    as_fractions=as_fractions)
    independent_sums = np.linalg.lstsq(solution_model.endmember_occupancies.T,
                                       vertices.T,
                                       rcond=None)[0].T.round(decimals=12)
    
    return independent_sums

def independent_row_indices(array):
    _, pivots, swaps = Matrix(array)._row_reduce(iszerofunc=lambda x: x.is_zero,
                                                 simpfunc=nsimplify)
    indices = np.array(range(len(array)))
    for swap in np.array(swaps):
        indices[swap] = indices[swap[::-1]]

    return indices[:len(pivots)]

def feasible_solution_basis_in_component_space(solution, components):
    """
    Note that this function finds the extreme endmembers and finds the subset within the components. Thus, starting with a solution with a disordered endmember and then restricting component range may produce a smaller solution than intended. For example, with the endmembers [A] and [A1/2B1/2], the extreme endmembers are [A] and [B]. A component space A--AB will result in only endmember [A] being valid!!
    """

    # 1) Convert components into a matrix
    component_array, component_elements = compositional_array([dictionarize_formula(c) for c in components])

    # 2) Get the full set of endmembers (dependent and otherwise)
    dependent_sums = dependent_endmember_sums(solution.solution_model)
    
    # 3) Get the endmember compositional array
    independent_endmember_array, endmember_elements = compositional_array(solution.endmember_formulae)
    all_endmember_array = dependent_sums.dot(independent_endmember_array).round(decimals=12)
    n_all = len(all_endmember_array)

    # 4) Find the endmembers that can be described with a linear combination of components

    # 4a) First, add elements to endmember_elements which are in component_elements
    for el in component_elements:
        if el not in endmember_elements:
            endmember_elements.append(el)
            all_endmember_array = np.concatenate((all_endmember_array, np.zeros((n_all,1))), axis=1)

    # 4b) Get rid of endmembers which have elements not in component_elements
    element_indices_for_removal = [i for i, el in enumerate(endmember_elements) if el not in component_elements]

    endmember_indices_for_removal = []
    for idx in element_indices_for_removal:
        endmember_indices_for_removal.extend(np.nonzero(all_endmember_array[:,idx])[0])
    possible_endmember_indices = np.array([i for i in range(n_all)
                                           if i not in np.unique(endmember_indices_for_removal)])

        
    # 4c) Cross-reference indices of elements
    n_el = len(component_elements)
    element_indexing = np.empty(n_el, dtype=int)
    for i in range(n_el):
        element_indexing[i] = endmember_elements.index(component_elements[i])

    # 4d) Find independent endmember set
    linear_solutions_exist = lambda A, B: [linprog(np.zeros(len(A)), A_eq=A.T, b_eq=b).success for b in B]
    exist = linear_solutions_exist(component_array,
                                   all_endmember_array[possible_endmember_indices[:, None],element_indexing])
    endmember_indices = possible_endmember_indices[exist]
    independent_indices = endmember_indices[independent_row_indices(dependent_sums[endmember_indices])]

    # 5) Return new basis in terms of proportions of the original endmember set
    return dependent_sums[independent_indices]

def complete_basis(basis):
    # Creates a full basis by filling remaining rows with
    # rows of the identity matrix with row indices not
    # in the column pivot list of the basis RREF
    n, m = basis.shape
    if n < m:
        complete_basis = np.empty((m, m))
        pivots=list(Matrix(basis).rref()[1])
        return np.concatenate((basis,
                               np.identity(m)[[i for i in range(m)
                                               if i not in pivots],:]),
                              axis=0)
    else:
        return basis
            
def transform_solution_to_new_basis(solution, new_basis, n_mbrs = None,
                                    solution_name=None, endmember_names=None,
                                    molar_fractions=None):

    new_basis = np.array(new_basis)
    if n_mbrs is None:
        n_mbrs, n_all_mbrs = new_basis.shape
    else:
        _, n_all_mbrs = new_basis.shape
        
    if solution_name is None:
        name = solution.name+' (modified)'
    else:
        name = solution_name 

    solution_type = solution.solution_type
    if solution_type == 'ideal':
        ESV_modifiers = [[0.,0.,0.] for v in new_basis]
        
    elif (solution_type == 'asymmetric' or
          solution_type == 'symmetric'):

        A = complete_basis(new_basis).T
            
        diag_a = np.diag(solution.solution_model.alphas)
        alphas = A.T.dot(solution.solution_model.alphas)
        inv_diag_alphas = np.diag(1./np.array(alphas))
        B = diag_a.dot(A).dot(inv_diag_alphas)
        alphas=list(alphas[0:n_mbrs])
        
        Qe = B.T.dot(solution.solution_model.We).dot(B)
        Qs = B.T.dot(solution.solution_model.Ws).dot(B)
        Qv = B.T.dot(solution.solution_model.Wv).dot(B)
        
        def new_interactions(Q, n_mbrs):
            return [[float((Q[i,j] + Q[j,i] - Q[i,i] - Q[j,j]) *
                           (alphas[i] + alphas[j])/2.)
                     for j in range(i+1, n_mbrs)]
                    for i in range(n_mbrs-1)]
    
        energy_interaction=new_interactions(Qe, n_mbrs)
        entropy_interaction=new_interactions(Qs, n_mbrs)
        volume_interaction=new_interactions(Qv, n_mbrs)

        ESV_modifiers = [[Qe[i,i]*diag_a[i,i], Qs[i,i]*diag_a[i,i], Qv[i,i]*diag_a[i,i]]
                         for i in range(n_mbrs)]
        

    else:
        raise Exception('The function to change basis for the {0} solution model has not yet been implemented.'.format(solution_type))

    # Create site formulae
    new_occupancies= np.array(new_basis).dot(solution.solution_model.endmember_occupancies)
    site_formulae = site_occupancies_to_strings(solution.solution_model.sites,
                                                solution.solution_model.site_multiplicities,
                                                new_occupancies)

    # Create endmembers
    endmembers = []
    for i, vector in enumerate(new_basis):
        nonzero_indices = np.nonzero(vector)[0]
        if len(nonzero_indices) == 1:
            endmembers.append([solution.endmembers[nonzero_indices[0]][0],
                               site_formulae[i]])
        else:
            mbr = CombinedMineral([solution.endmembers[idx][0]
                                   for idx in nonzero_indices],
                                  [vector[idx] for idx in nonzero_indices],
                                  ESV_modifiers[i])
            endmembers.append([mbr, site_formulae[i]])

    if endmember_names is not None:
        for i in range(n_mbrs):
            endmembers[i][0].params['name'] = endmember_names[i]
            endmembers[i][0].name = endmember_names[i]

    if n_mbrs == 1:
        return endmembers[0][0]
    else:
        new_solution = SolidSolution(name=name,
                             solution_type=solution_type,
                             endmembers=endmembers,
                             energy_interaction=energy_interaction,
                             volume_interaction=volume_interaction,
                             entropy_interaction=entropy_interaction,
                             alphas=alphas,
                             molar_fractions=molar_fractions)
        new_solution.parent = solution
        new_solution.basis = new_basis
        return new_solution


def feasible_solution_in_component_space(solution, components,
                                         solution_name=None, endmember_names=None,
                                         molar_fractions=None):
    """
    Note that this function finds the extreme endmembers and finds the subset within the components. Thus, starting with a solution with a disordered endmember and then restricting component range may produce a smaller solution than intended. For example, with the endmembers [A] and [A1/2B1/2], the extreme endmembers are [A] and [B]. A component space A--AB will result in only endmember [A] being valid!!
    """
    new_basis = feasible_solution_basis_in_component_space(solution, components)
    return transform_solution_to_new_basis(solution, new_basis,
                                           solution_name=solution_name,
                                           endmember_names=endmember_names,
                                           molar_fractions=molar_fractions)
