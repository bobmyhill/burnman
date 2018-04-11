# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import warnings

from scipy.optimize import nnls, minimize
from scipy.linalg import lu_factor, lu_solve

from string import ascii_uppercase as ucase
#from .processchemistry import read_masses, dictionarize_formula, formula_mass, solution_bounds
from .processchemistry import solution_bounds
from .nonlinear_solvers import damped_newton_solve
from collections import namedtuple

from .mineral import Mineral
from .solidsolution import SolidSolution
from .composite import Composite

from sympy import Matrix, nsimplify
from sympy import zeros as sp_zeros


def simplify_matrix(arr):
    def f(i,j):
        return nsimplify(arr[i][j])
    return Matrix( len(arr), len(arr[0]), f )

def get_formulae_indices_endmembers(assemblage, elements):
    #Listify the elements and sort them so they have a consistent order.
    #This will be the ordering of the rows.  The ordering of the columns
    #will be the ordering of the endmembers as they are passed in.
    
    formulae = []
    indices = []
    endmembers_per_phase = []
    for ph_idx, ph in enumerate(assemblage.phases):
        indices.append([])
        
        # Add the endmembers if it is a solid solution
        if isinstance(ph, SolidSolution):          
            for e_idx, e in enumerate(ph.endmembers):
                f = e[0].params['formula']
                if all(keys in elements for keys in f.keys()): # make sure endmember is within 
                    formulae.append(f)
                    indices[-1].append(e_idx)
            endmembers_per_phase.append(len(ph.endmembers))
                    
        # Add formula if it is a simple mineral
        elif isinstance(ph, Mineral):
            f = ph.params['formula']
            if all(keys in elements for keys in f.keys()):
                formulae.append(f)
                indices[-1].append(None)
                endmembers_per_phase.append(1)
        else:
            raise Exception('Unsupported mineral type, can only read burnman.Mineral or burnman.SolidSolution')
    
        
    return formulae, indices, endmembers_per_phase

def calculate_constraints(assemblage, indices):
    n_phase_mbrs = [len(i) for i in indices]
    n_mbrs = sum(n_phase_mbrs)
    
    bounds = []
    n_constraints = 0
    n_raw_constraints = 0
    for i, n in enumerate(n_phase_mbrs):
        e_indices = indices[i]
        n_constraints += 1
        if n == 1:
            bounds.append(np.array([[]]))
            n_raw_constraints += 1
        else:
            bounds.append(solution_bounds(assemblage.phases[i].solution_model.endmember_occupancies[e_indices]))
            n_constraints += len(bounds[-1][0])
            n_raw_constraints += len(bounds[-1][0])

            
    constraint_vector = np.zeros((n_constraints+2))
    constraint_matrix = np.zeros((n_constraints+2, n_mbrs+2)) # includes P and T
    raw_constraint_matrix = np.zeros((n_raw_constraints, n_mbrs))

    constraint_matrix[0,0] = -1 # P>0
    constraint_matrix[1,1] = -1 # T>0
    
    rcidx = 0 # index of current raw compositional constraint
    cidx = 2 # index of current compositional constraint
    pidx = 0 # starting index of current phase
    for i, n in enumerate(n_phase_mbrs):
        m = len(bounds[i][0])
        # The first endmember proportion is not a free variable (all endmembers in any solution must sum to one)
        # Re-express the constraints without the first endmember
        constraint_matrix[cidx,pidx+2] = -1. # First, need the proportions of the phases to be >0
        cidx += 1
        if m == 0:
            raw_constraint_matrix[rcidx,pidx] = 1.
            rcidx += 1
        else:
            constraint_vector[cidx:cidx+m] = -bounds[i][0] # f(mbr[0]) = (1 - f(mbr[1:]))
            constraint_matrix[cidx:cidx+m,pidx+1+2:pidx+n+2] = np.einsum('i, j', bounds[i][0], np.ones_like(bounds[i][1:,0])) - bounds[i].T[:,1:]
            raw_constraint_matrix[rcidx:rcidx+m,pidx:pidx+n] = bounds[i].T
            cidx += m 
        rcidx += m
        pidx += n
    return constraint_vector, constraint_matrix, raw_constraint_matrix

def func_min_cdist(x, equality_matrix, baseline_endmember_amounts, reaction_vectors):
    diffs =  equality_matrix.dot(baseline_endmember_amounts + reaction_vectors.dot(x))
    return np.sum(diffs*diffs)

def calculate_baseline_endmember_amounts(assemblage, equality_constraints, prm): 
    # We use the nullspace to find the vector 
    # that minimizes the change in composition of the phases.
    # The molar proportions of the phases do not matter.

    A = prm.stoichiometric_matrix[:,:] 
    b = prm.bulk_composition_vector
    
    # Check if the problem involves equality constraints
    # If it does, add the constraints to A, and a zero to b.
    reduced_nullspace = prm.stoic_nullspace[:,:] # matrix_copy
    for (code, constraint) in equality_constraints:
        if code == 'X':
            c_vector = parameter_constraints_to_endmember_constraint_vector(constraint, prm.indices)
            A = A.col_join(c_vector.T)
            b = np.append(b, np.array([0.]))
            vals = sorted([(i, v) for i, v in enumerate(reduced_nullspace * c_vector)], key=lambda x: np.abs(x[1]))
            row_indices, values = zip(*vals)
            a = Matrix(values[:-1])/values[-1]
            reduced_nullspace = reduced_nullspace[[row_indices[:-1]]] - np.outer(a, reduced_nullspace[[row_indices[-1]]])

    # Convert stoichiometric matrix from endmember to the reduced site formalism using the raw constraint matrix R.
    # This allows us to use nonnegative least squares.

    # x' = R.x, x' > 0
    # therefore Rinv.x' = x
    # We want to solve A.x = b
    # This is equivalent to A.Rinv.x' = b, for which nnls is appropriate
    R = simplify_matrix(prm.raw_constraint_matrix)
    Rinv = (R.H * R) ** -1 * R.H # the left inverse of R

    A = A*Rinv
    baseline_proportions = nnls(A, b)
    eps = 1.e-12
    if  baseline_proportions[1] > eps :
        raise Exception( "Composition cannot be represented by the given minerals.")
    # Now we can move within the nullspace (still subject to the non-negativity
    # constraints) to minimize the difference between the preferred and guessed
    # solution compositions
    # A(x + N^T.v) = b
    # A(Rinv.x' + N^T.v) = b
    # ARinv(x' + R.N^T.v) = b

    baseline_proportions = np.abs(baseline_proportions[0])
    if len(reduced_nullspace) != 0:
        reaction_vectors = reduced_nullspace.T
        num = np.eye((prm.raw_constraint_matrix.shape[1]))
        denom = np.eye((prm.raw_constraint_matrix.shape[1]))
        proportions = np.ones((denom.shape[1]))
        i=0
        for phase_idx, mbr_indices in enumerate(prm.indices):
            n_mbrs = len(mbr_indices)
            denom[i:i+n_mbrs] = np.einsum('ij->j', denom[i:i+n_mbrs])
            if n_mbrs > 1:
                guess = assemblage.phases[phase_idx].guess[mbr_indices]
                proportions[i:i+n_mbrs] = guess/np.sum(guess)
            i+=n_mbrs
        equality_matrix = num - np.einsum('ij, i->ij', denom, proportions)
        
        # convert back to simple endmembers
        baseline_endmember_amounts = Rinv.dot(baseline_proportions)

        cons_fun = lambda x:  R.dot(baseline_endmember_amounts + reaction_vectors.dot(x)) # >= 0
        cons = ({'type': 'ineq', 'fun': cons_fun}) 
        
        sol = minimize(func_min_cdist, [0.]*reaction_vectors.shape[1],
                       args=(equality_matrix, baseline_endmember_amounts,
                             reaction_vectors),
                       method='COBYLA', constraints=cons)

        assert(sol.status==1)

        # add the reaction vector
        baseline_endmember_amounts += reaction_vectors.dot(sol.x)
        assert(min(baseline_proportions) > -1.e-12)
    else:
        baseline_endmember_amounts = Rinv.dot(baseline_proportions)


    # Get compositional residuals
    res = prm.stoichiometric_matrix.dot(baseline_endmember_amounts) - prm.bulk_composition_vector
    if any(res > 1.e-9):
        print('Residuals:\n{0}'.format(res))
        raise Exception( "Baseline assemblage refinement failed." )
    return baseline_endmember_amounts

def calculate_baseline_endmember_amounts_2(assemblage, prm):
    n_phases = len(prm.indices)
    A = np.zeros((prm.stoichiometric_matrix.shape[0], n_phases))
    i=0
    for phase_idx, mbr_indices in enumerate(prm.indices):
        n_mbrs = len(mbr_indices)
        if n_mbrs == 1:
            A[phase_idx,:] = prm.stoichiometric_matrix[i,:] 
        else:
            guessed_proportions = assemblage.phases[phase_idx].guess[mbr_indices]
            guessed_proportions = Matrix(guessed_proportions/sum(guessed_proportions))
            A[:,phase_idx] = prm.stoichiometric_matrix[:,i:i+n_mbrs].dot(guessed_proportions)
        i += n_mbrs

    b = prm.bulk_composition_vector
    phase_proportions = nnls(A, b)[0]

    baseline_endmember_amounts = np.zeros((prm.stoichiometric_matrix.shape[1]))
    i=0
    for phase_idx, mbr_indices in enumerate(prm.indices):
        n_mbrs = len(mbr_indices)
        if n_mbrs == 1:
            baseline_endmember_amounts[i] = phase_proportions[phase_idx]
        else:
            guessed_proportions = assemblage.phases[phase_idx].guess[mbr_indices]
            guessed_proportions = np.array(guessed_proportions/sum(guessed_proportions))
            baseline_endmember_amounts[i:i+n_mbrs] = guessed_proportions*phase_proportions[phase_idx]
        i += n_mbrs
    
    res = prm.stoichiometric_matrix.dot(baseline_endmember_amounts) - prm.bulk_composition_vector
    if any(res > 1.e-9):
        print('Residuals:\n{0}'.format(res))
        raise Exception( "Baseline assemblage refinement failed." )
    
    return baseline_endmember_amounts
    

def get_parameters_from_state_and_endmember_amounts(state, assemblage, prm):
    parameters = np.zeros(len(prm.baseline_endmember_amounts) + 2)
    parameters[0:2] = state
    i=0
    phase_amounts = []
    for phase_idx, mbr_indices in enumerate(prm.indices):
        n_mbrs = len(mbr_indices)
        if mbr_indices[0] is None: # endmember
            parameters[i+2] = prm.baseline_endmember_amounts[i]
            i+=1
        else:
            mbr_amounts = prm.baseline_endmember_amounts[i:i+n_mbrs]
            parameters[i+2] = sum(mbr_amounts)
            if parameters[i+2] > 1.e-12:
                parameters[i+3:i+2+n_mbrs] = mbr_amounts[1:]/sum(mbr_amounts)
            else:
                mbr_prps = assemblage.phases[phase_idx].guess[mbr_indices]
                parameters[i+3:i+2+n_mbrs] = mbr_prps[1:]/sum(mbr_prps)
            i+=n_mbrs
    return parameters

def get_parameters(assemblage, indices):
    x_ph = assemblage.n_moles*np.array(assemblage.molar_fractions)
    n_indices = sum([len(i) for i in indices])
    params = np.zeros((n_indices+2))
    params[0] = assemblage.pressure
    params[1] = assemblage.temperature
    if np.isnan(params[0]):
        raise Exception('You need to set_state before getting parameters')
    j=2
    for i, mbr_indices in enumerate(indices):
        params[j] = x_ph[i]
        if mbr_indices[0] is not None:
            params[j+1:j+len(mbr_indices)] = [assemblage.phases[i].molar_fractions[idx]
                                              for idx in mbr_indices[1:]]
        j+=len(mbr_indices)
    return params


def parameters_to_endmember_amounts(parameters, indices):
    n_endmembers = len(parameters) - 2
    amounts = np.zeros(n_endmembers)
    j = 0
    for mbr_indices in indices:
        n_mbrs = len(mbr_indices)
        amounts[j:j+n_mbrs] = parameters[j+2:j+2+n_mbrs]
        total = amounts[j]
        amounts[j] = 1. - np.sum(amounts[j+1:j+n_mbrs])
        amounts[j:j+n_mbrs] *= total
        j+=n_mbrs
    return amounts


def parameter_constraints_to_endmember_constraint_vector(constraint, indices):
    """
    Unlike the parameter constraints, the endmember constraints Ax=b should always have b=0
    """
    n_endmembers = sum([len(i) for i in indices])
    values = []
    for i, start_idx in enumerate(phase_proportion_parameter_indices(indices)):
        n_mbrs = len(indices[i])
        values.extend([nsimplify(constraint[0][start_idx] + constraint[1])]*n_mbrs)

    return (Matrix(values))
    
def get_endmember_amounts(assemblage, indices):
    x_ph = assemblage.n_moles*np.array(assemblage.molar_fractions)
    n_indices = sum([len(i) for i in indices])
    amounts = np.zeros((n_indices))
    j=0
    for i, mbr_indices in enumerate(indices):
        if mbr_indices[0] == None:
            amounts[j] = x_ph[i]
        else:
            amounts[j:j+len(mbr_indices)] = [x_ph[i]*assemblage.phases[i].molar_fractions[idx]
                                             for idx in mbr_indices]
        j+=len(mbr_indices)
    return amounts
    
    
def set_compositions_and_state_from_parameters(x, assemblage, indices, endmembers_per_phase):
    assemblage.set_state(x[0], x[1])
    i=2
    phase_amounts = np.zeros((len(indices)))
    for phase_idx, mbr_indices in enumerate(indices):
        phase_amounts[phase_idx] = x[i]
        i+=1
        if mbr_indices[0] is not None:
            molar_fractions = [0.]*endmembers_per_phase[phase_idx]
            for idx in mbr_indices[1:]:
                molar_fractions[idx] = x[i]
                i+=1
            molar_fractions[mbr_indices[0]] = 1. - sum(molar_fractions)
            assemblage.phases[phase_idx].set_composition(molar_fractions)

    assert(np.all(phase_amounts > -1.e-8))
    phase_amounts = np.abs(phase_amounts)
    assemblage.n_moles = sum(phase_amounts)
    assemblage.set_fractions(phase_amounts/assemblage.n_moles)

'''
def scaling(equality_constraints, n_indices, n_reactions):
    s = np.ones((2 + n_indices))
    for i, (type_c, eq_c) in enumerate(equality_constraints):
        if type_c == 'P':
            s[i] = 1.e-9 # 1 GPa
        elif type_c == 'T':
            s[i] = 1.e-2 # 100 K
        elif type_c == 'S':
            s[i] = 0.1 # 10 J/K
        elif type_c == 'V':
            s[i] = 1.e6 # 1.e-6 m^3
        elif type_c == 'PT_ellipse':
            s[i] = 1. # already scaled
        elif type_c == 'X':
            s[i] = 1. # typically ~1
        else:
            raise Exception('constraint type not recognised')
    s[2:2+n_reactions] = 1.e-4 # 10. kJ/mol
    return s
'''
        
def F(x, assemblage, equality_constraints, prm):
    """
    x is a list of the pressure, temperature and compositional parameters
    (in that order).

    If equality_constraints[i][0] = 'P', F[i] = P - equality_constraints[i][1]
    If equality_constraints[i][0] = 'T', F[i] = T - equality_constraints[i][1]
    
    If one of equality_constraints[i][0] = 'PT', then the constraint is equivalent to
    the P-T point lying on an ellipse centered on a given point. 
    
    For each equality_constraints = X, then an extra 
    """

    set_compositions_and_state_from_parameters(x, assemblage, prm.indices, prm.endmembers_per_phase)
    new_endmember_amounts = get_endmember_amounts(assemblage, prm.indices)

    n_indices = sum([len(i) for i in prm.indices])
    n_reactions = len(prm.stoic_nullspace[:,0])
    partial_gibbs_vector = np.zeros((n_indices))
    j=0
    for i, mbr_indices in enumerate(prm.indices):
        n = len(mbr_indices)
        if mbr_indices[0] == None: # for endmembers
            partial_gibbs_vector[j] = assemblage.phases[i].gibbs
        else: # for solid solutions
            partial_gibbs_vector[j:j+n] = assemblage.phases[i].partial_gibbs[mbr_indices]
        j+=n

    # We want to find the root of the following equations
    eqns = np.zeros((2 + n_indices))
    for i, (type_c, eq_c) in enumerate(equality_constraints):
        if type_c == 'P':
            eqns[i] = x[0] - eq_c
        elif type_c == 'T':
            eqns[i] = x[1] - eq_c
        elif type_c == 'S':
            eqns[i] = assemblage.molar_entropy*assemblage.n_moles - eq_c
        elif type_c == 'V':
            eqns[i] = assemblage.molar_volume*assemblage.n_moles - eq_c
        elif type_c == 'PT_ellipse':
            v_scaled = (x[0:2] - eq_c[0])/eq_c[1]
            eqns[i] = np.linalg.norm(v_scaled) - 1.
        elif type_c == 'X':
            eqns[i] = np.dot(eq_c[0], x) - eq_c[1] # i.e. Ax = b
        else:
            raise Exception('constraint type not recognised')
        
    eqns[2:2+n_reactions] = np.dot(prm.stoic_nullspace, partial_gibbs_vector)
    eqns[2+n_reactions:] = np.dot(prm.stoic_colspace, (new_endmember_amounts - prm.baseline_endmember_amounts))
    return eqns #*scaling(equality_constraints, n_indices, n_reactions)

def jacobian(x, assemblage, equality_constraints, prm):
    # The solver always calls the Jacobian with the same
    # x parameters as used previously for the root functions
    # Therefore we don't need to set compositions or state again here
    # If the next two lines are uncommented, do it anyway.
    #set_compositions_from_parameters(x, assemblage, indices, endmembers_per_phase)
    #assemblage.set_state(P, T)

    # the jacobian J = dfi/dxj

    n_indices = sum([len(i) for i in prm.indices])
    n_reactions = len(prm.stoic_nullspace[:,0])

    # First, we find out the effect of the two constraint parameters on the
    # pressure and temperature functions:
    # i.e. dF(i, constraints)/dx[0, 1]  
    full_hessian = np.zeros((n_indices+2, n_indices+2))
    for i, (type_c, eq_c) in enumerate(equality_constraints):
        if type_c == 'P': # dP/dx
            full_hessian[i,0] = 1. # full_hessian[i, j!=0] = 0
        elif type_c == 'T': # dT/dx
            full_hessian[i,1] = 1. # full_hessian[i, j!=1] = 0
        elif type_c == 'S': # dS/dx
            # dS/dP = -aV, dS/dT = Cp/T
            full_hessian[i,0:2] = [-assemblage.n_moles*assemblage.alpha*assemblage.molar_volume, 
                                   assemblage.n_moles*assemblage.molar_heat_capacity_p/x[1]]
            j=2
            for k, mbr_indices in enumerate(prm.indices):
                n = len(mbr_indices)
                full_hessian[i,j] = assemblage.phases[k].molar_entropy
                if n > 1: # for solutions with >1 stable endmember
                    full_hessian[i,j+1:j+n] = (assemblage.n_moles*assemblage.molar_fractions[k]*
                                               (assemblage.phases[k].partial_entropies[mbr_indices[1:]] -
                                                assemblage.phases[k].partial_entropies[mbr_indices[0]]))
                j+=n
        elif type_c == 'V': # dV/dx
            # dV/dP = -V/K_T, dV/dT = aV
            full_hessian[i,0:2] = [-assemblage.n_moles*assemblage.molar_volume/assemblage.K_T,
                                   assemblage.n_moles*assemblage.molar_volume]
            j=2
            for k, mbr_indices in enumerate(prm.indices):
                n = len(mbr_indices)
                full_hessian[i,j] = assemblage.phases[k].molar_volume
                if n > 1: # for solutions with >1 stable endmember
                    full_hessian[i,j+1:j+n] = (assemblage.n_moles*assemblage.molar_fractions[k]*
                                               (assemblage.phases[k].partial_volumes[mbr_indices[1:]] -
                                                assemblage.phases[k].partial_volumes[mbr_indices[0]]))
                j+=n
        elif type_c == 'PT_ellipse': 
            v_scaled = (x[0:2] - eq_c[0])/eq_c[1]
            full_hessian[i,0:2] = v_scaled/(np.linalg.norm(v_scaled)*eq_c[1])
        elif type_c == 'X':
            full_hessian[i,:] = eq_c[0]
        else:
            raise Exception('constraint type not recognised')
        
    # Next, let's get the effect of pressure and temperature on each of the independent reactions
    # i.e. dF(i, reactions)/dx[0] and dF(i, reactions)/dx[1]
    partial_volumes_vector = np.zeros((n_indices))
    partial_entropies_vector = np.zeros((n_indices))
    j=0
    for i, mbr_indices in enumerate(prm.indices):
        n = len(mbr_indices)
        if mbr_indices[0] == None: # for endmembers
            partial_volumes_vector[j] = assemblage.phases[i].molar_volume
            partial_entropies_vector[j] = assemblage.phases[i].molar_entropy
        else: # for solid solutions
            partial_volumes_vector[j:j+n] = assemblage.phases[i].partial_volumes[mbr_indices]
            partial_entropies_vector[j:j+n] = assemblage.phases[i].partial_entropies[mbr_indices]
        j+=n
    reaction_volumes = np.dot(prm.stoic_nullspace, partial_volumes_vector)
    reaction_entropies = np.dot(prm.stoic_nullspace, partial_entropies_vector)
    
    full_hessian[2:2+len(reaction_volumes),0] = reaction_volumes  # dGi/dP = deltaVi
    full_hessian[2:2+len(reaction_volumes),1] = -reaction_entropies # dGi/dT = -deltaSi 
    
    # Pressure and temperature have no effect on the bulk compositional constraints
    # i.e. dF(i, bulk)/dx[0] and dF(i, bulk)/dx[1] = 0


    # Finally, let's build the compositional Hessian d2G/dfidfj = dmui/dfj
    # where fj is the fraction of endmember j in a phase
    phase_amounts = np.array(assemblage.molar_fractions)*assemblage.n_moles
    comp_hessian = np.zeros((n_indices, n_indices))
    dfi_dxj = np.zeros((n_indices, n_indices))
    dpi_dxj = np.zeros((n_indices, n_indices))
    j=0
    for i, mbr_indices in enumerate(prm.indices):
        n = len(mbr_indices)
        if mbr_indices[0] != None: # ignore pure phases, as changing proportions of a phase does not change its molar gibbs free energy
            comp_hessian[j:j+n,j:j+n] = assemblage.phases[i].gibbs_hessian[np.ix_(mbr_indices,mbr_indices)]
            dfi_dxj[j:j+n,j:j+n] = np.eye(n)
            dfi_dxj[j,j:j+n] -= 1. # x[0] = p(phase) and x[1:] = f[1:] - f[0]

            dpi_dxj[j:j+n,j:j+n] = dfi_dxj[j:j+n,j:j+n]*phase_amounts[i]
            dpi_dxj[j:j+n,j] = np.array(assemblage.phases[i].molar_fractions)[mbr_indices]
        else:
            dpi_dxj[j,j] = 1.
        j+=n
        
    reaction_hessian = prm.stoic_nullspace.dot(comp_hessian).dot(dfi_dxj) # dfi_dxj converts the endmember hessian to the parameter hessian.
    bulk_hessian = prm.stoic_colspace.dot(dpi_dxj) 
    full_hessian[2:,2:] = np.concatenate((reaction_hessian, bulk_hessian))
    return full_hessian # np.einsum('ij, i->ij', full_hessian, scaling(equality_constraints, n_indices, n_reactions))

def lambda_bounds(dx, x, indices):
    """
    x is a list of the pressure, temperature and compositional parameters
    (in that order).

    dx is the proposed newton step 
    """
    n_indices = sum([len(i) for i in indices])
    max_steps = np.ones((n_indices+2))*100000.

    # first two constraints are P and T
    max_steps[0:2] = [20.e9, 500.] # biggest reasonable P and T steps
    
    j=2
    for i, mbr_indices in enumerate(indices):
        n = len(mbr_indices)
        if x[j] + dx[j] < 0.: # if the phase proportion constraint would be broken, set a step that is marginally smaller
            max_steps[j] = max(x[j]*0.999, 0.001)
            #max_steps[j] = max(x[j]*0.5, 0.001) # allowed to cut amount by half...
        max_steps[j+1:j+n] = [max(xi*0.99, 0.01) for xi in x[j+1:j+n]]  # maximum compositional step
        j+=n

    max_lmda = min([1. if step <= max_steps[i] else max_steps[i]/step
                    for i, step in enumerate(np.abs(dx))])
    
    #return (1.e-8, 1.)
    return (min(1.e-8, max_lmda/4.), max_lmda)

def phase_proportion_parameter_indices(indices):
    i=2
    idxs = []
    for mbr_indices in indices:
        idxs.append(i)
        i+=len(mbr_indices)
    return idxs

def dxidxj(x, assemblage, equality_constraints):
    # dx[i]/dx[j], where dx[j] = P or T
    # dFi/dxj * dxj/dxk = dFi/dxk
    J = jacobian(parameters, assemblage, equality_constraints)
    return np.linalg.solve(J, J.T)

# The following two functions construct two common examples of
# compositional constraints. Note that constraints in the form
# sum(ax + b)/sum(cx + d) - f = 0
# can be recast as:
# (a-fc)*x = fd - b 
# which is less easy for a human to understand
# (in terms of chemical constraints), but much easier to solve.
def phase_proportion_constraint(phase, assemblage, indices, proportion):
    n_indices = sum([len(i) for i in indices])
    phase_idx = assemblage.phases.index(phase)
    proportion_indices = phase_proportion_parameter_indices(indices)

    constraints = []
    for p in proportion:
        constraints.append(['X', [np.zeros((n_indices+2)), 0.]])
        constraints[-1][-1][0][proportion_indices] = -p
        constraints[-1][-1][0][proportion_indices[phase_idx]] += 1.
            
    return constraints

def phase_composition_constraint(phase, assemblage, indices, constraint):
    phase_idx = assemblage.phases.index(phase)
    start_idx = int(sum([len(i) for i in indices[:phase_idx]])) + 3. # +3 comes from P, T and the proportion of the phase of interest
    n_indices = sum([len(i) for i in indices])
    mbr_indices = indices[phase_idx]

    site_names, numerator, denominator, value = constraint
    site_indices = [phase.solution_model.site_names.index(name) for name in site_names]
    atoms = np.dot(phase.solution_model.endmember_noccupancies[mbr_indices][:,site_indices], np.array([numerator, denominator]).T)

    atoms0 = atoms[0]
    atoms -= atoms[0]
    numer, denom = atoms.T[:,1:]

    constraints = []
    for v in value:
        f = v*atoms0[1] - atoms0[0]
        constraints.append(['X', [np.zeros((n_indices+2)), f]])
        constraints[-1][1][0][start_idx:start_idx+len(mbr_indices)-1] = numer - v*denom
        
    return constraints




##############################################################


def equilibrate(composition, assemblage, equality_constraints,
                initial_state=[5.e9, 1000.], tol=1.e-3,
                store_iterates=False, max_iterations=100.,verbose=True):
    
    prm = namedtuple('assemblage_parameters', [])
    
    # Process elements
    prm.elements = list(set(composition.keys()))
    prm.formulae, prm.indices, prm.endmembers_per_phase = get_formulae_indices_endmembers(assemblage, prm.elements)

    # Process parameter names
    prm.parameter_names = ['Pressure (Pa)', 'Temperature (K)']
    for i, mbr_indices in enumerate(prm.indices):
        prm.parameter_names.append('x({0})'.format(assemblage.phases[i].name))
        for mbr_idx in mbr_indices[1:]:
            prm.parameter_names.append(' p({0} in {1})'.format(assemblage.phases[i].endmembers[mbr_idx][0].name, assemblage.phases[i].name))


    

    # Find the bulk composition vector
    prm.bulk_composition_vector = np.array([composition[e] for e in prm.elements])

    # Populate the stoichiometric matrix
    def f(i,j):
        e = prm.elements[i]
        if e in prm.formulae[j]:
            return nsimplify(prm.formulae[j][e])
        else:
            return 0
    prm.stoichiometric_matrix = Matrix( len(prm.elements), len(prm.formulae), f )
    prm.stoic_colspace = np.array([v.T[:] for v in prm.stoichiometric_matrix.T.columnspace()])
    prm.stoic_nullspace = np.array([v.T[:] for v in prm.stoichiometric_matrix.nullspace()])    
    prm.constraint_vector, prm.constraint_matrix, prm.raw_constraint_matrix = calculate_constraints(assemblage, prm.indices)

    
    # Check equality constraints have the correct structure
    # Convert them into versions readable by the function and jacobian functions
    equality_constraint_lists = []
    for i in range(2):
        if equality_constraints[i][0] == 'phase_proportion':
            phase=equality_constraints[i][1][0]
            proportion=equality_constraints[i][1][1]
            if isinstance(proportion, float):
                proportion = np.array([proportion])
            if not isinstance(proportion, np.ndarray):
                raise Exception('The constraint proportion in equality {0} should be '
                                'a float or numpy array'.format(i+1))
            equality_constraint_lists.append(phase_proportion_constraint(phase, assemblage,
                                                                         prm.indices, proportion))
        elif equality_constraints[i][0] == 'phase_composition':
            phase=equality_constraints[i][1][0]
            constraint=equality_constraints[i][1][1]
            if isinstance(constraint[3], float):
                constraint = (constraint[0], constraint[1], constraint[2], np.array([constraint[3]]))
            if not isinstance(constraint[3], np.ndarray):
                raise Exception('The last constraint parameter in equality {0} should be '
                                'a float or numpy array'.format(i+1))
            
            equality_constraint_lists.append(phase_composition_constraint(phase, assemblage,
                                                                          prm.indices, constraint))
        elif equality_constraints[i][0] == 'X':
            constraint=equality_constraints[i][1][1]
            if isinstance(constraint[-1], float):
                constraint = (constraint[0], np.array([constraint[-1]]))
            if not isinstance(constraint[-1], np.ndarray):
                raise Exception('The last constraint parameter in equality {0} should be '
                                'a float or numpy array'.format(i+1))
            equality_constraint_lists.append([[equality_constraints[i][0],
                                               [equality_constraints[i][0][0], p]]
                                              for p in equality_constraints[i][0][1]])
            
        elif (equality_constraints[i][0] == 'P' or
              equality_constraints[i][0] == 'T' or
              equality_constraints[i][0] == 'PT_ellipse' or
              equality_constraints[i][0] == 'S' or
              equality_constraints[i][0] == 'V'):
            if isinstance(equality_constraints[i][1], float):
                equality_constraints[i] = (equality_constraints[i][0], np.array([equality_constraints[i][1]]))
            if not isinstance(equality_constraints[i][1], np.ndarray):
                raise Exception('The last parameter in equality {0} should be '
                                'a float or numpy array'.format(i+1))
            equality_constraint_lists.append([[equality_constraints[i][0], p]
                                              for p in equality_constraints[i][1]])
        else:
            raise Exception('The type of equality_constraint is '
                            'not recognised for constraint {0}.\n'
                            'Should be one of P, T, S, V, X,\n'
                            'PT_ellipse, phase_proportion, or phase_composition.'.format(i+1))

    
    sol_list = []
    n_c0 = len(equality_constraint_lists[0])
    n_c1 = len(equality_constraint_lists[1])

    proportion_indices = phase_proportion_parameter_indices(prm.indices)
    
    prm.baseline_endmember_amounts = calculate_baseline_endmember_amounts(assemblage,
                                                                          [equality_constraint_lists[0][0],
                                                                           equality_constraint_lists[1][0]],
                                                                          prm)

    prm.initial_parameters = get_parameters_from_state_and_endmember_amounts(initial_state, assemblage, prm)

    sol_list = np.empty(shape=(n_c0, n_c1)+(0,)).tolist()
    for i_c0 in range(n_c0):
        new_c0 = True
        for i_c1 in range(n_c1):
            if verbose:
                string = 'Processing solution'
                if n_c0 > 1:
                    string += ' {0}/{1}'.format(i_c0+1, n_c0)
                if n_c1 > 1:
                    string += ' {0}/{1}'.format(i_c1+1, n_c1)
                print(string+':')
            # modify initial state if necessary
            equality_constraints = [equality_constraint_lists[0][i_c0], equality_constraint_lists[1][i_c1]]
            for i in range(2):
                if equality_constraints[i][0] == 'P':
                    initial_state[0] = equality_constraints[i][1]
                elif equality_constraints[i][0] == 'T':
                    initial_state[1] = equality_constraints[i][1]
                elif equality_constraints[i][0] == 'PT_ellipse':
                    initial_state = equality_constraints[i][1][1]
    
            # Set the initial proportions and compositions of the phases in the assemblage:
            set_compositions_and_state_from_parameters(prm.initial_parameters, assemblage,
                                                       prm.indices, prm.endmembers_per_phase)
            
            
            sol = damped_newton_solve(F = lambda x: F(x, assemblage, equality_constraints, prm),
                                      J = lambda x: jacobian(x, assemblage, equality_constraints, prm),
                                      lambda_bounds = lambda dx, x: lambda_bounds(dx, x, prm.indices),
                                      guess = prm.initial_parameters,
                                      linear_constraints = (prm.constraint_matrix, prm.constraint_vector),
                                      tol=tol,
                                      store_iterates=store_iterates, max_iterations=max_iterations)

                
            if verbose:
                print(sol.text)
                
            sol_list[i_c0][i_c1] = sol
            new_c0 = False

            prev = []
            if i_c1 < n_c1 - 1:
                next_ecs = [i_c0, i_c1 + 1]
            elif i_c1 == n_c1 - 1 and i_c0 < n_c0 - 1:
                next_ecs = [i_c0+1, 0]
            else: # last value
                next_ecs = None

            if next_ecs is not None:
                cs = [equality_constraint_lists[0][next_ecs[0]], equality_constraint_lists[1][next_ecs[1]]]
                prev_sol = []
                if next_ecs[0] != 0:
                    prev_sol.append(sol_list[next_ecs[0] - 1][next_ecs[1]])
                if next_ecs[1] != 0:
                    prev_sol.append(sol_list[next_ecs[0]][next_ecs[1] - 1])

                updated_params = False
                for s in prev_sol:
                    if s.success and not updated_params:
                        dF = F(s.x, assemblage, cs, prm)
                        luJ = lu_factor(s.J)
                        new_parameters = s.x + lu_solve(luJ, -dF) # next guess based on a Newton step
                    
                        c = prm.constraint_matrix.dot(new_parameters) + prm.constraint_vector
                        if all(c <= 0.):
                            prm.initial_parameters = new_parameters
                            updated_params = True
                        else:
                            exhausted_phases = [assemblage.phases[phase_idx].name
                                                for phase_idx, v in
                                                enumerate(new_parameters[proportion_indices]) if v<0.]
                            if len(exhausted_phases) > 0 and verbose:
                                print('A phase might be exhausted before the next step: {0}'.format(exhausted_phases))
                if not updated_params:
                    prm.baseline_endmember_amounts = calculate_baseline_endmember_amounts(assemblage,
                                                                                          cs,
                                                                                          prm)
                    prm.initial_parameters = get_parameters_from_state_and_endmember_amounts(initial_state, assemblage, prm)

    # Finally, make dimensions of sol_list equal the input dimensions 
    if len(sol_list[0]) == 1:
        sol_list = list(zip(*sol_list)[0])
    if len(sol_list) == 1:
        sol_list = sol_list[0]
    return sol_list, prm



'''
    p = parameters_to_endmember_amounts(sol.x, prm.indices)
    denom = np.empty_like(prm.raw_constraint_matrix[:,:])
    proportions = np.ones((prm.raw_constraint_matrix.shape[0]))
    i=0
    for phase_idx, mbr_indices in enumerate(prm.indices):
        n_mbrs = len(mbr_indices)
        denom[i:i+n_mbrs] = np.einsum('ij->j', prm.raw_constraint_matrix[i:i+n_mbrs])
        if n_mbrs > 1:
            guess = np.array(assemblage.phases[phase_idx].molar_fractions)[mbr_indices]
            proportions[i:i+n_mbrs] = guess/np.sum(guess)
        i+=n_mbrs
    equality_matrix = prm.raw_constraint_matrix - np.einsum('ij, i->ij', denom, proportions)
    baseline_endmember_amounts = np.linalg.lstsq(prm.raw_constraint_matrix, p)[0]
    print(func_min_cdist([0.], equality_matrix, baseline_endmember_amounts, np.array([0.]), prm))
'''    
