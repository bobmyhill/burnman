# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2019 by the BurnMan team, released under the GNU
# GPL v2 or later.

from __future__ import absolute_import

import warnings
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize, LinearConstraint, curve_fit
from scipy.linalg import inv, sqrtm
from collections import Counter
from copy import deepcopy

from . import SolidSolution
from .composite import Composite
from .solutionbases import polytope_from_solution_model
from .equilibrate import equilibrium_order_init, equilibrium_order_fast

class CompositionStorage(object):
    """
    A very simple class whose instantiation creates objects with attributes:
    fitted_elements : list of strings
        Element / species identifiers
    composition : 1D numpy array
        amounts of the fitted elements
    compositional_uncertainties : 1D or 2D numpy array
        uncertainties (1 sigma) or variance-covariance matrix
    """
    def __init__(self, fitted_elements=None,
                 composition=None,
                 compositional_uncertainties=None):
        self.fitted_elements = fitted_elements
        self.composition = composition
        self.compositional_uncertainties = compositional_uncertainties


def store_composition(phase, fitted_elements, composition, compositional_uncertainties):
    """
    Attaches a phase_storage attribute to a phase object,
    and assigns a CompositionStorage object to that attribute.
    """
    phase.phase_storage = CompositionStorage(fitted_elements,
                                             composition,
                                             compositional_uncertainties)

class AnalysedComposite(Composite):
    """
    A clone of the Composite class, which in addition to all of the standard
    Composite attributes and methods also contains a
    phase_storage attribute and a stored compositions attribute.
    """
    def __init__(self, phases, fractions=None, fraction_type='molar', name='Unnamed composite'):
        Composite.__init__(self, phases, fractions, fraction_type, name)
        self.phase_storage = [CompositionStorage() for i in phases]
        self.stored_compositions = [['composition not assigned']
                                    for k in range(len(phases))]


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


def linear_function(A, *b):
    """
    A function returning the dot product of a matrix A and
    vector b.
    """
    return A.dot(b)


def linear_jacobian(A, *b):
    """
    A function returning the matrix A,
    for use as the Jacobian in linear least squares.
    """
    return A


def unconstrained_least_squares(A, b, b_uncertainties, p0):
    """
    Solves Ap = b using linear least squares,
    given uncertainties on b.
    Parameters
    ----------
    A : 2D numpy array
        The matrix A
    b : 1D numpy array
        The vector b
    b_uncertainties : 1D or 2D numpy array
        If 1D array, then this parameter is an array of 1 sigma uncertainties
        on the parameters b. If 2D, this parameter is the variance-covariance
        matrix of b
    p0 : A starting guess for p
    Returns
    -------
    popt : 1D numpy array
        The optimized parameters
    pcov : 2D numpy array
        The covariance matrix of the parameters
    """
    popt, pcov = curve_fit(linear_function, A, b,
                           p0=p0,
                           jac=linear_jacobian,
                           sigma=b_uncertainties, absolute_sigma=True)
    return (popt, pcov)


def equilibrate_phase(assemblage, phase_index):
    """
    For solutions which can undergo order-disorder,
    the endmember proportions are a function of the bulk composition
    and of the pressure, temperature and model parameters.

    This function takes the current state of a solution
    (bulk composition, pressure, temperature, parameters),
    equilibrates the endmember proportions, and calculates the
    uncertainties in endmember proportions.

    Requires
    --------
    - Assemblage must have the compositional matrix,
      measured composition and associated uncertainties for the appropriate
      phase stored in assemblage.stored_compositions[phase_index][1].
    - The phase must have the composition and desired state already set
      (via set_composition and set_state).
    - The phase must have an attribute named rxn_matrix, corresponding to
      the nullspace of the stoichiometric matrix.

    Parameters
    ----------
    assemblage : burnman.Composite
        The composite containing the phase of interest.
    phase_index : integer
        The index of the phase of interest in the composite.

    Updates
    ------
    phase.molar_fractions : 1D numpy array
        The endmember proportions of the phase
    phase.molar_fraction_covariances : 2D numpy array
        The variance-covariance matrix of endmember proportions.
    """
    phase = assemblage.phases[phase_index]
    A, Cov_b_norm, bounds = assemblage.stored_compositions[phase_index][1]

    # Re-equilibrate at a constant bulk composition
    #equilibrium_order(phase)
    equilibrium_order_fast(phase, bounds)

    # add R.dot(H) to transformation matrix
    A_new = np.vstack((A, phase.rxn_matrix.dot(phase.gibbs_hessian)))

    n_var, n_mbrs = A_new.shape
    n_b, n_mbrs = A.shape
    n_rxns = n_var - n_b

    # Add block to covariance matrix
    sig_G = 1. # nominal 1 J uncertainty

    Cov_b_new = np.zeros((n_var, n_var))
    Cov_b_new[:n_b, :n_b] = Cov_b_norm
    Cov_b_new[n_b:, n_b:] = np.eye(n_rxns) * sig_G * sig_G


    popt = phase.molar_fractions

    # Calculate the covariance matrix
    # (also from https://stats.stackexchange.com/a/333551)
    inv_Cov_b_new = np.linalg.inv(Cov_b_new)
    pcov = np.linalg.inv(A_new.T.dot(inv_Cov_b_new.dot(A_new)))

    # Convert the variance-covariance matrix from endmember amounts to
    # endmember proportions
    dpdx = (np.eye(n_mbrs) - popt).T # same as (1. - p[i] if i == j else -p[i])
    Cov_p = dpdx.dot(pcov).dot(dpdx.T)

    phase.molar_fraction_covariances = Cov_p


def fit_composition(fitted_elements, composition, compositional_uncertainties,
                    formulae, endmember_site_occupancies,
                    normalize=True, name=None, return_Ab=False):
    """
    It is assumed that any elements not in composition were not measured
    (but may exist in unknown quantities).
    If distinct oxidation states or site occupancies were measured
    (by Moessbauer, for example), then formulae should be modified

    fitted elements should be a list of strings
    composition and compositional uncertainties should be given as arrays
    formulae should either be given as arrays
    (with columns corresponding to elements) or as dictionaries
    If the compositional uncertainties can either be sigmas
    or covariance matrices

    The composition and associated uncertainties in endmember *amounts*,
    not proportions. If normalize=True, then the endmember amounts are
    normalized to a total of one.
    """

    # Construct A, b and Cov_b
    if type(formulae[0]) is dict or type(formulae[0]) is Counter:
        A = np.array([[f[e] if e in f else 0.
                       for e in fitted_elements]
                      for f in formulae]).T
    else:
        A = formulae.T

    b = composition

    if len(compositional_uncertainties.shape) == 1:
        Cov_b = np.diag(compositional_uncertainties
                        * compositional_uncertainties)
    else:
        Cov_b = compositional_uncertainties

    # Create the standard weighted least squares objective function
    # (https://stats.stackexchange.com/a/333551)
    n_mbrs = A.shape[1]
    m = inv(sqrtm(Cov_b))
    mA = m@A
    mb = m@b
    x = cp.Variable(n_mbrs)
    objective = cp.Minimize(cp.sum_squares(mA@x - mb))

    # Define the constraints
    # Ensure that element abundances / site occupancies
    # are exactly equal to zero if the user specifies that
    # they are equal to zero.
    # Also ensure all site occupancies are non-negative
    S, S_index = np.unique(A, axis=0, return_index=True)
    S = np.array([s for i, s in enumerate(S)
                  if np.abs(b[S_index[i]]) < 1.e-10
                  and any(np.abs(s) > 1.e-10)])
    constraints = [eq@x == 0 for eq in S]

    T = np.array([t for t in np.unique(endmember_site_occupancies.T, axis=0)
                  if t not in S and any(np.abs(t) > 1.e-10)])
    T = np.array([t for t in np.unique(endmember_site_occupancies.T, axis=0)
                  if any(np.abs(t) > 1.e-10)])
    #T = np.array(np.unique(endmember_site_occupancies.T, axis=0))
    if len(T) > 0:
        constraints.extend([-eq@x <= 0 for eq in T])

    # Set up the problem and solve it
    # We catch inaccurate solution warnings,
    # as we check for poor residuals at the end of this function
    warns = []
    prob = cp.Problem(objective, constraints)
    try:
        with warnings.catch_warnings(record=True) as w:
            res = prob.solve(solver=cp.ECOS)
            popt = np.array([x.value[i] for i in range(len(A.T))])
            warns.extend(w)
    except Exception as e:
        print('ECOS Solver failed. Trying default solver.')
        try:
            with warnings.catch_warnings(record=True) as w:
                res = prob.solve()
                popt = np.array([x.value[i] for i in range(len(A.T))])
                warns.extend(w)
        except Exception as e:
            print('Oh dear, there seems to be a problem '
                  'with the following composition:')
            print(fitted_elements)
            print(f'{b} (sum = {sum(b)})')
            raise Exception(e)

    rms_norm = np.sqrt(np.mean((A.dot(popt)-b)**2))/np.sqrt(np.mean(np.array(b)**2))
    if rms_norm > 0.1:
        print(f'COMPOSITIONAL RESIDUAL VERY HIGH FOR PHASE {name} {rms_norm} {res}')
        print('This may be because you have set some of your '
              'variances too low,\nor because the provided composition '
              'does not match the stoichiometry of the desired phase.')
        if len(warns) == 0:
            print('There were no solver warnings.')
        for w in warns:
            print(w.message)
        print(A)
        print(np.sqrt(np.mean((A.dot(popt)-b)**2)))
        print(fitted_elements)
        print(f'{b} (sum = {sum(b)})')
        print(A.dot(popt))
        print(f'Endmember proportions: {popt}')
        exit()

    if return_Ab:
        pcov = 0.
    else:
        # Calculate the covariance matrix
        # (also from https://stats.stackexchange.com/a/333551)
        inv_Cov_b = np.linalg.inv(Cov_b)
        pcov = np.linalg.inv(A.T.dot(inv_Cov_b.dot(A)))
    if normalize:
        sump = sum(popt)
        popt /= sump
        pcov /= sump * sump
        res /= sump
        Cov_b_norm = Cov_b / (sump * sump)

    popt[np.abs(popt) < 1.e-10] = 0

    if return_Ab:
        return (popt, pcov, res, A, Cov_b_norm)
    else:
        # Convert the variance-covariance matrix from endmember amounts to
        # endmember proportions
        dpdx = (np.eye(n_mbrs) - popt).T # same as (1. - p[i] if i == j else -p[i])
        pcov = dpdx.dot(pcov).dot(dpdx.T)
        return (popt, pcov, res)


def compute_and_set_phase_composition(assemblage, phase_index,
                                      midpoint_proportion, verbose=False):

    # Find the correct phase
    phase = assemblage.phases[phase_index]
    store = assemblage.phase_storage[phase_index]

    # Find the name of the phase
    try:
        name = phase.name
    except AttributeError():
        name = 'unnamed solution'

    # Collect the elements (or site-elements) for fitting,
    # the amounts of those elements (as phase.composition)
    # and the associated uncertainties
    fitted_elements = store.fitted_elements
    composition = store.composition
    compositional_uncertainties = store.compositional_uncertainties

    # Count the number of endmembers in the solution
    n_mbrs = len(phase.endmember_formulae)

    # If any of the fitted elements are site-elements (e.g. Si on site A)
    # then use both the site-element endmember formulae
    # and the elemental endmember formulae
    if any(['_' in e for e in fitted_elements]):
        sfs = phase.solution_model.site_formulae
        formulae = [merge_two_dicts(sfs[i],
                                    phase.endmember_formulae[i])
                    for i in range(n_mbrs)]
    else:
        formulae = phase.endmember_formulae

    # The endmember occupancies are provided as constraints
    # for the composition fitting
    occupancies = phase.solution_model.endmember_occupancies

    # Find the best fit composition
    sol = fit_composition(fitted_elements, composition,
                          compositional_uncertainties,
                          formulae, occupancies,
                          normalize=True, name=name,
                          return_Ab=phase.ordered)

    # Set the active composition of the phase
    # with the best fit composition, and assign the covariance matrix to
    # an attribute of the phase
    phase.set_composition((1. - midpoint_proportion) * sol[0]
                          + midpoint_proportion * phase.polytope_midpoint)

    if phase.ordered:
        if phase.temperature is None:
            raise Exception('Setting the initial composition of an '
                            'order-disorder compound requires '
                            'first setting state')
        try:
            equilibrium_order_init(phase)

            # calculate bounds for later equilibrations
            A = phase.solution_model.endmember_occupancies.T
            occs = A.dot(phase.molar_fractions)
            assert(phase.rxn_matrix.shape[0] == 1)
            d_occ_d_rxn = A.dot(phase.rxn_matrix[0,:])
            potential_neg_bounds = np.array([-occs[i]/d_occ_d_rxn[i]
                                             for i in range(len(occs))
                                             if d_occ_d_rxn[i] > 1.e-10])

            potential_pos_bounds = np.array([-occs[i]/d_occ_d_rxn[i]
                                             for i in range(len(occs))
                                             if d_occ_d_rxn[i] < -1.e-10])

            min_bound = np.max(potential_neg_bounds)
            max_bound = np.min(potential_pos_bounds)
            bounds = (min_bound, max_bound)

            # Uncomment to check equilibration.
            # The result should be very close to zero.
            # print(equilibrium_order_fast(phase, bounds)[0])


        except Exception as e:
            print(e)
            print(phase.molar_fractions)
            raise Exception('Could not equilibrate during initialization.')

        _, pcov, res, A, Cov_b_norm = sol

        # store original linear problem
        assemblage.stored_compositions[phase_index] = (phase.molar_fractions,
                                                       (A, Cov_b_norm, bounds))

    else:
        popt, pcov, res = sol

        if False in (pcov < 1.):
            raise Exception(f'Error: The covariance matrix for the '
                            f'endmember proportions of '
                            f'{phase.name} contains '
                            f'elements which are unexpectedly large (>1).\n'
                            f'You probably need to provide more '
                            f'constraints in order to uniquely constrain '
                            f'the composition of this phase.\n'
                            f'{fitted_elements}\n'
                            f'{composition}\n'
                            f'{phase.molar_fractions}\n'
                            f'{pcov}')

        assemblage.stored_compositions[phase_index] = (phase.molar_fractions,
                                                       pcov)

        # Optionally print some information to stdout
        if verbose:
            print(phase.name)
            for i in range(n_mbrs):
                print('{0}: {1:.3f} +/- {2:.3f}'.format(phase.endmember_names[i],
                                                        popt[i],
                                                        np.sqrt(pcov[i][i])))


def compute_and_store_phase_compositions(assemblage, midpoint_proportion,
                                         constrain_endmembers,
                                         proportion_cutoff=0,
                                         copy_storage=False,
                                         verbose=False):
    for i, phase in enumerate(assemblage.phases):
        if isinstance(phase, SolidSolution):
            # move each phase storage object into the assemblage list
            if copy_storage:
                try:
                    assemblage.phase_storage[i] = deepcopy(phase.phase_storage)
                except AttributeError:
                    pass

            compute_and_set_phase_composition(assemblage, i,
                                              midpoint_proportion, verbose)

    if constrain_endmembers:
        declare_constrained_endmembers(assemblage, proportion_cutoff)



def declare_constrained_endmembers(assemblage, proportion_cutoff):
    for iph, phase in enumerate(assemblage.phases):
        if isinstance(phase, SolidSolution):
            # Find which endmembers to use in the reaction and stoichiometric matrices
            b = phase.molar_fractions
            A = phase.all_endmembers_as_independent_endmember_proportions

            x0 = np.zeros(len(A))
            x0[phase.independent_row_indices] = b

            sol = minimize(lambda p: np.sum(-p*p), x0=x0, method='slsqp',
                           bounds=tuple((0,None) for i in range(A.shape[0])),
                           constraints = LinearConstraint(A.T, b, b))

            new_endmember_indices = np.argsort(sol.x)[::-1][:len(A[0])]
            new_endmember_proportions = sol.x[new_endmember_indices]

            # Select new endmembers with proportions above cutoff
            dominant_endmember_indices = [new_endmember_indices[i] for
                                          i, p in enumerate(new_endmember_proportions)
                                          if p > proportion_cutoff]
            if len(dominant_endmember_indices) < len(phase.molar_fractions):
                assemblage.solution_transformations[iph] = A[dominant_endmember_indices]


def create_polytope_attributes(solution):
    solution.polytope = polytope_from_solution_model(solution.solution_model)
    A = solution.polytope.dependent_endmembers_as_independent_endmember_proportions
    solution.all_endmembers_as_independent_endmember_proportions = A
    solution.polytope_midpoint = np.sum(A, axis=0)/len(A)
    solution.independent_row_indices = solution.polytope.independent_row_indices


def create_reaction_matrix(solution):
    assemblage = Composite([solution])
    solution.rxn_matrix = assemblage.stoichiometric_matrix(calculate_subspaces=True)[1]


def assemblage_affinity_misfit(assemblage, reuse_reaction_matrix=True):
    if reuse_reaction_matrix:
        try:
            reaction_matrix = assemblage.stored_reaction_matrix
        except AttributeError:
            reaction_matrix = assemblage.stoichiometric_matrix(calculate_subspaces=True,
                                                               use_transformed_solutions=True)[1]
            assemblage.stored_reaction_matrix = reaction_matrix
    else:
        reaction_matrix = assemblage.stoichiometric_matrix(calculate_subspaces=True,
                                                           use_transformed_solutions=True)[1]


    if len(reaction_matrix) == 0:
        print('No reactions between the phases'
              ' in this assemblage: {0}'.format([ph.name
                                                 for ph in assemblage.phases]))
        return 0.
    else:
        # d(partial_gibbs)i/d(variables)j can be split into blocks
        n_mbrs = reaction_matrix.shape[1]
        #print([ph.name for ph in assemblage.phases])
        #print(assemblage.pressure/1.e9,
        #      [ph.molar_fractions for ph in assemblage.phases
        #       if ph.name == 'garnet'][0])
        #print(reaction_matrix)
        #XXXX TODO!!
        #Check covariance matrix scaling.
        #Check phase compositions.

        Cov_mu = np.zeros((n_mbrs, n_mbrs))
        dmudPT = np.zeros((n_mbrs, 2))
        mu = np.zeros(n_mbrs)

        i = 0
        for iph, phase in enumerate(assemblage.phases):
            if isinstance(phase, SolidSolution):
                T = assemblage.solution_transformations[iph]
                if (T is not None):
                    p_mbrs = len(T)
                    hess = np.einsum('ij, kj->ki', phase.gibbs_hessian, T)

                    pGPT = np.array([phase.partial_gibbs,
                                     phase.partial_volumes,
                                     -phase.partial_entropies])
                    pGPT = np.einsum('ij, kj', pGPT, T)

                    Cov_mu[i:i+p_mbrs, i:i+p_mbrs] = (hess).dot(phase.molar_fraction_covariances).dot(hess.T)
                    mu[i:i+p_mbrs] = pGPT[0]
                    dmudPT[i:i+p_mbrs] = pGPT[1:].T
                    i += p_mbrs
                else:
                    p_mbrs = phase.n_endmembers
                    Cov_mu[i:i+p_mbrs, i:i+p_mbrs] = (phase.gibbs_hessian).dot(phase.molar_fraction_covariances).dot(phase.gibbs_hessian.T) # the hessian is symmetric, so transpose only taken for legibility...
                    dmudPT[i:i+p_mbrs] = np.array([phase.partial_volumes, -phase.partial_entropies]).T
                    mu[i:i+p_mbrs] = phase.partial_gibbs
                    i += p_mbrs
            else:
                dmudPT[i] = np.array([phase.V, -phase.S])
                mu[i] = phase.gibbs
                i += 1

        Cov_mu += dmudPT.dot(assemblage.state_covariances).dot(dmudPT.T)

        # Finally, we use the reaction matrix
        # (the nullspace of the stoichiometric matrix)
        # to calculate the affinities
        a = reaction_matrix.dot(mu)
        Cov_a = reaction_matrix.dot(Cov_mu).dot(reaction_matrix.T)
        try:
            chi_sqr = a.dot(np.linalg.solve(Cov_a, a))
        except:
            try:
                print(assemblage.experiment_id)
            except AttributeError:
                pass
            print([ph.name for ph in assemblage.phases])
            print(Cov_a)
            raise Exception('Could not find misfit for this assemblage')
    return chi_sqr
