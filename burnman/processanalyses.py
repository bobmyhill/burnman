# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2019 by the BurnMan team, released under the GNU
# GPL v2 or later.

from __future__ import absolute_import

import warnings
import numpy as np
from scipy.optimize import curve_fit, minimize, LinearConstraint
from collections import Counter
from copy import deepcopy

from . import SolidSolution
from .composite import Composite
from .solutionbases import polytope_from_solution_model
from .equilibrate import equilibrium_order

class CompositionStorage(object):
    def __init__(self, fitted_elements=None,
                 composition=None,
                 compositional_uncertainties=None):
        self.fitted_elements = fitted_elements
        self.composition = composition
        self.compositional_uncertainties = compositional_uncertainties


def store_composition(phase, fitted_elements, composition, compositional_uncertainties):
    phase.phase_storage = CompositionStorage(fitted_elements,
                                             composition,
                                             compositional_uncertainties)

class AnalysedComposite(Composite):
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
    A, b, b_uncertainty = assemblage.stored_compositions[phase_index][1]

    # Re-equilibrate at a constant bulk composition
    equilibrium_order(phase)

    # add R.dot(H) to transformation matrix
    A_new = np.vstack((A,
                       phase.rxn_matrix.dot(phase.gibbs_hessian)))

    n_rxns, n_mbrs = phase.rxn_matrix.shape
    b_new = np.vstack((b, np.zeros(n_rxns)))

    # Add block to covariance matrix
    n_c = len(b)
    n_var = n_c + n_rxns
    sig_G = 1. # nominal 1 J uncertainty
    if b_uncertainties.ndim == 1:
        b_new_uncertainties = np.hstack((b_uncertainties,
                                         np.ones(n_rxns) * sig_G))
    else:
        b_new_uncertainties = np.zeros((n_var, n_var))
        b_new_uncertainties[:n_c, :n_c] = b_uncertainties
        b_new_uncertainties[n_c:, n_c:] = np.eye(n_rxns) * sig_G


    # Calculate new covariance matrix
    popt, pcov = unconstrained_least_squares(A_new, b_new, b_new_uncertainties,
                                             p0=phase.molar_fractions)

    # Convert the variance-covariance matrix from endmember amounts to
    # endmember proportions
    p = phase.molar_fractions
    dpdx = (np.eye(n_mbrs) - p).T # same as (1. - p[i] if i == j else -p[i])
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

    if type(formulae[0]) is dict or type(formulae[0]) is Counter:
        stoichiometric_matrix = np.array([[f[e] if e in f else 0.
                                           for e in fitted_elements]
                                          for f in formulae])
    else:
        stoichiometric_matrix = formulae

    b = composition

    if len(compositional_uncertainties.shape) == 1:
        b_uncertainties = np.diag(compositional_uncertainties
                                  * compositional_uncertainties)
    else:
        b_uncertainties = compositional_uncertainties

    # ensure uncertainty matrix is not singular
    det = np.linalg.det(b_uncertainties)
    if det < 1.e-30:
        # warnings.warn(f'The compositional covariance matrix for the {name}
        # solution is nearly singular or not positive-definite
        # (determinant = {det}). '
        #              'This is likely to be because your fitting parameters
        # are not independent. '
        #              'For now, we increase all diagonal components by 1%. '
        #              'However, you may wish to redefine your
        # problem.')

        b_uncertainties += np.diag(np.diag(b_uncertainties))*0.01

    A = stoichiometric_matrix.T

    def endmember_constraints(site_occ, stoic):
        cons = [{'type': 'ineq', 'fun': lambda x, eq=eq: eq.dot(x)}
                for eq in site_occ]
        cons.extend([{'type': 'ineq', 'fun': lambda x, eq=eq: eq.dot(x)}
                     for i, eq in enumerate(stoic) if np.abs(b[i]) < 1.e-10])
        cons.extend([{'type': 'ineq', 'fun': lambda x, eq=eq: -eq.dot(x)}
                     for i, eq in enumerate(stoic) if np.abs(b[i]) < 1.e-10])

        #cons.extend([{'type': 'eq', 'fun': lambda x, eq=eq: eq.dot(x)}
        #             for i, eq in enumerate(stoic) if np.abs(b[i]) < 1.e-10])
        return cons

    cons = endmember_constraints(endmember_site_occupancies.T, A)

    p0 = np.array([0. for i in range(len(A.T))])
    popt, pcov = unconstrained_least_squares(A, b, b_uncertainties, p0=p0)

    res = np.sqrt((A.dot(popt) - b).dot(np.linalg.solve(b_uncertainties,
                                                        A.dot(popt) - b)))

    # Check constraints
    if any([c['fun'](popt) < -1.e-10 for c in cons]):
        warnings.warn('Warning: Simple least squares predicts an unfeasible '
                      'solution composition for {0} solution. '
                      'Recalculating with site constraints. '
                      'The covariance matrix must be '
                      'treated with caution.'.format(name))
        fn = lambda x, A, b, b_uncertainties: np.sqrt((A.dot(x) - b).dot(np.linalg.solve(b_uncertainties, A.dot(x) - b)))

        # Try with default options first, then COBYLA
        sol = minimize(fn, popt,
                       args=(A, b, b_uncertainties),
                       constraints=cons)
        popt = sol.x
        res = sol.fun

        if not sol.success:
            sol = minimize(fn, popt,
                           args=(A, b, b_uncertainties),
                           method='COBYLA',
                           constraints=cons)
            popt = sol.x
            res = sol.fun

        if not sol.success:
            print(sol)
            print(f'popt: {popt}')
            print([c['fun'](popt) for c in cons])
            raise Exception(f'BAD composition for phase {name}')

    if normalize:
        sump = sum(popt)
        popt /= sump
        pcov /= sump*sump
        res /= sump

    popt[np.abs(popt) < 1.e-10] = 0

    if return_Ab:
        return (popt, pcov, res, A, b, b_uncertainties)
    else:
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
            equilibrium_order(phase)
        except:
            print('passing for now')

        popt, pcov, res, A, b, b_uncertainties = sol

        # store original linear problem
        assemblage.stored_compositions[phase_index] = (phase.molar_fractions,
                                                       (A, b, b_uncertainties))

    else:

        # Convert the variance-covariance matrix from endmember amounts to
        # endmember proportions (which must sum to one)
        popt, pcov, res = sol
        sum_popt = sum(popt)
        dpdx = (np.eye(n_mbrs) - popt/sum_popt).T # same as (1. - p[i] if i == j else -p[i])
        Cov_p = dpdx.dot(pcov).dot(dpdx.T)

        if False in (Cov_p < 1.):
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
                            f'{Cov_p}')

        assemblage.stored_compositions[phase_index] = (phase.molar_fractions,
                                                       Cov_p)

        # Optionally print some information to stdout
        if verbose:
            print(phase.name)
            for i in range(n_mbrs):
                print('{0}: {1:.3f} +/- {2:.3f}'.format(phase.endmember_names[i],
                                                        popt[i],
                                                        np.sqrt(Cov_p[i][i])))


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
        n_mbrs = reaction_matrix.shape[0]
        Cov_mu = np.zeros((n_mbrs, n_mbrs))
        dmudPT = np.zeros((n_mbrs, 2))
        mu = np.zeros(n_mbrs)

        i = 0
        for iph, phase in enumerate(assemblage.phases):
            if isinstance(phase, SolidSolution):
                p_mbrs = phase.n_endmembers
                Cov_mu[i:i+p_mbrs, i:i+p_mbrs] = (phase.gibbs_hessian).dot(phase.molar_fraction_covariances).dot(phase.gibbs_hessian.T) # the hessian is symmetric, so transpose only taken for legibility...
                dmudPT[i:i+p_mbrs] = np.array([phase.partial_volumes, -phase.partial_entropies]).T
                mu[i:i+n_mbrs] = phase.partial_gibbs
                i += n_mbrs
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
