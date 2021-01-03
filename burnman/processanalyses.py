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


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


def fit_composition(fitted_elements, composition, compositional_uncertainties,
                    formulae, endmember_site_occupancies,
                    normalize=True, name=None):
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
    if np.linalg.det(b_uncertainties) < 1.e-30:
        # warnings.warn('The compositional covariance matrix for the {0}
        # solution is nearly singular or not positive-definite
        # (determinant = {1}). '
        #              'This is likely to be because your fitting parameters
        # are not independent. '
        #              'For now, we increase all diagonal components by 1%. '
        #              'However, you may wish to redefine your
        # problem.'.format(name, np.linalg.det(b_uncertainties)))

        b_uncertainties += np.diag(np.diag(b_uncertainties))*0.01

    A = stoichiometric_matrix.T

    def endmember_constraints(site_occ, stoic):
        cons = [{'type': 'ineq', 'fun': lambda x, eq=eq: eq.dot(x)}
                for eq in site_occ]
        cons.extend([{'type': 'ineq', 'fun': lambda x, eq=eq: eq.dot(x)}
                     for i, eq in enumerate(stoic) if np.abs(b[i]) < 1.e-10])
        cons.extend([{'type': 'ineq', 'fun': lambda x, eq=eq: -eq.dot(x)}
                     for i, eq in enumerate(stoic) if np.abs(b[i]) < 1.e-10])
        return cons

    cons = endmember_constraints(endmember_site_occupancies.T, A)


    fn = lambda A, *proportions: A.dot(proportions)
    popt, pcov = curve_fit(fn, A, b,
                           p0=np.array([0. for i in range(len(A.T))]),
                           sigma=b_uncertainties, absolute_sigma=True)

    res = np.sqrt((A.dot(popt) - b).dot(np.linalg.solve(b_uncertainties,
                                                        A.dot(popt) - b)))

    # Check constraints
    if any([c['fun'](popt)<0. for c in cons]):
        warnings.warn('Warning: Simple least squares predicts an unfeasible '
                      'solution composition for {0} solution. '
                      'Recalculating with site constraints. '
                      'The covariance matrix must be '
                      'treated with caution.'.format(name))
        fn = lambda x, A, b, b_uncertainties: np.sqrt((A.dot(popt) - b).dot(np.linalg.solve(b_uncertainties, A.dot(popt) - b)))
        sol = minimize(fn, popt,
                       args=(A, b, b_uncertainties),
                       method='COBYLA',
                       constraints=cons)
        popt = sol.x
        res = sol.fun
        if not sol.success:
            raise Exception("BAD composition")

    if normalize:
        sump = sum(popt)
        popt /= sump
        pcov /= sump*sump
        res /= sump

    popt[np.abs(popt) < 1.e-10] = 0
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
    popt, pcov, res = fit_composition(fitted_elements, composition,
                                      compositional_uncertainties,
                                      formulae, occupancies,
                                      normalize=True, name=name)

    # Convert uncertainties in endmember amounts
    # into uncertainties in endmember proportions (which must sum to one)
    sum_popt = sum(popt)
    dpdx = np.zeros((n_mbrs, n_mbrs))
    for i in range(n_mbrs):
        for j in range(n_mbrs):
            dpdx[i, j] = (1. - popt[i]/sum_popt
                          if i == j else -popt[i]/sum_popt)
    Cov_p = dpdx.dot(pcov).dot(dpdx.T)

    # Set the active composition of the phase
    # with the best fit composition, and assign the covariance matrix to
    # an attribute of the phase
    phase.set_composition((1. - midpoint_proportion) * popt
                          + midpoint_proportion * phase.polytope_midpoint)

    phase.molar_fraction_covariances = Cov_p

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

    n_phases = len(assemblage.phases)
    assemblage.stored_compositions = [['composition not assigned']
                                      for k in range(n_phases)]

    for k in range(n_phases):
        if isinstance(assemblage.phases[k], SolidSolution):
            if False in (assemblage.phases[k].molar_fraction_covariances < 1.):
                raise Exception(f'Oh No!\n{assemblage.phases[k].name}\n'
                                f'{assemblage.phase_storage[k].fitted_elements}\n'
                                f'{assemblage.phase_storage[k].composition}\n'
                                f'{assemblage.phases[k].molar_fractions}\n'
                                f'{assemblage.phases[k].molar_fraction_covariances}')

            assemblage.stored_compositions[k] = (assemblage.phases[k].molar_fractions,
                                                 assemblage.phases[k].molar_fraction_covariances)
            # print(run_id, assemblage.phases[k].name,
            #       assemblage.phases[k].molar_fractions)


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


def assemblage_affinity_misfit(assemblage, reuse_reaction_matrix=True):
    if reuse_reaction_matrix:
        try:
            reaction_matrix = assemblage.stored_reaction_matrix
        except:
            reaction_matrix = assemblage.stoichiometric_matrix(calculate_subspaces=True)[1]
            assemblage.stored_reaction_matrix = reaction_matrix
    else:
        reaction_matrix = assemblage.stoichiometric_matrix(calculate_subspaces=True)[1]


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
