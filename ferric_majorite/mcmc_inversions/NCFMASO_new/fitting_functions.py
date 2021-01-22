from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))
import burnman
from burnman.processanalyses import equilibrate_phase
from burnman.processanalyses import assemblage_affinity_misfit
from global_constants import transition_chisqr, transition_chi

class Storage(dict):
    def __init__(self, dictoflists):
        self.update(dictoflists)
        # Make dictionaries
        self['dict_endmember_args'] = {a[0]: {} for a
                                       in self['endmember_args']}
        for a in self['endmember_args']:
            self['dict_endmember_args'][a[0]][a[1]] = a[2]

        self['dict_solution_args'] = {a[0]: {} for a in self['solution_args']}
        for a in self['solution_args']:
            self['dict_solution_args'][a[0]]['{0}{1}{2}'.format(a[1],
                                                                a[2],
                                                                a[3])] = a[4]

        self['dict_experiment_uncertainties'] = {}
        for u in self['experiment_uncertainties']:
            self['dict_experiment_uncertainties'][u[0]] = {'P': 0., 'T': 0.}

        for u in self['experiment_uncertainties']:
            self['dict_experiment_uncertainties'][u[0]][u[1]] = u[2]


def param_list_to_string(params):
    string = '['
    for i, p in enumerate(params):
        string += '{0:.4f}'.format(p)
        if i < len(params) - 1:
            string += ', '
    string += ']'
    return string


def get_params(storage):
    """
    This function gets the parameters from the stored parameter lists
    """

    # Endmember parameters
    args = [a[2]/a[3] for a in storage['endmember_args']]

    # Solution parameters
    args.extend([a[4]/a[5] for a in storage['solution_args']])

    # Experimental uncertainties
    args.extend([u[2]/u[3] for u in storage['experiment_uncertainties']])
    return args


def set_params(args, dataset, storage, special_constraint_function):
    """
    This function sets the parameters *both* in the parameter lists,
    the parameter dictionaries and also in the minerals / solutions.
    """

    i = 0

    # Endmember parameters
    for j, a in enumerate(storage['endmember_args']):
        storage['dict_endmember_args'][a[0]][a[1]] = args[i]*a[3]
        storage['endmember_args'][j][2] = args[i]*a[3]
        dataset['endmembers'][a[0]].params[a[1]] = args[i]*a[3]
        i += 1

    # Solution parameters
    for j, a in enumerate(storage['solution_args']):
        storage['dict_solution_args'][a[0]][a[1]] = args[i]*a[5]
        storage['solution_args'][j][4] = args[i]*a[5]
        ss = dataset['solutions'][a[0]]
        row = int(a[2])
        col = int(a[3])
        m0 = row
        m1 = row + col + 1
        mod = 2./(ss.solution_model.alphas[m0] + ss.solution_model.alphas[m1])
        val = args[i]*a[5]
        if a[1] == 'E':
            ss.energy_interaction[row][col] = val
            ss.solution_model.We[m0, m1] = val*mod
        elif a[1] == 'S':
            ss.entropy_interaction[row][col] = val
            ss.solution_model.Ws[m0, m1] = val*mod
        elif a[1] == 'V':
            ss.volume_interaction[row][col] = val
            ss.solution_model.Wv[m0, m1] = val*mod
        else:
            raise Exception('Not implemented')
        i += 1

    # Reinitialize solutions to fill We, Ws, Wv
    # This is quite expensive if we only need to modify a few values
    # It is now redundant, as we replace the values directly in the code above
    # for name in dataset['solutions']:
    #     dataset['solutions'][name].set_model()
    #     # The old method completely reinitialized the solution
    #     # burnman.SolidSolution.__init__(dataset['solutions'][name])

    # Experimental uncertainties
    for j, u in enumerate(storage['experiment_uncertainties']):
        storage['dict_experiment_uncertainties'][u[0]][u[1]] = args[i]*u[3]
        storage['experiment_uncertainties'][j][2] = args[i]*u[3]
        i += 1

    # Special one-off constraints
    if special_constraint_function is not None:
        special_constraint_function(dataset, storage)
    return None


def minimize_func(params, dataset, storage, special_constraint_function):
    # Set parameters
    set_params(params, dataset, storage,
               special_constraint_function=special_constraint_function)

    chisqr = []
    # Run through all assemblages for affinity misfit
    # This is what takes most of the time
    for i, assemblage in enumerate(dataset['assemblages']):
        # print(i, assemblage.experiment_id,
        # [phase.name for phase in assemblage.phases])

        # Assign a state to the assemblage
        P, T = np.array(assemblage.nominal_state)
        expt_id = assemblage.experiment_id
        try:
            P += storage['dict_experiment_uncertainties'][expt_id]['P']
            T += storage['dict_experiment_uncertainties'][expt_id]['T']
        except KeyError:
            pass

        assemblage.set_state(P, T)

        # Assign compositions and uncertainties to solid solutions
        for j, phase in enumerate(assemblage.phases):
            if isinstance(phase, burnman.SolidSolution):
                phase.set_composition(assemblage.stored_compositions[j][0])
                cov = assemblage.stored_compositions[j][1]
                phase.molar_fraction_covariances = cov

                if phase.ordered:
                    equilibrate_phase(assemblage, j)

        # Calculate the misfit and store it
        assemblage.chisqr = assemblage_affinity_misfit(assemblage)
        # chisqr.append(assemblage.chisqr)

        # Modify the chisqr because experimental chisqr
        # distribution is long-tailed
        if assemblage.chisqr < transition_chisqr:
            chisqr.append(assemblage.chisqr)
        else:
            chisqr.append(-transition_chisqr
                          + 2. * transition_chi * np.sqrt(assemblage.chisqr))

    # Endmember priors
    for p in storage['endmember_priors']:
        c = np.power(((storage['dict_endmember_args'][p[0]][p[1]]
                       - p[2])/p[3]), 2.)
        # print('endmember_prior', p[0], p[1], dict_endmember_args[p[0]][p[1]],
        # p[2], p[3], c)
        chisqr.append(c)

    # Solution priors
    for p in storage['solution_priors']:
        key = '{0}{1}{2}'.format(p[1], p[2], p[3])
        c = np.power(((storage['dict_solution_args'][p[0]][key]
                       - p[4])/p[5]), 2.)
        # print('solution_prior', c)
        chisqr.append(c)

    # Experiment uncertainties
    for u in storage['experiment_uncertainties']:
        c = np.power(u[2]/u[3], 2.)
        # print('pressure uncertainty', c)
        chisqr.append(c)

    # calculate the squared misfit.
    # this is an approximation to the negative log probability
    # see http://www.physics.utah.edu/~detar/phys6720/handouts/
    # curve_fit/curve_fit/node2.html
    half_sqr_misfit = np.sum(chisqr) / 2.

    print_params = False
    if print_params:
        print(param_list_to_string(params))

    if np.isnan(half_sqr_misfit) or not np.isfinite(half_sqr_misfit):
        return np.inf  # catch for if one or more EoSes fail
    else:
        return half_sqr_misfit


def log_probability(params, dataset, storage, special_constraint_function):
    return -minimize_func(params,
                          dataset,
                          storage,
                          special_constraint_function)