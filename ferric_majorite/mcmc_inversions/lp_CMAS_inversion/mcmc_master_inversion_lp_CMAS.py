from __future__ import absolute_import
from __future__ import print_function

import platform
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from input_dataset import create_minerals
from fitting_functions import Storage, log_probability
from fitting_functions import get_params, set_params
from fitting_functions import set_dataset_params_from_storage
from output_plots import chain_plotter, plots
import pickle
import emcee
from multiprocessing import Pool

import pandas as pd
import seaborn as sns

print('TODO!! Check endmembers and priors, add solution parameters and priors')
if len(sys.argv) == 2:
    if sys.argv[1] == '--fit':
        run_inversion = True
        print('Running inversion')
    else:
        run_inversion = False
        print('Not running inversion. Use --fit as command line'
              ' argument to invert parameters')
else:
    run_inversion = False
    print('Not running inversion. Use --fit as command line argument'
          ' to invert parameters')


# Component defining endmembers (for H_0 and S_0) are:
# Fe: Fe metal (BCC, FCC, HCP)
# O: O2
# Mg: MgO per
# Si: SiO2 qtz
# Al: Mg3Al2Si3O12 pyrope
# Ca: CaMgSi2O6 diopside
# Na: NaAlSi2O6 jadeite


mineral_dataset = create_minerals()
endmembers = mineral_dataset['endmembers']
solutions = mineral_dataset['solutions']
child_solutions = mineral_dataset['child_solutions']

endmember_args = [[mbr, 'H_0', endmembers[mbr].params['H_0'], 1.e3]
                  for mbr in ['sp', 'gr', 'cen', 'cats',
                              'oen', 'mgts']]

endmember_args.extend([[mbr, 'S_0', endmembers[mbr].params['S_0'], 1.]
                       for mbr in ['sp', 'gr', 'di', # 'cen', 'cats',
                                   'oen', 'mgts']])

endmember_priors = [[mbr, 'S_0', endmembers[mbr].params['S_0_orig'][0],
                     endmembers[mbr].params['S_0_orig'][1]]
                    for mbr in ['gr',
                                'di',
                                'oen',
                                'sp']]


solution_args = [['opx', 'E', 0, 1,
                  solutions['opx'].energy_interaction[0][1], 1.e3],  # oen-mgts
                 ['opx', 'E', 0, 2,
                  solutions['opx'].energy_interaction[0][2], 1.e3],  # oen-odi
                 ['opx', 'E', 2, 0,
                  solutions['opx'].energy_interaction[1][0], 1.e3],  # mgts-odi
                 ['cpx', 'E', 0, 1,
                  solutions['cpx'].energy_interaction[0][1], 1.e3],  # di-cen
                 ['cpx', 'E', 0, 3,
                  solutions['cpx'].energy_interaction[0][3], 1.e3],  # di-cats
                 ['cpx', 'E', 2, 1,
                  solutions['cpx'].energy_interaction[2][1], 1.e3],  # cen-cats
                 ['gt', 'E', 0, 1,
                  solutions['gt'].energy_interaction[0][1], 1.e3]]  # py-gr

solution_priors = [['opx', 'E', 0, 1, 12.5e3, 2.e3],  # oen-mgts
                   ['opx', 'E', 0, 2, 32.2e3, 5.e3],  # oen-odi
                   ['opx', 'E', 2, 0, 75.5e3, 30.e3],  # mgts-odi
                   ['gt', 'E', 0, 1, 30.e3, 0.1e3]]  # py-gr

# Some fairly lax priors on cpx solution parameters
for (i, j) in [(0, 1),
               (0, 3),
               (2, 1)]:
    solution_priors.append(['cpx', 'E', i, j,
                            solutions['cpx'].energy_interaction[i][j],
                            2.e3 + 0.3*solutions['cpx'].energy_interaction[i][j]])

# Uncertainties from Frost data
experiment_uncertainties = []

def special_constraints(dataset, storage):

    endmembers = dataset['endmembers']
    solutions = dataset['solutions']

    # 1) Destabilise fwd
    endmembers['fa'].set_state(6.25e9, 1673.15)
    endmembers['frw'].set_state(6.25e9, 1673.15)
    endmembers['fwd'].set_state(6.25e9, 1673.15)

    # First, determine the entropy which will give the fa-fwd reaction
    # the same slope as the fa-frw reaction
    dPdT = (endmembers['frw'].S
            - endmembers['fa'].S)/(endmembers['frw'].V
                                   - endmembers['fa'].V)  # = dS/dV

    dV = endmembers['fwd'].V - endmembers['fa'].V
    dS = dPdT*dV
    endmembers['fwd'].params['S_0'] += (endmembers['fa'].S
                                        - endmembers['fwd'].S + dS)
    endmembers['fwd'].params['H_0'] += (endmembers['frw'].gibbs
                                        - endmembers['fwd'].gibbs
                                        + 100.)  # fwd less stable than frw

    # 2) Fix odi (just in case we decide to fit di H, S or V at some point)
    endmembers['odi'].params['H_0'] = endmembers['di'].params['H_0'] - 0.1e3
    endmembers['odi'].params['S_0'] = endmembers['di'].params['S_0'] - 0.211
    endmembers['odi'].params['V_0'] = endmembers['di'].params['V_0'] + 0.005e-5

    # 3) Make sure the temperature dependence of ordering
    # is preserved in Mg-Fe opx and hpx
    wMgFe = solutions['opx'].energy_interaction[0][0] / 4. + 2.25e3
    Etweak = solutions['opx'].energy_interaction[0][0] / 4. - 8.35e3

    solutions['opx'].energy_interaction[0][-1] = wMgFe  # oen-ofm
    solutions['opx'].energy_interaction[1][-1] = wMgFe  # ofs-ofm
    endmembers['ofm'].property_modifiers[0][1]['delta_E'] = Etweak
    endmembers['hfm'].property_modifiers[0][1]['delta_E'] = Etweak

    # 4) Copy interaction parameters from opx to hpx:
    solutions['hpx'].alphas = solutions['opx'].alphas
    solutions['hpx'].energy_interaction = solutions['opx'].energy_interaction
    solutions['hpx'].entropy_interaction = solutions['opx'].entropy_interaction
    solutions['hpx'].volume_interaction = solutions['opx'].volume_interaction


# Create storage object
storage = Storage({'endmember_args': endmember_args,
                   'solution_args': solution_args,
                   'endmember_priors': endmember_priors,
                   'solution_priors': solution_priors,
                   'experiment_uncertainties': experiment_uncertainties})


# Create labels for each parameter
labels = [a[0]+'_'+a[1] for a in endmember_args]
labels.extend(['{0}_{1}[{2},{3}]'.format(a[0], a[1], a[2], a[3])
               for a in solution_args])
labels.extend(['{0}_{1}'.format(a[0], a[1]) for a in experiment_uncertainties])


#######################
# EXPERIMENTAL DATA ###
#######################

from datasets import endmember_reactions

# CMS
from datasets import Carlson_Lindsley_1988_CMS_opx_cpx

# MAS
from datasets import Gasparik_Newton_1984_MAS_opx_sp_fo
from datasets import Gasparik_Newton_1984_MAS_py_opx_sp_fo
from datasets import Perkins_et_al_1981_MAS_py_opx
#from datasets import Liu_et_al_2016_gt_bdg_cor
#from datasets import Liu_et_al_2017_bdg_cor
#from datasets import Hirose_et_al_2001_ilm_bdg_gt

# CMAS
from datasets import Perkins_Newton_1980_CMAS_opx_cpx_gt
from datasets import Klemme_ONeill_2000_CMAS_opx_cpx_gt_ol_sp


assemblages = [assemblage for assemblage_list in
               [module.get_assemblages(mineral_dataset)
                for module in [endmember_reactions,  # 73, 2713
                               Gasparik_Newton_1984_MAS_opx_sp_fo,  # 14 634
                               Gasparik_Newton_1984_MAS_py_opx_sp_fo,  # 2, 230
                               Perkins_et_al_1981_MAS_py_opx,  # 91, 553
                               Carlson_Lindsley_1988_CMS_opx_cpx,  # 40, 4644
                               Perkins_Newton_1980_CMAS_opx_cpx_gt,  # 12, 894
                               Klemme_ONeill_2000_CMAS_opx_cpx_gt_ol_sp  # 14 15588
                               ]]
               for assemblage in assemblage_list]

dataset = {'endmembers': mineral_dataset['endmembers'],
           'solutions': mineral_dataset['solutions'],
           'child_solutions': mineral_dataset['child_solutions'],
           'assemblages': assemblages}


# Initialize parameters and pepare internal arrays
# This should speed things up after depickling
def initialise_params():
    from import_params import FMSO_storage, transfer_storage

    print('Setting parameters from FMSO output... but not adding them to inversion')
    set_dataset_params_from_storage(dataset, FMSO_storage,
                                    special_constraints)

    lnprob = log_probability(get_params(storage), dataset, storage,
                             special_constraints)
    print('Initial ln(p) = {0}'.format(lnprob))
    return None


print('Fitting {0} assemblages'.format(len(dataset['assemblages'])))
initialise_params()

########################
# RUN THE MINIMIZATION #
########################
if run_inversion:
    # Make sure we always get the same walker starting points
    # (good for bug checking)
    np.random.seed(1234)

    jiggle_x0 = 1.e-3
    walker_multiplication_factor = 4  # this number must be greater than 2!
    n_steps_burn_in = 0  # number of steps in the burn in period (not used)
    n_steps_mcmc = 6000  # number of steps in the full mcmc run
    n_discard = 0  # discard this number of steps from the full mcmc run
    thin = 1  # thin by this factor when calling get_chain

    x0 = get_params(storage)
    ndim = len(x0)
    nwalkers = ndim*walker_multiplication_factor

    thisfilename = os.path.basename(__file__)
    base = os.path.splitext(thisfilename)[0]

    burnfile = base+'_sampler_after_burn_in.pickle'
    mcmcfile = base+'_sampler_after_mcmc_run.pickle'

    print('Running MCMC inversion with {0} parameters and {1} walkers'.format(ndim, nwalkers))
    print('This inversion will involve {0} burn-in steps and {1} stored steps'.format(n_steps_burn_in, n_steps_mcmc))
    print('The walkers will be clustered around a start point with a random jiggle of {0}'.format(jiggle_x0))
    print('The samplers will be saved to the following two pickle files:')
    print(burnfile)
    print(mcmcfile)

    print('using n threads')

    p0 = x0 + jiggle_x0*np.random.randn(nwalkers, ndim)

    if platform.node() == "axolotl.gly.bris.ac.uk": #"ix.gly.bris.ac.uk":
        print('This computer is ix.gly.bris.ac.uk. Leaving two cpus free.')
        processes = os.cpu_count()-2  # NT suggests keeping two cpus free
    else:
        print('This computer is not ix.gly.bris.ac.uk. Using all cpus.')
        processes = None  # default is to use all of the cpus

    with Pool(processes=processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args=[dataset, storage,
                                              special_constraints],
                                        pool=pool)

        if n_steps_burn_in > 0:
            print('Starting burn-in')
            state = sampler.run_mcmc(p0, n_steps_burn_in, progress=True)
            pickle.dump(sampler, open(burnfile,'wb'))

            sampler.reset()
        else:
            state = p0

        sampler = pickle.load(open(mcmcfile+'int','rb'))
        sampler.pool = pool  # deal with change in pool!
        state = sampler.run_mcmc(sampler._previous_state, n_steps_mcmc, progress=True)

        #print('Burn-in complete. Starting MCMC run.')
        #state = sampler.run_mcmc(state, n_steps_mcmc, progress=True)

        print('100% complete. Pickling')
        pickle.dump(sampler, open(mcmcfile,'wb'))

    print('Chain shape: {0}'.format(sampler.get_chain().shape))
    print('Mean acceptance fraction: {0:.2f}'
          ' (should ideally be between 0.25 and 0.5)'.format(np.mean(sampler.acceptance_fraction)))

    if np.mean(sampler.acceptance_fraction) < 0.15:
        print(sampler.get_chain().shape)
        print(sampler.acceptance_fraction)
        exit()

    try:
        tau = sampler.get_autocorr_time()
        print(tau)
    except emcee.autocorr.AutocorrError as e:
        print(e)

    plot_chains = True
    if plot_chains:
        chain_plotter(sampler, labels)

    flat_samples = sampler.get_chain(discard=1600, thin=thin, flat=True)
    #flat_samples = sampler.get_chain(discard=n_discard, thin=thin, flat=True)

    # use the 50th percentile for the preferred params
    # (might not represent the best fit
    # if the distribution is strongly non-Gaussian...)
    mcmc_params = np.array([np.percentile(flat_samples[:, i], [50])[0]
                            for i in range(ndim)])
    mcmc_unc = np.array([np.percentile(flat_samples[:, i], [16, 50, 84])
                         for i in range(ndim)])

    for i in range(ndim):
        mcmc_i = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc_i)
        txt = "\\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc_i[1], q[0], q[1], labels[i])
        print(txt)

    Mcorr = np.corrcoef(flat_samples.T)
    triu_Mcorr = np.triu(m=Mcorr, k=1)
    abs_Mcorr_sorted_indices = np.unravel_index(np.argsort(np.abs(triu_Mcorr),
                                                           axis=None)[::-1],
                                                triu_Mcorr.shape)
    Mcov = np.cov(flat_samples.T) # each row corresponds to a different variable
    #import corner
    #fig = corner.corner(flat_samples, labels=labels);
    #fig.savefig('corner_plot.pdf')
    #plt.show()

    fig, ax = plt.subplots(figsize=(50,50))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    ax = sns.heatmap(pd.DataFrame(Mcorr),
                     xticklabels=labels,
                     yticklabels=labels,
                     cmap=cmap,
                     vmin=-1.,
                     center=0.,
                     vmax=1.)
    fig.savefig('parameter_correlations.pdf')
    plt.show()

    set_params(mcmc_params, dataset, storage, special_constraints)
    # print(minimize(minimize_func, get_params(),
    # args=(assemblages), method='BFGS')) # , options={'eps': 1.e-02}))

# Print the current parameters
print(storage)

# Make plots
plots(dataset, storage)
