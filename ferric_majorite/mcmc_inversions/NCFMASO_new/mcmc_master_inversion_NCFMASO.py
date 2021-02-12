from __future__ import absolute_import
from __future__ import print_function

import platform
import os
# The following line is recommended by emcee documentation
# to turn off parallelization by numpy.
# I suspect this was the cause of high load on ix.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import numpy as np
np.__config__.show()
import matplotlib.pyplot as plt
import emcee
from multiprocessing import Pool
import pandas as pd
import seaborn as sns

from fitting_functions import get_params, set_params
from fitting_functions import minimize_func, log_probability
from create_dataset import create_dataset, special_constraints
from output_plots import chain_plotter, plots
from datetime import datetime


print(f'Start time: {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')

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

dataset, storage, labels = create_dataset()

# Using this probability function with params as the only
# argument means that we only pickle dataset, storage and
# special_constraint_function once. Because these objects
# are quite complex, this should be faster than passing
# args to the EnsembleSampler.
#def log_probability_global(params):
#    return -minimize_func(params,
#                          dataset,
#                          storage,
#                          special_constraints)

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
    n_steps_mcmc = 20000  # number of steps in the full mcmc run
    n_discard = 0  # discard this number of steps from the full mcmc run
    thin = 1  # thin by this factor when calling get_chain

    x0 = get_params(storage)
    ndim = len(x0)
    nwalkers = ndim*walker_multiplication_factor

    thisfilename = os.path.basename(__file__)
    base = os.path.splitext(thisfilename)[0]
    hdffile = base+'_sampler_after_mcmc_run.hdf5'

    print(f'Running MCMC inversion with {ndim} parameters '
          f'and {nwalkers} walkers')
    print(f'This inversion will involve {n_steps_burn_in} burn-in steps '
          f'and {n_steps_mcmc} stored steps')
    print('The walkers will be clustered around a start point '
          f'with a random jiggle of {jiggle_x0}')
    print('The samplers will be saved to the following hdf file:')
    print(hdffile)

    print('using n threads')

    p0 = x0 + jiggle_x0*np.random.randn(nwalkers, ndim)

    if platform.node() == "ix.gly.bris.ac.uk":  # "ix.gly.bris.ac.uk":
        print('This computer is ix.gly.bris.ac.uk. Leaving two cpus free.')
        processes = os.cpu_count()-2  # NT suggests keeping two cpus free
    else:
        print('This computer is not ix.gly.bris.ac.uk. Using all cpus.')
        processes = None  # default is to use all of the cpus

    with Pool(processes=processes) as pool:
        new_outfile = False
        if new_outfile:
            backend = emcee.backends.HDFBackend(hdffile+'.save')
        else:
            backend = emcee.backends.HDFBackend(hdffile)
        #sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_global,
        #                                pool=pool, backend=backend)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args=[dataset, storage,
                                              special_constraints],
                                        pool=pool, backend=backend)

        new_inversion = False
        if new_inversion:
            # only reset backend if you want to start a new inversion!
            backend.reset(nwalkers, ndim)
            p0 = x0 + jiggle_x0*np.random.randn(nwalkers, ndim)
        else:
            p0 = sampler._previous_state

        if new_outfile:
            backend = emcee.backends.HDFBackend(hdffile)
            backend.reset(nwalkers, ndim)
            #sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_global,
            #                                pool=pool, backend=backend)

            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                            args=[dataset, storage,
                                                  special_constraints],
                                            pool=pool, backend=backend)

        state = sampler.run_mcmc(p0, n_steps_mcmc,
                                 progress=True)
        print('100% complete.')

    print('Chain shape: {0}'.format(sampler.get_chain().shape))
    mean_acceptance_fraction = np.mean(sampler.acceptance_fraction)
    print(f'Mean acceptance fraction: {mean_acceptance_fraction:.2f}'
          ' (should ideally be between 0.25 and 0.5)')

    if mean_acceptance_fraction < 0.15:
        print(sampler.get_chain().shape)
        print(sampler.acceptance_fraction)
        exit()

    try:
        tau = sampler.get_autocorr_time()
        print(tau)
    except emcee.autocorr.AutocorrError as e:
        print(e)

    plot_chains = False
    if plot_chains:
        chain_plotter(sampler, labels)

    flat_samples = sampler.get_chain(discard=1600, thin=thin, flat=True)

    # use the 50th percentile as an estimate for each param
    # (this might not represent the best fit
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
    Mcov = np.cov(flat_samples.T)  # rows corresponds to variables
    # import corner
    # fig = corner.corner(flat_samples, labels=labels);
    # fig.savefig('corner_plot.pdf')
    # plt.show()

    fig, ax = plt.subplots(figsize=(50, 50))
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

# Print the current parameters
print(get_params(storage))

# Make plots
# plots(dataset, storage)

print(f'End time: {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
