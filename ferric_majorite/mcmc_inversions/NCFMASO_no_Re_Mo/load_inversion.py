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
from output_plots import chain_plotter, plots
import emcee
from multiprocessing import Pool

import pandas as pd
import seaborn as sns

from create_dataset import create_dataset

dataset, storage, labels, special_constraints = create_dataset()

def load_inversion(show_correlations=True):
    ########################
    # READ THE INVERSION #
    ########################
    thisfilename = os.path.basename(__file__)
    base = os.path.splitext(thisfilename)[0]
    hdffile='mcmc_master_inversion_NCFMASO_sampler_after_mcmc_run.hdf5'
    reader=emcee.backends.HDFBackend(hdffile, read_only=True)


    print('Chain shape: {0}'.format(reader.get_chain().shape))
    n_samples, n_walkers, n_params = reader.get_chain().shape
    try:
        tau = reader.get_autocorr_time()
        print(tau)
    except emcee.autocorr.AutocorrError as e:
        print(e)

    plot_chains = False
    if plot_chains:
        chain_plotter(reader, labels)


    thin = 1  # thin by this factor when calling get_chain
    n_used_samples = 100 # integer value. Multiply by number of walkers to get the length of the flattened chain
    discard = n_samples - n_used_samples
    flat_samples = reader.get_chain(discard=discard, thin=thin, flat=True)
    #flat_samples = reader.get_chain(discard=n_discard, thin=thin, flat=True)

    # use the 50th percentile for the preferred params
    # (might not represent the best fit
    # if the distribution is strongly non-Gaussian...)
    mcmc_params = np.array([np.percentile(flat_samples[:, i], [50])[0]
                            for i in range(n_params)])
    mcmc_unc = np.array([np.percentile(flat_samples[:, i], [16, 50, 84])
                         for i in range(n_params)])

    for i in range(n_params):
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

    if show_correlations:
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

    return mcmc_params, dataset, storage, special_constraints
