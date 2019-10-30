from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from load_inversion import load_inversion
from fitting_functions import get_params
from output_plots import chain_plotter, plots
from iron_saturated_plots import iron_saturated_KLB_plot, iron_saturated_MORB_plot
from buffered_plots import buffered_KLB_plot, buffered_MORB_plot

mcmc_params, dataset, storage, special_constraints = load_inversion()

# Print the current parameters
print(get_params(storage))

# Make plots
# plots(dataset, storage)
#iron_saturated_KLB_plot(dataset, storage)
iron_saturated_MORB_plot(dataset, storage)
#buffered_KLB_plot(dataset, storage, buffer='EMOD', n_log_units=0.)
#buffered_KLB_plot(dataset, storage, buffer='Re-ReO2', n_log_units=0.)
#buffered_MORB_plot(dataset, storage, buffer='Re-ReO2', n_log_units=0.)
