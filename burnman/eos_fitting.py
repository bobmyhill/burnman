# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2019 by the BurnMan team, released under the GNU
# GPL v2 or later.

from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from .tools import flatten
from scipy.odr import RealData, Model, ODR


class EOSModel(object):
    def __init__(self, mineral, fit_params, data, data_covariances, flags):
        self.mineral = mineral
        self.fit_params = fit_params
        self.data = data
        self.data_covariances = data_covariances
        self.flags = flags
        self.starting_guesses = self.get_params()
        self.dof = len(self.data) - len(self.starting_guesses)
        self.model_function = Model(self.fitting_function, implicit=1)
        self.model_data = RealData(data.T, 1, covx=data_covariances.T)
        self.odr_model = ODR(self.model_data, self.model_function, beta0=self.starting_guesses)

    def set_params(self, param_values):
        i=0
        for param in self.fit_params:
            if isinstance(self.mineral.params[param], float):
                self.mineral.params[param] = param_values[i]
                i += 1
            else:
                for j in range(len(self.mineral.params[param])):
                    self.mineral.params[param][j] = param_values[i]
                    i += 1

    def get_params(self):
        return np.array(flatten([self.mineral.params[prm] for prm in self.fit_params]))

    def fitting_function(self, params, x):
        self.set_params(params)
        pressures, temperatures, properties = x
        n_data = len(pressures)
        set_flags = list(set(self.flags))
        model_properties = np.empty_like(properties)
        for uflag in set_flags:
            mask = [i for i in range(n_data) if self.flags[i] == uflag]
            model_properties[mask] = self.mineral.evaluate([uflag], pressures[mask], temperatures[mask])[0]
        return properties - model_properties

    def run(self):
        self.output = self.odr_model.run()

        # Compute or alias some attributes
        self.delta = self.output.delta.T
        self.data_mle = self.output.xplus.T
        invCdelta = np.linalg.solve(self.data_covariances, self.delta)
        weighted_square_residuals = np.einsum('ij, ij -> i', self.delta, invCdelta)
        self.weighted_residuals = np.sqrt(weighted_square_residuals)
        self.WSS = self.output.sum_square_delta  # equivalent to summing weighted_square_residuals
        self.popt = self.output.beta
        self.pcov = self.output.cov_beta * self.output.res_var
        self.goodness_of_fit = self.output.res_var  # equivalent to WSS/dof
        return self.output


def fit_PTp_data(mineral, fit_params, flags, data, data_covariances=None, verbose=True):
    """
    Given a mineral of any type, a list of fit parameters
    and a set of P-T-property points and (optional) uncertainties,
    this function returns a list of optimized parameters
    and their associated covariances, fitted using the
    scipy.optimize.curve_fit routine.

    Parameters
    ----------
    mineral : mineral
        Mineral for which the parameters should be optimized.

    fit_params : list of strings
        List of dictionary keys contained in mineral.params
        corresponding to the variables to be optimized
        during fitting. Initial guesses are taken from the existing
        values for the parameters

    flags : string or list of strings
        Attribute names for the property to be fit for the whole
        dataset or each datum individually (e.g. 'V')

    data : numpy array of observed P-T-property values

    data_covariances : numpy array of P-T-property covariances (optional)
        If not given, all covariance matrices are chosen
        such that C00 = 1, otherwise Cij = 0
        In other words, all data points have equal weight,
        with all error in the pressure

    Returns
    -------
    model : instance of fitted model
        Fitting-related attributes are as follows:
            dof : integer
                Degrees of freedom of the system
            data_mle : 2D numpy array
                Maximum likelihood estimates of the observed data points
                on the best-fit curve
            weighted_residuals : numpy array
                Weighted residuals
            weights : numpy array
                1/(data variances normal to the best fit curve)
            WSS : float
                Weighted sum of squares residuals
            popt : numpy array
                Optimized parameters
            pcov : 2D numpy array
                Covariance matrix of optimized parameters
            noise_variance : float
                Estimate of the variance of the data normal to the curve
    """

    # If only one property flag is given, assume it applies to all data
    if type(flags) is str:
        flags = np.array([flags] * len(data[:,0]))

    if data_covariances is None:
        raise Exception('You must specify data_covariances')
    elif len(data_covariances.shape) == 2:
        data_covariances = np.array([[[s[0], 0, 0],
                                      [0, s[1], 0],
                                      [0, 0, s[2]]]
                                      for s in data_covariances])

    model = EOSModel(mineral, fit_params, data, data_covariances, flags)
    model.run()

    if verbose == True:
        confidence_interval = 0.9
        confidence_bound, indices, probabilities = nonlinear_fitting.extreme_values(model.weighted_residuals, confidence_interval)
        if indices != []:
            print('The function nonlinear_fitting.extreme_values(model.weighted_residuals, confidence_interval) '
                  'has determined that there are {0:d} data points which have residuals which are not '
                  'expected at the {1:.1f}% confidence level (> {2:.1f} s.d. away from the model fit).\n'
                  'Their indices and the probabilities of finding such extreme values are:'.format(len(indices), confidence_interval*100., confidence_bound))
            for i, idx in enumerate(indices):
                print('[{0:d}]: {1:.4f} ({2:.1f} s.d. from the model)'.format(idx, probabilities[i], np.abs(model.weighted_residuals[idx])))
            print('You might consider removing them from your fit, '
                  'or increasing the uncertainties in their measured values.\n')

    return model


def fit_PTV_data(mineral, fit_params, data, data_covariances=None, verbose=True):
    """
    A simple alias for the fit_PTp_data for when all the data is volume data
    """
    return fit_PTp_data(mineral=mineral, flags='V',
                        data=data, data_covariances=data_covariances,
                        fit_params=fit_params, verbose=verbose)
