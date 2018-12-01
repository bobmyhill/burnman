from __future__ import absolute_import
# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2017 by the BurnMan team, released under the GNU
# GPL v2 or later.

import numpy as np
import scipy.optimize as opt
from . import equation_of_state as eos
from ..tools import bracket
import warnings
import pkgutil
from scipy.linalg import solve
from scipy.special import binom
from ..constants import gas_constant



class BroshCalphad(eos.EquationOfState):

    """
    Class for the high pressure CALPHAD equation of state by Brosh et al. (2007).
    """

    def volume(self, pressure, temperature, params):
        """
        Returns volume :math:`[m^3]` as a function of pressure :math:`[Pa]`.
        """
        return 0.

    def pressure(self, temperature, volume, params):
        return 0.

    def isothermal_bulk_modulus(self, pressure, temperature, volume, params):
        """
        Returns isothermal bulk modulus :math:`K_T` :math:`[Pa]` as a function of pressure :math:`[Pa]`,
        temperature :math:`[K]` and volume :math:`[m^3]`.
        """
        return 0.

    def adiabatic_bulk_modulus(self, pressure, temperature, volume, params):
        """
        Returns adiabatic bulk modulus :math:`K_s` of the mineral. :math:`[Pa]`.
        """
        return 0.

    def shear_modulus(self, pressure, temperature, volume, params):
        """
        Returns shear modulus :math:`G` of the mineral. :math:`[Pa]`
        """
        return 0.

    def entropy(self, pressure, temperature, volume, params):
        """
        Returns the molar entropy :math:`\mathcal{S}` of the mineral. :math:`[J/K/mol]`
        """
        return 0.
    
    def molar_internal_energy(self, pressure, temperature, volume, params):
        """
        Returns the internal energy :math:`\mathcal{E}` of the mineral. :math:`[J/mol]`
        """
        
        return 0.

    def _gibbs_1bar(self, temperature, params):
        # first, identify which of the piecewise segments we're in
        i = np.argmax(zip(*params['gibbs_coefficients'])[0] > temperature)

        # select the appropriate coefficients
        coeffs = params['gibbs_coefficients'][i][1]
        gibbs = (coeffs[0] + coeffs[1]*temperature + coeffs[2]*temperature*np.log(temperature) +
                 coeffs[3]/temperature + coeffs[4]/(temperature*temperature) +
                 coeffs[5]/(temperature*temperature*temperature) +
                 coeffs[6]*np.power(temperature, -9.) +
                 coeffs[7]*temperature*temperature +
                 coeffs[8]*temperature*temperature*temperature +
                 coeffs[9]*np.power(temperature, 4.) +
                 coeffs[10]*np.power(temperature, 7.) +
                 coeffs[11]*np.sqrt(temperature) +
                 coeffs[12]*np.log(temperature))
        return gibbs
        
    def gibbs_free_energy(self, pressure, temperature, volume, params):
        """
        Returns the Gibbs free energy :math:`\mathcal{G}` of the mineral. :math:`[J/mol]`
        """
    
        X = [1./(1. - params['a'][i-2] +
                 params['a'][i-2]*np.power((1. + i/(3.*params['a'][i-2]) *
                                            pressure/params['K_0']), 1./i))
             for i in range(2, 6)] # eq. A2

        K0b = params['K_0']/(1. + params['delta'][0]) # eq. B1b
        XT = [1./(1. - params['ab'][i-2] +
                  params['ab'][i-2]*np.power((1. + i/(3.*params['ab'][i-2]) *
                                              pressure/K0b), 1./i))
              for i in range(2, 6)] # eq. A2, B1b

        d = lambda k, Xi: -3.*np.log(Xi) if k == 3 else np.power(Xi, 3. - k) * k / (k-3.) # eq. A9b
    
        Gamma = lambda i, Xi: (3.*np.power(params['a'][i-2], 1. - i) / i *
                               np.sum([binom(i, k)*np.power(params['a'][i-2] - 1., i-k) *
                                       d(k, Xi) for k in range(0, i+1)])) # eq. A9
        GammaT = lambda i, XTi: (3.*np.power(params['ab'][i-2], 1. - i) / i *
                                 np.sum([binom(i, k)*np.power(params['ab'][i-2] - 1., i-k) *
                                         d(k, XTi) for k in range(0, i+1)])) # eq. A9

    
        theta = params['theta_0'] * np.exp(params['grueneisen_0'] /
                                           (1. + params['delta'][0]) *
                                           (GammaT(2, XT[0]) - GammaT(2, 1.))) # eq. B1
    
        f = np.sqrt(1. + 2.*params['b'][1] *
                    (1. + params['delta'][1])*pressure/params['K_0'])
        
        I = (1. / (1. + params['b'][1]) *
             (params['b'][1] + f) *
             np.exp(1./params['b'][1] - f/params['b'][1])) # eq. D2
        
        G_C = params['K_0']*params['V_0']*np.sum([params['c'][i-2]*(Gamma(i, X[i-2]) - Gamma(i, 1))
                                                  for i in range(2, 6)]) # eq. A8
        
        G_QH = (3. * params['n'] * gas_constant * temperature *
                np.log(1. - np.exp(-theta/temperature))) # eq. 5
        
        G_QH0 = (3. * params['n'] * gas_constant * temperature *
                np.log(1. - np.exp(-params['theta_0']/temperature))) # eq. 5 at 1 bar
        
        return G_C + G_QH - (G_QH0 - self._gibbs_1bar(temperature, params))*I



        
        
        return self.molar_internal_energy(pressure, temperature, volume, params) + volume*pressure
        
    def molar_heat_capacity_v(self, pressure, temperature, volume, params):
        """
        Since this equation of state does not contain temperature effects, simply return a very large number. :math:`[J/K/mol]`
        """
        return 1.e99

    def molar_heat_capacity_p(self, pressure, temperature, volume, params):
        """
        Since this equation of state does not contain temperature effects, simply return a very large number. :math:`[J/K/mol]`
        """
        return 1.e99

    def thermal_expansivity(self, pressure, temperature, volume, params):
        """
        Since this equation of state does not contain temperature effects, simply return zero. :math:`[1/K]`
        """
        return 0.

    def grueneisen_parameter(self, pressure, temperature, volume, params):
        """
        Since this equation of state does not contain temperature effects, simply return zero. :math:`[unitless]`
        """
        return 0.

    def calculate_transformed_parameters(self, params):
        
        Z = {str(sl[0]): int(sl[1])
             for sl in [line.split() for line
                        in pkgutil.get_data('burnman',
                                            'data/input_masses/atomic_numbers.dat').decode('ascii').split('\n')
                        if len(line) > 0 and line[0] != '#']}
        
        
        nZs = [(n_at, float(Z[el])) for (el, n_at) in params['formula'].iteritems()]
        
        
        X3_300TPa = [np.power(1. - params['a'][i-2] +
                              params['a'][i-2]*np.power((1. + float(i)/(3.*params['a'][i-2]) *
                                                         300.e12/params['K_0']), 1./float(i)), -3.)
                     for i in range(2, 6)] # eq. A2
        X3_330TPa = [np.power(1. - params['a'][i-2] +
                              params['a'][i-2]*np.power((1. + float(i)/(3.*params['a'][i-2]) *
                                                         330.e12/params['K_0']), 1./float(i)), -3.)
                     for i in range(2, 6)] # eq. A2
        
        V_QSM_300TPa = np.sum([n_at *
                               ( 0.02713 *
                                 np.exp(0.97626*np.log(Zi) -
                                        0.057848*np.log(Zi)*np.log(Zi))
                               )
                               for (n_at, Zi) in nZs])*1.e-6 # eq. A6a, m^3/mol
        
        V_QSM_330TPa = np.sum([n_at *
                               ( 0.025692 *
                                 np.exp(0.97914*np.log(Zi) -
                                        0.057741*np.log(Zi)*np.log(Zi))
                               )
                               for (n_at, Zi) in nZs])*1.e-6 # eq. A6b, m^3/mol
        
        A = np.array([[1., 1., 1., 1.], # eq A3
                      [0., 6., 8., 9.], # eq A4
                      X3_300TPa, # eq A5a
                      X3_330TPa]) # eq A5b
        
        b = np.array([1., 8., V_QSM_300TPa/params['V_0'], V_QSM_330TPa/params['V_0']])
    
        return solve(A, b) # does not quite reproduce the published values of c; A.c consistently gives b[2], b[3] ~1% larger than Brosh

    
    def validate_parameters(self, params):
        """
        Check for existence and validity of the parameters
        """

        if 'P_0' not in params:
            params['P_0'] = 1.e5

        if 'a' not in params:
            params['a'] =  [(float(i)-1.)/(3.*params['Kprime_0'] - 1.)
                            for i in range(2, 6)] # eq. A2

        if 'ab' not in params:
            params['ab'] =  [(float(i)-1.)/(3.*params['b'][0] - 1.)
                             for i in range(2, 6)] # eq. A2, B1b
    
        if 'c' not in params:
            params['c'] = self.calculate_transformed_parameters(params)
            
        # Check that all the required keys are in the dictionary
        expected_keys = ['gibbs_coefficients',
                         'V_0', 'K_0', 'Kprime_0',
                         'theta_0', 'grueneisen_0', 'delta', 'b']
        for k in expected_keys:
            if k not in params:
                raise KeyError('params object missing parameter : ' + k)

        # Finally, check that the values are reasonable.
        if params['P_0'] < 0.:
            warnings.warn('Unusual value for P_0', stacklevel=2)
        if params['V_0'] < 1.e-7 or params['V_0'] > 1.e-3:
            warnings.warn('Unusual value for V_0', stacklevel=2)
        if params['K_0'] < 1.e9 or params['K_0'] > 1.e13:
            warnings.warn('Unusual value for K_0', stacklevel=2)
        if params['Kprime_0'] < 0. or params['Kprime_0'] > 10.:
            warnings.warn('Unusual value for Kprime_0', stacklevel=2)
            
