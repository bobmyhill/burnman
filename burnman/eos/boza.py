from __future__ import absolute_import
# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2017 by the BurnMan team, released under the GNU
# GPL v2 or later.


import numpy as np
import warnings

from . import birch_murnaghan as bm
from . import equation_of_state as eos

    
class BOZA(eos.EquationOfState):

    """
    Class for the Belmonte et al. (2017) HP-HT EoS.
    """

    def _1bar_V_params(self, temperature, params):
        intadT = lambda T, a: ( 0.5*a[0]*T*T + a[1]*T + a[2]*np.log(T)
                                -a[3]/T - 0.5*a[4]/T/T )
        
        V = params['V_0']*np.exp(intadT(temperature, params['alpha']) -
                                 intadT(params['T_0'], params['alpha']))
        K = ( params['K_0'] +
              (temperature - params['T_0'])*params['dKdT'] +
              np.power(temperature - params['T_0'], 2.)*params['d2KdT2'] )
        Kprime = ( params['Kprime_0'] +
                   params['dKdT'] * (temperature - params['T_0']) *
                   np.log(temperature/params['T_0']) )
        return {'V_0': V, 'K_0': K, 'Kprime_0': Kprime, 'P_0': params['P_0']}
    
    def volume(self, pressure, temperature, params):
        """
        Returns volume [m^3] as a function of pressure [Pa] and temperature [K]
        """
    
        return bm.volume(pressure, self._1bar_V_params(temperature, params))

    def pressure(self, temperature, volume, params):
        """
        Returns pressure [Pa] as a function of temperature [K] and volume[m^3]
        """
        HTparams = self._1bar_V_params(temperature, params)
        return bm.birch_murnaghan(HTparams['V_0'] / volume, HTparams)

    def grueneisen_parameter(self, pressure, temperature, volume, params):
        """
        Returns grueneisen parameter [unitless] as a function of pressure,
        temperature, and volume.
        """
        alpha = self.thermal_expansivity(
            pressure, temperature, volume, params)
        K_T = self.isothermal_bulk_modulus(
            pressure, temperature, volume, params)
        C_V = self.heat_capacity_v(pressure, temperature, volume, params)
        return alpha * K_T * volume / C_V

    def isothermal_bulk_modulus(self, pressure, temperature, volume, params):
        """
        Returns isothermal bulk modulus [Pa] as a function of pressure [Pa],
        temperature [K], and volume [m^3].  EQ 13+2
        """
        
        return bm.bulk_modulus(volume, self._1bar_V_params(temperature, params))

    # calculate the shear modulus as a function of P, V, and T
    def shear_modulus(self, pressure, temperature, volume, params):
        """
        Not implemented.
        Returns 0.
        Could potentially apply a fixed Poissons ratio as a rough estimate.
        """
        return 0.

    # Cv, heat capacity at constant volume
    def heat_capacity_v(self, pressure, temperature, volume, params):
        """
        Returns heat capacity at constant volume at the pressure, temperature, and volume [J/K/mol].
        """
        C_p = self.heat_capacity_p(pressure, temperature, volume, params)
        V = self.volume(pressure, temperature, params)
        alpha = self.thermal_expansivity(pressure, temperature, volume, params)
        K_T = self.isothermal_bulk_modulus(
            pressure, temperature, volume, params)
        return C_p - V * temperature * alpha * alpha * K_T

    def thermal_expansivity(self, pressure, temperature, volume, params):
        """
        Returns thermal expansivity at the pressure, temperature, and volume [1/K]
        """
        dT = 1.
        V0 = self.volume(pressure, temperature - 0.5*dT, params)
        V1 = self.volume(pressure, temperature + 0.5*dT, params)
        return 1./volume*(V1 - V0)/dT

    def adiabatic_bulk_modulus(self, pressure, temperature, volume, params):
        """
        Returns adiabatic bulk modulus [Pa] as a function of pressure [Pa],
        temperature [K], and volume [m^3].
        """
        K_T = self.isothermal_bulk_modulus(pressure, temperature, volume, params)
        C_p = self.heat_capacity_p(pressure, temperature, volume, params)
        C_v = self.heat_capacity_v(pressure, temperature, volume, params)
        K_S = K_T * C_p / C_v
        return K_S

    def _intCpdT(self, T0, temperature, params):
        integral = lambda T, c: ( c[0]*T + 0.5*c[1]*T*T - c[2]/T + 2.*c[3]*np.sqrt(T)
                                  -0.5*c[4]/T/T + 1./3.*c[5]*T*T*T + 1./4.*c[6]*T*T*T*T
                                  + c[7]*np.log(T))
        
        return integral(temperature, params['C_p']) - integral(T0, params['C_p'])
    
    def _intCpoverTdT(self, T0, temperature, params):
        integral = lambda T, c: ( c[0]*np.log(T) + c[1]*T -0.5*c[2]/T/T - 2*c[3]/np.sqrt(T)
                                  -1./3.*c[4]/T/T/T + 1./2.*c[5]*T*T + 1./3.*c[6]*T*T*T
                                  - c[7]/T )
        
        return integral(temperature, params['C_p']) - integral(T0, params['C_p'])

    def gibbs_free_energy(self, pressure, temperature, volume, params):
        """
        Returns the gibbs free energy [J/mol] as a function of pressure [Pa]
        and temperature [K].
        """
        HTparams = self._1bar_V_params(temperature, params)
        x = np.power(volume/HTparams['V_0'], -1./3.)
        return ( params['H_0'] - temperature*params['S_0'] + volume*pressure + 
                 self._intCpdT(params['T_0'], temperature, params) -
                 temperature*self._intCpoverTdT(params['T_0'], temperature, params) -
                 bm.intPdV(x, HTparams) )
    
    
    def helmholtz_free_energy(self, pressure, temperature, volume, params):
        return self.gibbs_free_energy(pressure, temperature, volume, params) - pressure * self.volume(pressure, temperature, params)

    def entropy(self, pressure, temperature, volume, params):
        """
        Returns the entropy [J/K/mol] as a function of pressure [Pa]
        and temperature [K].
        """
        
        dT = 1.
        G0 = self.gibbs_free_energy(pressure, temperature - 0.5*dT, volume, params)
        G1 = self.gibbs_free_energy(pressure, temperature + 0.5*dT, volume, params)
        return -(G1 - G0)/dT

    def enthalpy(self, pressure, temperature, volume, params):
        """
        Returns the enthalpy [J/mol] as a function of pressure [Pa]
        and temperature [K].
        """
        gibbs = self.gibbs_free_energy(pressure, temperature, volume, params)
        entropy = self.entropy(pressure, temperature, volume, params)
        return gibbs + temperature * entropy

    def heat_capacity_p(self, pressure, temperature, volume, params):
        """
        Returns the heat capacity [J/K/mol] as a function of pressure [Pa]
        and temperature [K].
        """
        
        dT = 1.
        S0 = self.entropy(pressure, temperature - 0.5*dT, volume, params)
        S1 = self.entropy(pressure, temperature + 0.5*dT, volume, params)
        return temperature*(S1 - S0)/dT

    def validate_parameters(self, params):
        """
        Check for existence and validity of the parameters
        """
        if 'T_0' not in params:
            params['T_0'] = 298.15


        # First, let's check the EoS parameters for Tref
        bm.BirchMurnaghanBase.validate_parameters(bm.BirchMurnaghanBase(), params)

        # Now check all the required keys for the
        # thermal part of the EoS are in the dictionary
        expected_keys = ['H_0', 'S_0', 'V_0', 'n', 'molar_mass']
        for k in expected_keys:
            if k not in params:
                raise KeyError('params object missing parameter : ' + k)


        # Finally, check that the values are reasonable.
        if params['T_0'] < 0.:
            warnings.warn('Unusual value for T_0', stacklevel=2)
        if params['G_0'] is not float('nan') and (params['G_0'] < 0. or params['G_0'] > 1.e13):
            warnings.warn('Unusual value for G_0', stacklevel=2)
        if params['Gprime_0'] is not float('nan') and (params['Gprime_0'] < -5. or params['Gprime_0'] > 10.):
            warnings.warn('Unusual value for Gprime_0', stacklevel=2)

        # no test for H_0
        if params['S_0'] is not float('nan') and params['S_0'] < 0.:
            warnings.warn('Unusual value for S_0', stacklevel=2)
        if params['V_0'] < 1.e-7 or params['V_0'] > 1.e-2:
            warnings.warn('Unusual value for V_0', stacklevel=2)


        if params['n'] < 1. or params['n'] > 1000.:
            warnings.warn('Unusual value for n', stacklevel=2)
        if params['molar_mass'] < 0.001 or params['molar_mass'] > 10.:
            warnings.warn('Unusual value for molar_mass', stacklevel=2)
