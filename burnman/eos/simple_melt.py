# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2017 by the BurnMan team, released under the GNU
# GPL v2 or later.

from __future__ import absolute_import

import numpy as np
from scipy.optimize import brentq
from scipy.special import expi
from mpmath import gammainc
import warnings

# Try to import the jit from numba.  If it is
# not available, just go with the standard
# python interpreter
try:
    from numba import jit
except ImportError:
    def jit(fn):
        return fn


from . import reciprocal_kprime as rkprime
from . import equation_of_state as eos
from ..tools import bracket




class SimpleMelt(eos.EquationOfState):

    """
    Base class for a simple melt
    """

    def _intVdP(self, xi, params):
        
        a = params['Kprime_inf']
        b = (params['Kprime_0']/params['Kprime_inf']/params['Kprime_inf'] -
             params['Kprime_0']/params['Kprime_inf'] - 1.)
        c = params['Kprime_0'] - params['Kprime_inf']
        f = (params['Kprime_0']/params['Kprime_inf'] - 1.)
        
        i1 = float( params['V_0'] * params['K_0'] *
                    np.exp(f / a) * np.power(a, b - 1.) /
                    np.power(f, b + 2.) *
                    ( f * params['Kprime_0'] * gammainc( b + 1. ,
                                                         f * (1./a - xi) ) -
                      a * c * gammainc( b + 2., f * (1./a - xi) ) ) )
        i0 = float(params['V_0'] * params['K_0'] * np.exp(f / a) * np.power(a, b - 1.) /
                   np.power(f, b + 2.) *
                   ( f * params['Kprime_0'] * gammainc( b + 1., f/a ) -
                     a * c * gammainc( b + 2., f/a ) ) )
        
        return i1 - i0


    def _temperature_principal_isentrope(self, volume, params):
        lmda = params['q_0']/np.log(params['gamma_0']/params['gamma_inf'])
        intgammaoverVdV = ( params['gamma_inf'] / lmda *
                            (expi(params['q_0']/lmda * np.power(volume/params['V_0'], lmda)) -
                             expi(params['q_0']/lmda)) )
        return params['T_0']*np.exp(-intgammaoverVdV)
    

    def _grueneisen_parameter(self, volume, params):
        lmda = params['q_0']/np.log(params['gamma_0']/params['gamma_inf'])
        return params['gamma_0'] * np.exp(params['q_0']/lmda * (np.power(volume/params['V_0'], lmda) - 1.)) # eq 101

    def _thermal_pressure(self, temperature, volume, params):
        T0 = self._temperature_principal_isentrope(volume, params)
        Pth = self._grueneisen_parameter(volume, params) * params['C_v'] * (temperature - T0) / volume
        return Pth

    def _delta_pressure(self, volume, pressure, temperature, params):
        dP = pressure - (rkprime.RKprime.pressure(rkprime.RKprime(), temperature, volume, params) +
                         self._thermal_pressure(temperature, volume, params))
        return dP

    def _delta_internal_energy_principal_isentrope(self, volume, params):
        pressure = rkprime.RKprime.pressure(rkprime.RKprime(), 0., volume, params)
        K_S = rkprime.RKprime.adiabatic_bulk_modulus(rkprime.RKprime(), pressure, 0., volume, params)
        xi = pressure/K_S
        U = -(pressure*volume - params['P_0']*params['V_0']) + self._intVdP(xi, params)
        return U
    
    def volume(self, pressure, temperature, params):
        """
        Returns molar volume. :math:`[m^3]`
        """
        Vlim = (params['V_0'] *
                np.power ( params['Kprime_0'] /
                           (params['Kprime_0'] - params['Kprime_inf']),
                           params['Kprime_0'] /
                           params['Kprime_inf'] /
                           params['Kprime_inf'] ) *
                np.exp(-1./params['Kprime_inf']) )
        args = (pressure, temperature, params)
        return brentq(self._delta_pressure, 1.e-2*params['V_0'], Vlim - 1.e-12, args=args)
        

    def pressure(self, temperature, volume, params):
        """
        Returns the pressure of the mineral at a given temperature and volume [Pa]
        """
        P = ( rkprime.RKprime.pressure(rkprime.RKprime(), 0., volume, params) +
              self._thermal_pressure(temperature, volume, params) )
        return P

    def grueneisen_parameter(self, pressure, temperature, volume, params):
        """
        Returns grueneisen parameter :math:`[unitless]`
        """
        return self._grueneisen_parameter(volume, params)

    def adiabatic_bulk_modulus(self, pressure, temperature, volume, params):
        """
        Returns adiabatic bulk modulus. :math:`[Pa]`
        """

        # K_S = -V dP/dV|S
        dV = params['V_0']*1.e-4
        S = self.entropy(pressure, temperature, volume, params)
        V0 = volume - 0.5 * dV
        V1 = volume + 0.5 * dV
        T_isentrope0 = self._temperature_principal_isentrope(V0, params)
        T_isentrope = self._temperature_principal_isentrope(volume, params)
        T_isentrope1 = self._temperature_principal_isentrope(V1, params)

        T0 = temperature/T_isentrope * T_isentrope0
        T1 = temperature/T_isentrope * T_isentrope1

        P0 = self.pressure(T0, V0, params)
        P1 = self.pressure(T1, V1, params)

        K_S = -volume * (P1 - P0)/(V1 - V0)
        return K_S

    def shear_modulus(self, pressure, temperature, volume, params):
        """
        Returns shear modulus. :math:`[Pa]`
        """
        
        #raise NotImplementedError("")
        return 0.

    def heat_capacity_v(self, pressure, temperature, volume, params):
        """
        Returns heat capacity at constant volume. :math:`[J/K/mol]`
        """
        
        return params['C_v']

    def isothermal_bulk_modulus(self, pressure, temperature, volume, params):
        """
        Returns isothermal bulk modulus :math:`[Pa]`
        """

        K_S = self.adiabatic_bulk_modulus(pressure, temperature, volume, params)
        gr = self.grueneisen_parameter(pressure, temperature, volume, params)
        C_v = self.heat_capacity_v(pressure, temperature, volume, params)
        K_T = K_S - gr*gr*C_v*temperature/volume

        return K_T
    
    def heat_capacity_p(self, pressure, temperature, volume, params):
        """
        Returns heat capacity at constant pressure. :math:`[J/K/mol]`
        """
        alpha = self.thermal_expansivity(pressure, temperature, volume, params)
        gr = self.grueneisen_parameter(pressure, temperature, volume, params)
        C_v = self.heat_capacity_v(pressure, temperature, volume, params)
        C_p = C_v * (1. + gr * alpha * temperature)
        return C_p

    def thermal_expansivity(self, pressure, temperature, volume, params):
        """
        Returns thermal expansivity. :math:`[1/K]`
        """
        C_v = self.heat_capacity_v(pressure, temperature, volume, params)
        gr = self.grueneisen_parameter(pressure, temperature, volume, params)
        K_T = self.isothermal_bulk_modulus(pressure, temperature, volume, params)
        alpha = gr * C_v / K_T / volume
        return alpha
    
    def entropy(self, pressure, temperature, volume, params):
        """
        Returns the entropy at the pressure and temperature of the mineral [J/K/mol]
        """
        T_isentrope = self._temperature_principal_isentrope(volume, params)
        
        DS = params['C_v'] * np.log(temperature/T_isentrope)
        return params['S_0'] + DS
    
    def internal_energy(self, pressure, temperature, volume, params):
        """
        Returns the internal energy at the pressure and temperature of the mineral [J/mol]
        """
        
        DE_isentrope = self._delta_internal_energy_principal_isentrope(volume, params)
        T_isentrope = self._temperature_principal_isentrope(volume, params)
        DE_isochore = params['C_v']*(temperature - T_isentrope)
        E_0 = params['F_0'] + params['T_0']*params['S_0']
        
        return E_0 + DE_isentrope + DE_isochore

    def gibbs_free_energy(self, pressure, temperature, volume, params):
        """
        Returns the Gibbs free energy at the pressure and temperature of the mineral [J/mol]
        """
        G = ( self.helmholtz_free_energy(pressure, temperature, volume, params) +
              pressure * volume )
        return G
    
    def enthalpy(self, pressure, temperature, volume, params):
        """
        Returns the enthalpy at the pressure and temperature of the mineral [J/mol]
        """

        return ( self.internal_energy(pressure, temperature, volume, params) + 
                 pressure * volume )

    def helmholtz_free_energy(self, pressure, temperature, volume, params):
        """
        Returns the Helmholtz free energy at the pressure and temperature of the mineral [J/mol]
        """
        
        F = ( self.internal_energy(pressure, temperature, volume, params) -
              temperature * self.entropy(pressure, temperature, volume, params) )

        return F

    def validate_parameters(self, params):
        """
        Check for existence and validity of the parameters
        """

        # First, let's check the EoS parameters for Tref
        rkprime.RKprime.validate_parameters(rkprime.RKprime(), params)

        # Now check all the required keys for the
        # thermal part of the EoS are in the dictionary
        expected_keys = ['molar_mass', 'n', 'gamma_0', 'gamma_inf',
                         'q_0', 'C_v', 'F_0', 'T_0', 'S_0',
                         'lambda_0', 'lambda_inf']
        for k in expected_keys:
            if k not in params:
                raise KeyError('params object missing parameter : ' + k)

        # Finally, check that the values are reasonable.
        '''
        if params['T_0'] < 0.:
            warnings.warn('Unusual value for T_0', stacklevel=2)
        if params['molar_mass'] < 0.001 or params['molar_mass'] > 10.:
            warnings.warn('Unusual value for molar_mass', stacklevel=2)
        if params['n'] < 1. or params['n'] > 1000.:
            warnings.warn('Unusual value for n', stacklevel=2)
        if params['Debye_0'] < 1. or params['Debye_0'] > 10000.:
            warnings.warn('Unusual value for Debye_0', stacklevel=2)
        if params['grueneisen_0'] < -0.005 or params['grueneisen_0'] > 10.:
            warnings.warn('Unusual value for grueneisen_0', stacklevel=2)
        if params['q_0'] < -10. or params['q_0'] > 10.:
            warnings.warn('Unusual value for q_0', stacklevel=2)
        if params['eta_s_0'] < -10. or params['eta_s_0'] > 10.:
            warnings.warn('Unusual value for eta_s_0', stacklevel=2)
        '''
