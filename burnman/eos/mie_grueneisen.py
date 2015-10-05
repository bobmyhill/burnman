# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

import numpy as np
import scipy.optimize as opt

import warnings

import equation_of_state as eos


class MG(eos.EquationOfState):
    """
    Base class for the Mie-Grueneisen finite strain solid equation of state 
    detailed in :cite:`deKoker2008`.
    """

    # Finite strain
    def __finite_strain(self, temperature, volume, params): # f(V), eq. 9
        return (1./2.)*(np.power(params['V_0']/volume, 2./3.) - 1.0)        

    # Pressure
    def pressure(self, temperature, volume, params): # eq. 4
        return self._pressure_c(temperature, volume, params) \
            + self.heat_capacity_v(0., temperature, volume, params) \
            * self.grueneisen_parameter(0., temperature, volume, params) / volume \
            * (temperature - params['T_0'])

    def internal_energy(self, pressure, temperature, volume, params): # eq. 5
        E = params['E_0'] \
            + self._energy_c(temperature, volume, params) \
            + self.heat_capacity_v(0., temperature, volume, params) \
            * (temperature - params['T_0'])
        return E

    def _pressure_c(self, temperature, volume, params): # eq. 6
        f = self.__finite_strain(temperature, volume, params)
        K_0 = params['K_0']
        K_prime_0 = params['K_prime_0']
        K_dprime_0 = params['K_dprime_0']
        a3 = 3. * ( K_prime_0 - 4. )
        a4 = 9. * ( K_0 * K_dprime_0 + K_prime_0 * (K_prime_0 - 7.) ) + 143.
        return 3.*params['K_0']*np.power(1+2.*f, 2.5)*(f + a3*f*f/2. + a4/6.*f*f*f)

    def _energy_c(self, temperature, volume, params): # eq. 7
        f = self.__finite_strain(temperature, volume, params)
        K_0 = params['K_0']
        K_prime_0 = params['K_prime_0']
        K_dprime_0 = params['K_dprime_0']
        a3 = 3.*(K_prime_0 - 4.)
        a4 = 9.* ( K_0 * K_dprime_0 + K_prime_0 * (K_prime_0 - 7.) ) + 143.
        return params['T_0'] * self._int_aKT_dV(temperature, volume, params) \
            + 9.*K_0*params['V_0']*(f*f/2. + a3*f*f*f/6. + a4*f*f*f*f/24.)

    # Evaluate the integral of aK_T(V,To) from V1 to V2
    def _int_aKT_dV(self, temperature, volume, params):
        g_x = params['grueneisen_V_x'] 
        gp_x = params['grueneisen_prime_V_x']
        c_x = params['C_v_V_x'] 
        cp_x = params['C_v_prime_V_x'] 
        V_x = params['V_x']
        return (g_x - gp_x)*(c_x - cp_x)*np.log(volume/V_x) \
            + (g_x - gp_x)*cp_x * (volume/V_x - 1.) \
            + (c_x - cp_x)*gp_x * (volume/V_x - 1.) \
            + gp_x * cp_x * 0.5 * (volume*volume/V_x/V_x - 1.)

    
    def volume(self, pressure, temperature, params):
        p_residual = lambda x: pressure - self.pressure(temperature, x, params)
        tol = 0.0001
        sol = opt.fsolve(p_residual, 0.8e-6, xtol=1e-12, full_output=True)
        if sol[2] != 1:
            raise ValueError('Cannot find volume, likely outside of the range of validity for EOS')
        else:
            return sol[0][0]


    def isothermal_bulk_modulus(self, pressure,temperature, volume, params):
        """
        Returns isothermal bulk modulus :math:`[Pa]` 
        """
        return 0.

    def adiabatic_bulk_modulus(self, pressure, temperature, volume, params):
        """
        Returns adiabatic bulk modulus. :math:`[Pa]` 
        """
        return 0.

    def grueneisen_parameter(self, pressure, temperature, volume, params): # eq. 13
        """
        Returns grueneisen parameter. :math:`[unitless]` 
        """
        return params['grueneisen_V_x'] \
            + params['grueneisen_prime_V_x'] \
            * ((volume/params['V_x']) - 1.)

    def shear_modulus(self, pressure, temperature, volume, params):
        """
        Returns shear modulus. :math:`[Pa]` 
        """
        return 0.

    def heat_capacity_v(self, pressure, temperature, volume, params): # eq. 12
        """
        Returns heat capacity at constant volume. :math:`[J/K/mol]` 
        """
        return params['C_v_V_x'] \
            + params['C_v_prime_V_x'] \
            * ((volume/params['V_x']) - 1.)

    def heat_capacity_p(self, pressure, temperature, volume, params):
        """
        Returns heat capacity at constant pressure. :math:`[J/K/mol]` 
        """
        return 0.

    def thermal_expansivity(self, pressure, temperature, volume, params):
        """
        Returns thermal expansivity. :math:`[1/K]` 
        """
        return 0.

    def gibbs_free_energy( self, pressure, temperature, volume, params):
        """
        Returns the Gibbs free energy at the pressure and temperature of the mineral [J/mol]
        """
        G = self.helmholtz_free_energy( pressure, temperature, volume, params) + pressure * volume
        return G

    def entropy( self, pressure, temperature, volume, params): # 3.41 of de Koker thesis
        """
        Returns the entropy at the pressure and temperature of the mineral [J/K/mol]
        """
        S = params['S_0'] + self._int_aKT_dV(temperature, volume, params) \
            + self.heat_capacity_v(pressure, temperature, volume, params)*np.log(temperature/params['T_0'])
        return S  

    def enthalpy( self, pressure, temperature, volume, params):
        """
        Returns the enthalpy at the pressure and temperature of the mineral [J/mol]
        """
        
        return self.helmholtz_free_energy( pressure, temperature, volume, params) + \
               temperature * self.entropy( pressure, temperature, volume, params) + \
               pressure * self.volume( pressure, temperature, params)

    def helmholtz_free_energy( self, pressure, temperature, volume, params):
        """
        Returns the Helmholtz free energy at the pressure and temperature of the mineral [J/mol]
        """
        F = self.internal_energy(pressure, temperature, volume, params) \
            + temperature*self.entropy(pressure, temperature, volume, params)
        return F

    def validate_parameters(self, params):
        """
        Check for existence and validity of the parameters
        """
  
        #check that all the required keys are in the dictionary
        expected_keys = ['V_0', 'T_0', 'E_0', 'S_0', 'K_0', 'K_prime_0', 'K_dprime_0', 'C_v_V_x', 'grueneisen_V_x']
        for k in expected_keys:
            if k not in params:
                raise KeyError('params object missing parameter : ' + k)
        
        #now check that the values are reasonable.  I mostly just
        #made up these values from experience, and we are only 
        #raising a warning.  Better way to do this? [IR]


