# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

import numpy as np
import scipy.optimize as opt
from scipy.misc import factorial

import warnings

import equation_of_state as eos
import burnman.constants as constants
from burnman.processchemistry import read_masses

atomic_masses=read_masses()

class DKS_L(eos.EquationOfState):
    """
    Base class for the finite strain liquid equation of state detailed
    in :cite:`deKoker2013` (supplementary materials).
    """

    # Atomic momenta
    # eq. S7; qi/V. 
    # Note the typo in that expression - 
    # there is a factor of temperature missing
    # (see de Koker thesis, eq. 3.14)
    # TODO?: Should V=Vi, rather than Vensemble? If so, how?

    def _ln_partition_function(self, mass, temperature):
        return 3./2.*np.log(temperature) \
            + 3./2.*np.log(mass*constants.Boltzmann \
                               /(2*np.pi*constants.Dirac*constants.Dirac)) \

    def _atomic_momenta(self, temperature, volume, params): # F_ig, eq. S6
        # ideal gas
        # see also eq. 16.72 of Callen., 1985; p. 373
        V = volume/constants.Avogadro
        figoverRT=0.
        for element, N in params['formula'].iteritems(): # N is a.p.f.u
            if N > 1.e-5:
                mass = atomic_masses[element]/constants.Avogadro
                figoverRT += -N*(np.log(V) + self._ln_partition_function(mass, temperature) \
                                     + 1.) + N*np.log(N)
        return constants.gas_constant*temperature*figoverRT
            

    def _atomic_entropy(self, temperature, volume, params): # F_ig, eq. S6
        # ideal gas
        V = volume/constants.Avogadro
        entropy_sum=0.
        for element, N in params['formula'].iteritems(): # N is a.p.f.u
            if N > 1.e-5:
                mass = atomic_masses[element]/constants.Avogadro
                entropy_sum -= -N*(np.log(V) + self._ln_partition_function(mass, temperature) \
                                     + 5./2.) + N*np.log(N)
        return constants.gas_constant*entropy_sum

    def _atomic_C_v(self, temperature, volume, params): # F_ig, eq. S6
        # ideal gas
        n_atoms=0
        for element, N in params['formula'].iteritems():
            n_atoms += N
        return 1.5*constants.gas_constant*n_atoms

    def _atomic_pressure(self, temperature, volume, params): # PV = nRT
        n_atoms=0
        for element, N in params['formula'].iteritems():
            n_atoms += N
        return n_atoms*constants.gas_constant*temperature / volume

    def _atomic_K_T(self, temperature, volume, params): # V * d/dV(-nRT/V) = V*nRT/V^2
        n_atoms=0
        for element, N in params['formula'].iteritems():
            n_atoms += N
        return n_atoms*constants.gas_constant*temperature / volume

    def _atomic_alphaK_T(self, temperature, volume, params): # d/dT(nRT/V) = nR/V
        n_atoms=0
        for element, N in params['formula'].iteritems():
            n_atoms += N
        return n_atoms*constants.gas_constant / volume

    # Finite strain
    def _finite_strain(self, temperature, volume, params): # f(V), eq. S3a
        return (1./2.)*(np.power(params['V_0']/volume, 2./3.) - 1.0)        

    def _dfdV(self, temperature, volume, params): # f(V), eq. S3a
        return (-1./3.)*np.power(params['V_0']/volume, 2./3.)/volume  

    def _d2fdV2(self,temperature, volume, params):
        return (5./9.)*np.power(params['V_0']/volume, 2./3.)/volume/volume

    # Temperature
    def _theta(self, temperature, volume, params): # theta, eq. S3b
        return np.power(temperature/params['T_0'], params['m']) - 1.

    def _dthetadT(self, temperature, volume, params):
        return params['m']*np.power(temperature/params['T_0'], params['m']) \
            / temperature

    def _d2thetadT2(self, temperature, volume, params):
        return params['m']*(params['m']-1.)*np.power(temperature/params['T_0'], params['m']) \
            / temperature / temperature

    # Electronic component
    def _zeta(self, temperature, volume, params): # eq. S5a, beta in deKoker thesis (3.34)
        return params['zeta_0']*(np.power(volume/params['el_V_0'], params['xi']))

    def _dzetadV(self, temperature, volume, params):
        return params['zeta_0']*params['xi']*(np.power(volume/params['el_V_0'], params['xi']))/volume

    def _d2zetadV2(self, temperature, volume, params):
        return params['zeta_0'] \
            * params['xi'] * (params['xi'] - 1.) \
            * (np.power(volume/params['el_V_0'], params['xi'])) \
            / volume / volume

    def _Tel(self, temperature, volume, params): # eq. S5b
        return params['Tel_0']*(np.power(volume/params['el_V_0'], params['eta']))
                            
    def _dTeldV(self, temperature, volume, params):
        return params['Tel_0'] * params['eta'] \
            * (np.power(volume/params['el_V_0'], params['eta'])) \
            / volume

    def _d2TeldV2(self, temperature, volume, params):
        return params['Tel_0'] \
            * params['eta'] * (params['eta'] - 1.) \
            * (np.power(volume/params['el_V_0'], params['eta'])) \
            / volume / volume

    def _gimel(self, temperature_el, temperature, volume, params): # -F_el/zeta, 3.30 in de Koker thesis
        return 0.5*(temperature*temperature - temperature_el*temperature_el) \
            - temperature*temperature_el*np.log(temperature/temperature_el)

    def _dgimeldTel(self, temperature_el, temperature, volume, params): 
        return (temperature-temperature_el) - temperature*np.log(temperature/temperature_el)

    def _dgimeldT(self, temperature_el, temperature, volume, params): 
        return (temperature-temperature_el) - temperature_el*np.log(temperature/temperature_el)

    def _d2gimeldTdTel(self, temperature_el, temperature, volume, params):
        return -np.log(temperature/temperature_el)
            
    def _d2gimeldTel2(self, temperature_el, temperature, volume, params):
        return (temperature/temperature_el)  - 1.
      
    def _electronic_excitation_energy(self, temperature, volume, params): # F_el
        temperature_el = self._Tel(temperature, volume, params)
        if temperature < temperature_el:
            F_el = 0
        else:
            F_el = -self._zeta(temperature, volume, params) \
                * self._gimel(temperature_el, temperature, volume, params)
        return F_el

    def _electronic_excitation_entropy(self, temperature, volume, params): # S_el
        temperature_el = self._Tel(temperature, volume, params)
        if temperature < temperature_el:
            S_el = 0
        else:
            S_el = self._zeta(temperature, volume, params) \
                * self._dgimeldT(temperature_el, temperature, volume, params)
        return S_el
        

    def _electronic_excitation_pressure(self, temperature, volume, params): # P_el
        temperature_el = self._Tel(temperature, volume, params)
        if temperature < temperature_el:
            P_el = 0
        else:
            P_el =  self._dzetadV(temperature, volume, params) \
                * self._gimel(temperature_el, temperature, volume, params) \
                + self._zeta(temperature, volume, params) \
                * self._dTeldV(temperature, volume, params) \
                * self._dgimeldTel(temperature_el, temperature, volume, params)
        return P_el
 
    def _electronic_excitation_K_T(self, temperature, volume, params): # K_T_el
        temperature_el = self._Tel(temperature, volume, params)
        if temperature < temperature_el:
            K_T_el = 0
        else:
            K_T_el =  -volume \
                * ( self._d2zetadV2(temperature, volume, params) \
                        * self._gimel(temperature_el, temperature, volume, params) \
                        + 2. * self._dzetadV(temperature, volume, params) \
                        * self._dgimeldTel(temperature_el, temperature, volume, params) \
                        * self._dTeldV(temperature, volume, params) \
                        + self._zeta(temperature, volume, params) \
                        * ( self._d2TeldV2(temperature, volume, params) \
                                * self._dgimeldTel(temperature_el, temperature, volume, params) \
                                + self._dTeldV(temperature, volume, params) \
                                * self._dTeldV(temperature, volume, params) \
                                * self._d2gimeldTel2(temperature_el, temperature, volume, params)))
        return K_T_el
    
    def _electronic_excitation_alphaK_T(self, temperature, volume, params): # (alphaK_T)_el
        temperature_el = self._Tel(temperature, volume, params)
        if temperature < temperature_el:
            alphaK_T_el = 0
        else:
            alphaK_T_el = self._dzetadV(temperature, volume, params) \
                * self._dgimeldT(temperature_el, temperature, volume, params) \
                + self._zeta(temperature, volume, params) \
                * self._d2gimeldTdTel(temperature_el, temperature, volume, params) \
                * self._dTeldV(temperature, volume, params)
        return alphaK_T_el

    def _electronic_C_v(self, temperature, volume, params): # C_el, eq. 3.28 of de Koker thesis
        temperature_el = self._Tel(temperature, volume, params)
        zeta = self._zeta(temperature, volume, params)

        if temperature > temperature_el:
            Cv_el = zeta*(temperature - temperature_el)
        else:
            Cv_el = 0.
        return Cv_el

    # Bonding energy
    def _bonding_energy(self, temperature, volume, params): # F_xs, eq. S2
        f = self._finite_strain(temperature, volume, params)
        theta = self._theta(temperature, volume, params)
        energy = 0.
        for i in range(len(params['a'])):
            ifact=factorial(i, exact=False)
            for j in range(len(params['a'][0])):
                jfact=factorial(j, exact=False)
                energy += params['a'][i][j]*np.power(f, i)*np.power(theta, j)/ifact/jfact         
        return energy

    def _bonding_entropy(self, temperature, volume, params): # F_xs, eq. 3.18
        f = self._finite_strain(temperature, volume, params)
        theta = self._theta(temperature, volume, params)
        entropy = 0.
        for i in range(len(params['a'])):
            ifact = factorial(i, exact=False)
            for j in range(len(params['a'][0])):
                if j > 0:
                    jfact = factorial(j, exact=False)
                    entropy += j*params['a'][i][j]*np.power(f, i)*np.power(theta, j-1.)/ifact/jfact         
        return -self._dthetadT(temperature, volume, params)*entropy

    def _bonding_pressure(self, temperature, volume, params): # P_xs, eq. 3.17 of de Koker thesis
        f = self._finite_strain(temperature, volume, params)
        theta = self._theta(temperature, volume, params)
        pressure=0.
        for i in range(len(params['a'])):
            ifact=factorial(i, exact=False)
            if i > 0:
                for j in range(len(params['a'][0])):
                    jfact=factorial(j, exact=False)
                    pressure += float(i)*params['a'][i][j]*np.power(f, float(i)-1.)*np.power(theta, float(j))/ifact/jfact
        return -self._dfdV(temperature, volume, params)*pressure

    def _bonding_K_T(self, temperature, volume, params): # K_T_xs, eq. 3.20 of de Koker thesis
        f = self._finite_strain(temperature, volume, params)
        theta = self._theta(temperature, volume, params)
        K_ToverV=0.
        for i in range(len(params['a'])):
            ifact=factorial(i, exact=False)
            for j in range(len(params['a'][0])):
                if i > 0:
                    jfact=factorial(j, exact=False)
                    prefactor = float(i) * params['a'][i][j] \
                        * np.power(theta, float(j)) / ifact / jfact 
                    K_ToverV += prefactor*self._d2fdV2(temperature, volume, params) \
                        * np.power(f, float(i-1))
                if i > 1:
                    dfdV = self._dfdV(temperature, volume, params)
                    K_ToverV += prefactor * dfdV * dfdV \
                        * float(i-1) * np.power(f, float(i-2))
        return volume*K_ToverV

    def _bonding_alphaK_T(self, temperature, volume, params): # eq. 3.21 of de Koker thesis
        f = self._finite_strain(temperature, volume, params)
        theta = self._theta(temperature, volume, params)
        sum_factors = 0.
        for i in range(len(params['a'])):
            ifact=factorial(i, exact=False)
            if i > 0:
                for j in range(len(params['a'][0])):
                    if j > 0:
                        jfact=factorial(j, exact=False)
                        sum_factors += float(i)*float(j)*params['a'][i][j] \
                            * np.power(f, float(i-1)) * np.power(theta, float(j-1)) \
                            / ifact / jfact
                            
        return -self._dfdV(temperature, volume, params) \
            * self._dthetadT(temperature, volume, params) \
            * sum_factors
            

    def _bonding_C_v(self, temperature, volume, params): # Cv_xs, eq. 3.22 of de Koker thesis
        f = self._finite_strain(temperature, volume, params)
        theta = self._theta(temperature, volume, params)
        C_voverT=0.
        for i in range(len(params['a'])):
            ifact=factorial(i, exact=False)
            for j in range(len(params['a'][0])):
                if j > 0:
                    jfact=factorial(j, exact=False)
                    prefactor = float(j)*params['a'][i][j]*np.power(f, float(i))/ifact/jfact
                    C_voverT += prefactor * self._d2thetadT2(temperature, volume, params) \
                        * np.power(theta, float(j-1))
                if j > 1:
                    dthetadT = self._dthetadT(temperature, volume, params)
                    C_voverT += prefactor * dthetadT * dthetadT \
                        * float(j-1) * np.power(theta, float(j-2))
        return -temperature*C_voverT


    def _aK_T(self, temperature, volume, params):
        return self._atomic_alphaK_T(temperature, volume, params) \
            + self._electronic_excitation_alphaK_T(temperature, volume, params) \
            + self._bonding_alphaK_T(temperature, volume, params) 

    # Pressure
    def pressure(self, temperature, volume, params):
        P = self._atomic_pressure(temperature, volume, params) + \
            self._electronic_excitation_pressure(temperature, volume, params) + \
            self._bonding_pressure(temperature, volume, params)
        return P

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
        K_T = self._atomic_K_T(temperature, volume, params) \
            + self._bonding_K_T(temperature, volume, params) \
            + self._electronic_excitation_K_T(temperature, volume, params)
        return K_T

    def adiabatic_bulk_modulus(self, pressure, temperature, volume, params):
        """
        Returns adiabatic bulk modulus. :math:`[Pa]` 
        """
        K_S = self.isothermal_bulk_modulus(pressure,temperature, volume, params) \
            * ( 1. + temperature \
                    * self.thermal_expansivity(pressure, temperature, volume, params) \
                    * self.grueneisen_parameter(pressure, temperature, volume, params) )
        return K_S

    def grueneisen_parameter(self, pressure, temperature, volume, params):
        """
        Returns grueneisen parameter. :math:`[unitless]` 
        """
        gamma = self._aK_T(temperature, volume, params) \
            * volume / self.heat_capacity_v(pressure, temperature, volume, params)
        return gamma

    def shear_modulus(self, pressure, temperature, volume, params):
        """
        Returns shear modulus. :math:`[Pa]` 
        Zero for fluids
        """
        return 0.

    def heat_capacity_v(self, pressure, temperature, volume, params):
        """
        Returns heat capacity at constant volume. :math:`[J/K/mol]` 
        """
        C_v = self._atomic_C_v(temperature, volume, params) + \
            self._bonding_C_v(temperature, volume, params) + \
            self._electronic_C_v(temperature, volume, params)
        return C_v

    def heat_capacity_p(self, pressure, temperature, volume, params):
        """
        Returns heat capacity at constant pressure. :math:`[J/K/mol]` 
        """
        C_p = self.heat_capacity_v(pressure,temperature, volume, params) \
            * ( 1. + temperature \
                    * self.thermal_expansivity(pressure, temperature, volume, params) \
                    * self.grueneisen_parameter(pressure, temperature, volume, params) )
        return C_p

    def thermal_expansivity(self, pressure, temperature, volume, params):
        """
        Returns thermal expansivity. :math:`[1/K]` 
        """
        alpha = self._aK_T(temperature, volume, params) \
            / self.isothermal_bulk_modulus(0., temperature, volume, params)
        return alpha

    def gibbs_free_energy( self, pressure, temperature, volume, params):
        """
        Returns the Gibbs free energy at the pressure and temperature of the mineral [J/mol]
        """
        G = self.helmholtz_free_energy( pressure, temperature, volume, params) + pressure * volume 
        return G

    def entropy( self, pressure, temperature, volume, params):
        """
        Returns the entropy at the pressure and temperature of the mineral [J/K/mol]
        """
        S = self._atomic_entropy(temperature, volume, params) + \
            self._electronic_excitation_entropy(temperature, volume, params) + \
            self._bonding_entropy(temperature, volume, params)
        return S 

    def enthalpy( self, pressure, temperature, volume, params):
        """
        Returns the enthalpy at the pressure and temperature of the mineral [J/mol]
        """
        H = self.helmholtz_free_energy( pressure, temperature, volume, params) + \
            temperature * self.entropy( pressure, temperature, volume, params) + \
            pressure * self.volume( pressure, temperature, params)
        return H

    def helmholtz_free_energy( self, pressure, temperature, volume, params):
        """
        Returns the Helmholtz free energy at the pressure and temperature of the mineral [J/mol]
        """
        F = self._atomic_momenta(temperature, volume, params) \
            + self._electronic_excitation_energy(temperature, volume, params) \
            + self._bonding_energy(temperature, volume, params)
        return F

    def internal_energy(self, pressure, temperature, volume, params):
        E = self.helmholtz_free_energy(pressure, temperature, volume, params) + \
            temperature*self.entropy(pressure, temperature, volume, params)
        return E

    def validate_parameters(self, params):
        """
        Check for existence and validity of the parameters
        """
  
        #check that all the required keys are in the dictionary
        expected_keys = ['V_0', 'T_0', 'O_theta', 'O_f', 'm', 'a', 'zeta_0', 'xi', 'Tel_0', 'eta', 'el_V_0']
        for k in expected_keys:
            if k not in params:
                raise KeyError('params object missing parameter : ' + k)
        
        #now check that the values are reasonable.  I mostly just
        #made up these values from experience, and we are only 
        #raising a warning.  Better way to do this? [IR]


