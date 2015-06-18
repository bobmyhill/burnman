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
    def _partition_function(self, mass, temperature): # eq. S7; qi/V. Note the typo - there should be temperature in this expression
        # Should V=Vi, rather than Vensemble?
        return np.power(mass*constants.Boltzmann*temperature/(2*np.pi*constants.Dirac*constants.Dirac), 3./2.)
    
    def _atomic_momenta(self, temperature, volume, params): # F_ig, eq. S6
        # ideal gas
        entropy_sum=0.
        for element, N in params['formula'].iteritems(): # N is a.p.f.u
            if N > 1.e-5:
                q=(volume/constants.Avogadro)*self._partition_function((atomic_masses[element]/constants.Avogadro), temperature) # masses are in kg/mol
                entropy_sum+=N*(1. + np.log(q/N)) # see also eq. 16.72 of Callen., 1985; p. 373
        return -constants.gas_constant*temperature*entropy_sum

    def _atomic_entropy(self, temperature, volume, params): # F_ig, eq. S6
        # ideal gas
        entropy_sum=0.
        for element, N in params['formula'].iteritems(): # N is a.p.f.u
            if N > 1.e-5:
                q=(volume/constants.Avogadro)*self._partition_function((atomic_masses[element]/constants.Avogadro), temperature) # masses are in kg/mol
                entropy_sum+=N*(5./2. + np.log(q/N)) # see also eq. 16.72 of Callen., 1985; p. 373
        return constants.gas_constant*entropy_sum


    def _atomic_pressure(self, temperature, volume, params): # PV = nRT
        n_atoms=0
        for element, N in params['formula'].iteritems():
            n_atoms += N
        return n_atoms*constants.gas_constant*temperature / volume

    # Finite strain
    def _finite_strain(self, temperature, volume, params): # f(V), eq. S3a
        return (1./2.)*(np.power(params['V_0']/volume, 2./3.) - 1.0)        

    def _dfdV(self, temperature, volume, params): # f(V), eq. S3a
        return (-1./3.)*np.power(params['V_0']/volume, 2./3.)/volume  

    # Temperature
    def _theta(self, temperature, volume, params): # theta, eq. S3b
        return np.power(temperature/params['T_0'], params['m']) - 1.

    def _dthetadT(self, temperature, volume, params):
        return params['m']*np.power(temperature/params['T_0'], params['m']) \
            / temperature


    # Electronic component
    def _zeta(self, temperature, volume, params): # eq. S5a, beta in deKoker thesis (3.34)
        return params['zeta_0']*(np.power(volume/params['el_V_0'], params['xi']))

    def _dzetadV(self, temperature, volume, params):
        return params['xi']*params['zeta_0']*(np.power(volume/params['el_V_0'], params['xi']))/volume

    def _Tel(self, temperature, volume, params): # eq. S5b
        return params['Tel_0']*(np.power(volume/params['el_V_0'], params['eta']))
                            
    def _dTeldV(self, temperature, volume, params):
        return params['eta']*params['Tel_0']*(np.power(volume/params['el_V_0'], params['eta']))/volume
                  
    def _electronic_excitation_energy(self, temperature, volume, params): # F_el
        temperature_el = self._Tel(temperature, volume, params)
        if temperature < temperature_el:
            F_el = 0
        else:
            F_el = -self._zeta(temperature, volume, params)*(0.5*(temperature*temperature - temperature_el*temperature_el) - temperature*temperature_el*np.log(temperature/temperature_el))
        return F_el

    def _electronic_excitation_entropy(self, temperature, volume, params): # P_el
        temperature_el = self._Tel(temperature, volume, params)
        if temperature < temperature_el:
            S_el = 0
        else:
            S_el = self._zeta(temperature, volume, params)*( temperature - temperature_el - temperature_el*np.log(temperature/temperature_el))
        return S_el

    def _electronic_excitation_pressure(self, temperature, volume, params): # P_el
        temperature_el = self._Tel(temperature, volume, params)
        if temperature < temperature_el:
            P_el = 0
        else:
            P_el =  self._dzetadV(temperature, volume, params) * (0.5*(temperature*temperature - temperature_el*temperature_el) - temperature*temperature_el*np.log(temperature/temperature_el))
            P_el += self._zeta(temperature, volume, params)*self._dTeldV(temperature, volume, params)*((temperature-temperature_el) - temperature*np.log(temperature/temperature_el))
        return P_el
    
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
        return 0.

    def adiabatic_bulk_modulus(self, pressure, temperature, volume, params):
        """
        Returns adiabatic bulk modulus. :math:`[Pa]` 
        """
        return 0.

    def grueneisen_parameter(self, pressure, temperature, volume, params):
        """
        Returns grueneisen parameter. :math:`[unitless]` 
        """
        return 0.

    def shear_modulus(self, pressure, temperature, volume, params):
        """
        Returns shear modulus. :math:`[Pa]` 
        """
        return 0.

    def heat_capacity_v(self, pressure, temperature, volume, params):
        """
        Returns heat capacity at constant volume. :math:`[J/K/mol]` 
        """
        return 0.

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
        
        return self.helmholtz_free_energy( pressure, temperature, volume, params) + \
               temperature * self.entropy( pressure, temperature, volume, params) + \
               pressure * self.volume( pressure, temperature, params)

    def helmholtz_free_energy( self, pressure, temperature, volume, params):
        """
        Returns the Helmholtz free energy at the pressure and temperature of the mineral [J/mol]
        """
        F = self._atomic_momenta(temperature, volume, params) + \
            self._electronic_excitation_energy(temperature, volume, params) + \
            self._bonding_energy(temperature, volume, params)
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
        expected_keys = ['V_0', 'T_0', 'E_0', 'S_0', 'O_theta', 'O_f', 'm', 'a', 'zeta_0', 'xi', 'Tel_0', 'eta', 'el_V_0']
        for k in expected_keys:
            if k not in params:
                raise KeyError('params object missing parameter : ' + k)
        
        #now check that the values are reasonable.  I mostly just
        #made up these values from experience, and we are only 
        #raising a warning.  Better way to do this? [IR]


