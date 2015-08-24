# BurnMan - a lower mantle toolkit
# Copyright (C) 2012-2014, Myhill, R., Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

import numpy as np
from scipy import optimize, integrate
import warnings

import modified_tait as mt
import equation_of_state as eos

import einstein

from burnman.endmemberdisorder import *


T_0=298.15 # Standard temperature = 25 C
P_0=1.e5 # Standard pressure = 1.e5 Pa


class V_AG(eos.EquationOfState):
    """
    Base class for the Vinet-Anderson-Grueneisen equation of state.

    An instance "m" of a Mineral can be assigned this 
    equation of state with the command m.set_method('v_ag')
    (or by initialising the class with the param 
    equation_of_state = 'v_ag'
    """

    def volume(self, pressure,temperature,params):
        """
        Returns volume [m^3] as a function of pressure [Pa] and temperature [K]
        """
        V_RT = self.__room_temperature_volume(pressure, params)
        eta=V_RT/params['V_0']
        volume=V_RT*np.exp(params['a_0'] \
                               * (temperature-params['T_0']) \
                               * np.exp(-params['delta_0']/params['kappa'] \
                                             *(1 - np.power(eta, params['kappa']))))
        return volume


    def pressure(self, temperature, volume, params):
        """
        Returns pressure [Pa] as a function of temperature [K] and volume[m^3]
        """
        P = optimize.fsolve(find_pressure, [1.e9], args=(volume, temperature, params))[0]
        return P
            
    def grueneisen_parameter(self, pressure, temperature, volume, params):
        """
        Returns grueneisen parameter [unitless] as a function of pressure,
        temperature, and volume.
        """
        alpha = self.thermal_expansivity (pressure, temperature, volume, params)
        K_T = self.isothermal_bulk_modulus (pressure, temperature, volume, params)
        C_V = self.heat_capacity_v( pressure, temperature, volume, params)
        return alpha * K_T * volume / C_V

    def isothermal_bulk_modulus(self, pressure,temperature,volume, params):
        """
        Returns isothermal bulk modulus [Pa] as a function of pressure [Pa],
        temperature [K], and volume [m^3].
        """
        dP = 10. # Pa
        V_minus = self.volume(pressure-dP,temperature,params)
        V = self.volume(pressure,temperature,params)
        V_plus = self.volume(pressure+dP,temperature,params)

        KT = -(2.*dP*V)/(V_plus - V_minus)
        return KT

    #calculate the shear modulus as a function of P, V, and T
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
        C_p=self.heat_capacity_p(pressure, temperature, volume, params)
        V=self.volume(pressure,temperature,params)
        alpha=self.thermal_expansivity(pressure, temperature, volume , params)
        K_T=self.isothermal_bulk_modulus(pressure,temperature,volume, params)
        return C_p - V*temperature*alpha*alpha*K_T


    def thermal_expansivity(self, pressure, temperature, volume , params):
        """
        Returns thermal expansivity at the pressure, temperature, and volume [1/K]
        """
        V = self.volume(pressure, params['T_0'], params)
        eta=V/params['V_0']
        alpha=params['a_0'] \
            * np.exp(-params['delta_0']/params['kappa'] \
                          * (1-np.power(eta, params['kappa'])))
        return alpha

    def adiabatic_bulk_modulus(self,pressure,temperature,volume,params):
        """
        Returns adiabatic bulk modulus [Pa] as a function of pressure [Pa],
        temperature [K], and volume [m^3].  
        """
        K_T= self.isothermal_bulk_modulus(pressure,temperature,volume,params)
        alpha = self.thermal_expansivity(pressure,temperature,volume,params)
        C_p = self.heat_capacity_p(pressure, temperature, volume, params)
        C_v = self.heat_capacity_v(pressure, temperature, volume, params)
        K_S = K_T*C_p/C_v
        return K_S

    def gibbs_free_energy(self, pressure,temperature, volume, params):
        """
        Returns the gibbs free energy [J/mol] as a function of pressure [Pa]
        and temperature [K].
        """
        intVdP = integrate.quad(lambda x: self.volume(x, temperature, params), \
                                    1.e5, pressure)[0]

        # Add order-disorder terms if required
        if params.has_key('landau_Tc'): # For a phase transition described by Landau term
            Gdisord=gibbs_disorder_Landau(pressure, temperature, params)
        else:
            if params.has_key('BW_deltaH'): # Add Bragg-Williams disordering
                Gdisord=gibbs_disorder_BW(pressure, temperature, params) - gibbs_disorder_BW(P_0, T_0, params)
            else:
                Gdisord=0.0

        if params.has_key('magnetic_moment'):
            Gmagnetic=self.__magnetic_gibbs(pressure, temperature, params)
        else:
            Gmagnetic=0.0

        return params['H_0'] + self.__intCpdT(temperature, params) - temperature*(params['S_0'] + self.__intCpoverTdT(temperature, params)) + intVdP + Gdisord + Gmagnetic


    def entropy(self,pressure,temperature, volume, params):
        """
        Returns the entropy [J/K/mol] as a function of pressure [Pa]
        and temperature [K].
        """
        dT = 0.1
        gibbs_minus=self.gibbs_free_energy(pressure,temperature-dT,volume, params)
        gibbs_plus=self.gibbs_free_energy(pressure,temperature+dT,volume, params)

        return -(gibbs_plus - gibbs_minus)/(2.*dT)



    def enthalpy(self, pressure, temperature, volume, params):
        """
        Returns the enthalpy [J/mol] as a function of pressure [Pa]
        and temperature [K].
        """
        gibbs=self.gibbs_free_energy(pressure,temperature,volume, params)
        entropy=self.entropy(pressure,temperature,volume, params)
        
        # Add order-disorder terms if required
        if params.has_key('landau_Tc'): # For a phase transition described by Landau term
            Hdisord=enthalpy_disorder_Landau(pressure, temperature, params)
        else:
            if params.has_key('BW_deltaH'): # Add Bragg-Williams disordering
                Hdisord=enthalpy_disorder_BW(pressure, temperature, params) - enthalpy_disorder_BW(P_0, T_0, params)
            else:
                Hdisord=0.0

        return gibbs + temperature*entropy + Hdisord


    def heat_capacity_p(self, pressure, temperature, volume, params):
        """
        Returns the heat capacity [J/K/mol] as a function of pressure [Pa]
        and temperature [K].
        """
        dT = 1.
        G_minus = self.gibbs_free_energy(pressure,temperature-dT, volume, params)
        G = self.gibbs_free_energy(pressure,temperature, volume, params)
        G_plus = self.gibbs_free_energy(pressure,temperature+dT, volume, params)

        C_p = -temperature * ( G_plus + G_minus - 2*G ) / (dT*dT)
        return C_p


    def __intCpdT (self, temperature, params):
        """
        Returns the thermal addition to the standard state enthalpy [J/mol]
        at ambient pressure [Pa]
        """
        return (params['Cp'][0]*temperature + 0.5*params['Cp'][1]*np.power(temperature,2.) - params['Cp'][2]/temperature + 2.*params['Cp'][3]*np.sqrt(temperature)) - (params['Cp'][0]*T_0 + 0.5*params['Cp'][1]*T_0*T_0 - params['Cp'][2]/T_0 + 2.0*params['Cp'][3]*np.sqrt(T_0))

    def __intCpoverTdT (self, temperature, params):
        """
        Returns the thermal addition to the standard state entropy [J/K/mol]
        at ambient pressure [Pa]
        """
        return (params['Cp'][0]*np.log(temperature) + params['Cp'][1]*temperature - 0.5*params['Cp'][2]/np.power(temperature,2.) - 2.0*params['Cp'][3]/np.sqrt(temperature)) - (params['Cp'][0]*np.log(T_0) + params['Cp'][1]*T_0 - 0.5*params['Cp'][2]/(T_0*T_0) - 2.0*params['Cp'][3]/np.sqrt(T_0))

    def __magnetic_gibbs(self, pressure, temperature, params):
        """
        Returns the magnetic contribution to the Gibbs free energy [J/mol]
        Expressions are those used by Chin, Hertzman and Sundman (1987)
        as reported in Sundman in the Journal of Phase Equilibria (1991)
        """

        structural_parameter=params['magnetic_structural_parameter']
        tau=temperature/(params['curie_temperature'][0] + pressure*params['curie_temperature'][1])
        magnetic_moment=params['magnetic_moment'][0] + pressure*params['magnetic_moment'][1]

        A = (518./1125.) + (11692./15975.)*((1./structural_parameter) - 1.)
        if tau < 1: 
            f=1.-(1./A)*(79./(140.*structural_parameter*tau) + (474./497.)*(1./structural_parameter - 1.)*(np.power(tau, 3.)/6. + np.power(tau, 9.)/135. + np.power(tau, 15.)/600.))
        else:
            f=-(1./A)*(np.power(tau,-5)/10. + np.power(tau,-15)/315. + np.power(tau, -25)/1500.)
        return constants.gas_constant*temperature*np.log(magnetic_moment + 1.)*f
        
    def __room_temperature_pressure(self, volume, params):
        x = np.power(volume/params['V_0'], 1./3.)
        P = 3*params['K_0'] / x / x \
            *(1.-x)*np.exp(1.5*(params['Kprime_0'] - 1.)*(1.-x))
        return P

    def __find_room_temperature_volume(self, volume, pressure, params):
        return pressure - self.__room_temperature_pressure(volume[0], params)

    def __find_pressure(self, pressure, volume, temperature, params):
        return volume - self.volume(pressure, temperature, params)

    def __room_temperature_volume(self, pressure, params):
        return optimize.fsolve(self.__find_room_temperature_volume, 0.2*params['V_0'], args=(pressure, params))[0]

    def validate_parameters(self, params):
        """
        Check for existence and validity of the parameters
        """

        #if G and Gprime are not included this is presumably deliberate,
        #as we can model density and bulk modulus just fine without them,
        #so just add them to the dictionary as nans
        if 'H_0' not in params:
            params['H_0'] = float('nan')
        if 'S_0' not in params:
            params['S_0'] = float('nan')
        if 'G_0' not in params:
            params['G_0'] = float('nan')
        if 'Gprime_0' not in params:
            params['Gprime_0'] = float('nan')
  
        #check that all the required keys are in the dictionary
        expected_keys = ['H_0', 'S_0', 'Cp', 'V_0', 'K_0', 'Kprime_0', 'a_0', 'delta_0']
        for k in expected_keys:
            if k not in params:
                raise KeyError('params object missing parameter : ' + k)
        

        if params['S_0'] is not float('nan') and params['S_0'] < 0.:
            warnings.warn( 'Unusual value for S_0', stacklevel=2 )
        if params['V_0'] < 1.e-7 or params['V_0'] > 1.e-2:
            warnings.warn( 'Unusual value for V_0', stacklevel=2 )


        if params['a_0'] < 0. or params['a_0'] > 1.e-3:
            warnings.warn( 'Unusual value for a_0', stacklevel=2 )
        if params['K_0'] < 1.e9 or params['K_0'] > 1.e13:
            warnings.warn( 'Unusual value for K_0', stacklevel=2 )
        if params['Kprime_0'] < 0. or params['Kprime_0'] > 10.:
            warnings.warn( 'Unusual value for Kprime_0', stacklevel=2 )

