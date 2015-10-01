# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

import numpy as np
import scipy.optimize as opt
import warnings

import birch_murnaghan as bm
import burnman.debye as debye
import equation_of_state as eos

def _debye_temperature_fast(x,gruen_0,debye_0, q_0):
    f = 1./2. * (pow(x, 2./3.) - 1.)
    a1_ii = 6. * gruen_0 # EQ 47
    a2_iikk = -12.*gruen_0+36.* gruen_0 * gruen_0 - 18.*q_0*gruen_0 # EQ 47
    return debye_0 * np.sqrt(1. + a1_ii * f + 1./2. * a2_iikk*f*f)

def _grueneisen_parameter_fast(V_0, volume, gruen_0, q_0):
    x = V_0 / volume
    f = 1./2. * (pow(x, 2./3.) - 1.)
    a1_ii = 6. * gruen_0 # EQ 47
    a2_iikk = -12.*gruen_0 + 36.*gruen_0*gruen_0 - 18.*q_0*gruen_0 # EQ 47
    nu_o_nu0_sq = 1.+ a1_ii*f + (1./2.)*a2_iikk * f*f # EQ 41
    return 1./6./nu_o_nu0_sq * (2.*f+1.) * ( a1_ii + a2_iikk*f )

def _volume_opt_func(x, temperature, T_0, b_iikk, b_iikkmm, V_0, pressure, q_0, gruen_0, n, debye_0):
    debye_T = _debye_temperature_fast(V_0/x,gruen_0,debye_0, q_0)
    gr = _grueneisen_parameter_fast(V_0, x, gruen_0, q_0)
    
    E_th =  debye.thermal_energy(temperature, debye_T, n) #thermal energy at temperature T
    E_th_ref = debye.thermal_energy(T_0, debye_T, n) #thermal energy at reference temperature
        
    fx = 0.5*(pow(V_0/x,2./3.)-1.) # EQ 24
    ret = (1./3.)*(pow(1.+2.*fx,5./2.))*((b_iikk*fx) \
                                             +(0.5*b_iikkmm*fx*fx)) + gr*(E_th - E_th_ref)/x - pressure #EQ 21
    return ret

class SLBBase(eos.EquationOfState):
    """
    Base class for the finite strain-Mie-Grueneiesen-Debye equation of state detailed
    in :cite:`Stixrude2005`.  For the most part the equations are
    all third order in strain, but see further the :class:`burnman.slb.SLB2` and 
    :class:`burnman.slb.SLB3` classes.
    """

    def __debye_temperature(self,x,params):
        """
        Finite strain approximation for Debye Temperature [K]
        x = ref_vol/vol
        """
        f = 1./2. * (pow(x, 2./3.) - 1.)
        a1_ii = 6. * params['grueneisen_0'] # EQ 47
        a2_iikk = -12.*params['grueneisen_0']+36.*pow(params['grueneisen_0'],2.) - 18.*params['q_0']*params['grueneisen_0'] # EQ 47
        return params['Debye_0'] * np.sqrt(1. + a1_ii * f + 1./2. * a2_iikk*f*f)

    def _volume_dependent_q(self, mineral):
        """
        Finite strain approximation for :math:`q`, the isotropic volume strain
        derivative of the grueneisen parameter.
        """
        if 'q' not in mineral.eos_data:
            x = mineral.params['V_0'] / mineral.molar_volume()
            grueneisen_0 = mineral.params['grueneisen_0']
            q_0 = mineral.params['q_0']
            f = 1./2. * (pow(x, 2./3.) - 1.)
            a1_ii = 6. * grueneisen_0 # EQ 47
            a2_iikk = -12.*grueneisen_0+36.*pow(grueneisen_0, 2.) - 18.*q_0*grueneisen_0 # EQ 47
            nu_o_nu0_sq = 1.+ a1_ii*f + (1./2.)*a2_iikk * f*f # EQ 41
            gr = mineral.grueneisen_parameter()
            mineral.eos_data['q'] = 1./9.*(18.*gr - 6. - 1./2. / nu_o_nu0_sq * (2.*f+1.)*(2.*f+1.)*a2_iikk/gr)
        return mineral.eos_data['q']


    def _debye_T(self, mineral):
        if 'debye_T' not in mineral.eos_data:
            grueneisen_0 = mineral.params['grueneisen_0']
            q_0 = mineral.params['q_0']
            Debye_0 = mineral.params['Debye_0']
            mineral.eos_data['debye_T'] = _debye_temperature_fast(mineral.params['V_0']/mineral.molar_volume(),
                                                                  grueneisen_0, Debye_0, q_0)
        return mineral.eos_data['debye_T']

    def _isotropic_eta_s(self, mineral):
        """
        Finite strain approximation for :math:`eta_{s0}`, the isotropic shear
        strain derivative of the grueneisen parameter.
        """
        if 'eta_s' not in mineral.eos_data:
            x = mineral.params['V_0']/mineral.molar_volume()
            f = 1./2. * (pow(x, 2./3.) - 1.)
            grueneisen_0 = mineral.params['grueneisen_0']
            q_0 = mineral.params['q_0']
            a2_s = -2.*grueneisen_0 - 2.*mineral.params['eta_s_0'] # EQ 47
            a1_ii = 6. * grueneisen_0 # EQ 47
            a2_iikk = -12.*grueneisen_0+36.*pow(grueneisen_0, 2.) - 18.*q_0*grueneisen_0 # EQ 47
            nu_o_nu0_sq = 1.+ a1_ii*f + (1./2.)*a2_iikk * pow(f,2.) # EQ 41
            gr = mineral.grueneisen_parameter()
            mineral.eos_data['eta_s'] = - gr - (1./2. * pow(nu_o_nu0_sq, -1.) * pow((2.*f)+1., 2.)*a2_s) # EQ 46 NOTE the typo from Stixrude 2005
        return mineral.eos_data['eta_s']

    def pressure(self, temperature, volume, params):
        return bm.birch_murnaghan(params['V_0']/volume, params) + \
                self.__thermal_pressure(temperature,volume, params) - \
                self.__thermal_pressure(300.,volume, params)

    #calculate isotropic thermal pressure, see
    # Matas et. al. (2007) eq B4
    def __thermal_pressure(self, T, V, params):
        Debye_T = self.__debye_temperature(params['V_0']/V, params)
        gr = self.__grueneisen_parameter(T, V, params)
        P_th = gr * debye.thermal_energy(T,Debye_T, params['n'])/V
        return P_th

    def _volume_opt_func2(self, x, temperature, T_0, b_iikk, b_iikkmm, V_0, pressure, q_0, gruen_0, n, debye_0):
        debye_T = self.__debye_temperature_fast(V_0/x,gruen_0,debye_0, q_0)
        gr = self._grueneisen_parameter_fast(V_0, x, gruen_0, q_0)

        E_th =  debye.thermal_energy(temperature, debye_T, n) #thermal energy at temperature T
        E_th_ref = debye.thermal_energy(T_0, debye_T, n) #thermal energy at reference temperature
        
        fx = 0.5*(pow(V_0/x,2./3.)-1.) # EQ 24
        ret = (1./3.)*(pow(1.+2.*fx,5./2.))*((b_iikk*fx) \
                                             +(0.5*b_iikkmm*fx*fx)) + gr*(E_th - E_th_ref)/x - pressure #EQ 21
        return ret*ret

    def volume(self, mineral):
        """
        Returns molar volume. :math:`[m^3]`
        """

        b_iikk= 9.*mineral.params['K_0'] # EQ 28
        b_iikkmm= 27.*mineral.params['K_0']*(mineral.params['Kprime_0']-4.) # EQ 29
        f = lambda x: 0.5*(pow(mineral.params['V_0']/x,2./3.)-1.) # EQ 24

        n = mineral.params['n']
        gruen_0 = mineral.params['grueneisen_0']
        V_0 = mineral.params['V_0']
        q_0 = mineral.params['q_0']
        debye_0 = mineral.params['Debye_0']
        T_0 = mineral.params['T_0']

        func = lambda x: _volume_opt_func(x, mineral.temperature, T_0, b_iikk, b_iikkmm, V_0, mineral.pressure, q_0, gruen_0, n, debye_0)
        func2 = lambda x: pow(self._volume_opt_func2(x, mineral.temperature, T_0, b_iikk, b_iikkmm, V_0, mineral.pressure, q_0, gruen_0, n, debye_0),2.)

        # we need to have a sign change in [a,b] to find a zero. Let us start with a
        # conservative guess:
        a = 0.6*mineral.params['V_0']
        b = 1.2*mineral.params['V_0']

        # if we have a sign change, we are done:
        if func(a)*func(b)<0:
            return opt.brentq(func, a, b)
        else:
            tol = 0.0001
            sol = opt.fmin(func2, 1.0*mineral.params['V_0'], ftol=tol, full_output=1, disp=0)
            if sol[1] > tol*2:
                raise ValueError('Cannot find volume, likely outside of the range of validity for EOS')
            else:
                warnings.warn("May be outside the range of validity for EOS")
                return sol[0]

    def pressure( self, temperature, volume, params):
        """
        Returns the pressure of the mineral at a given temperature and volume [Pa]
        """
        debye_T = self.__debye_temperature(params['V_0']/volume, params)
        gr = self.grueneisen_parameter(0.0, temperature, volume, params) #does not depend on pressure
        E_th = debye.thermal_energy(temperature, debye_T, params['n'])
        E_th_ref = debye.thermal_energy(300., debye_T, params['n']) #thermal energy at reference temperature

        b_iikk= 9.*params['K_0'] # EQ 28
        b_iikkmm= 27.*params['K_0']*(params['Kprime_0']-4.) # EQ 29
        f = 0.5*(pow(params['V_0']/volume,2./3.)-1.) # EQ 24
        P = (1./3.)*(pow(1.+2.*f,5./2.))*((b_iikk*f) \
            +(0.5*b_iikkmm*pow(f,2.))) + gr*(E_th - E_th_ref)/volume #EQ 21

        return P

    def grueneisen_parameter(self, mineral):
        gruen_0 = mineral.params['grueneisen_0']
        V_0 = mineral.params['V_0']
        q_0 = mineral.params['q_0']
        return _grueneisen_parameter_fast(V_0, mineral.V, gruen_0, q_0)


    def isothermal_bulk_modulus(self, mineral):
        """
        Returns isothermal bulk modulus :math:`[Pa]` 
        """
        temperature = mineral.temperature
        volume = mineral.molar_volume()
        T_0 = mineral.params['T_0']
        gruen_0 = mineral.params['grueneisen_0']
        V_0 = mineral.params['V_0']
        q_0 = mineral.params['q_0']
        Debye_0 = mineral.params['Debye_0']
        n = mineral.params['n']

        debye_T = self._debye_T(mineral)
        gr = mineral.grueneisen_parameter()

        E_th = debye.thermal_energy(temperature, debye_T, n) #thermal energy at temperature T
        E_th_ref = debye.thermal_energy(T_0, debye_T, n) #thermal energy at reference temperature

        C_v = debye.heat_capacity_v(temperature, debye_T, n) #heat capacity at temperature T
        C_v_ref = debye.heat_capacity_v(T_0, debye_T, n) #heat capacity at reference temperature

        q = self._volume_dependent_q(mineral)

        K = bm.bulk_modulus(volume, mineral.params) \
            + (gr + 1.-q)* ( gr / volume ) * (E_th - E_th_ref) \
            - ( pow(gr , 2.) / volume )*(C_v*temperature - C_v_ref*T_0)

        return K

    def adiabatic_bulk_modulus(self, mineral):
        """
        Returns adiabatic bulk modulus. :math:`[Pa]` 
        """
        K_T = mineral.isothermal_bulk_modulus()
        alpha = mineral.thermal_expansivity()
        gr = mineral.grueneisen_parameter()
        K_S = K_T*(1. + gr * alpha * mineral.temperature)
        return K_S

    def shear_modulus(self, mineral):
        """
        Returns shear modulus. :math:`[Pa]` 
        """
        volume = mineral.molar_volume()
        T_0 = mineral.params['T_0']
        debye_T = self._debye_T(mineral)
        n = mineral.params['n']

        eta_s = self._isotropic_eta_s(mineral) # TODO

        E_th = debye.thermal_energy(mineral.temperature, debye_T, n)
        E_th_ref = debye.thermal_energy(T_0, debye_T, n)

        if self.order==2:
            return bm.shear_modulus_second_order(volume, mineral.params) - eta_s * (E_th-E_th_ref) / volume
        elif self.order==3:
            return bm.shear_modulus_third_order(volume, mineral.params) - eta_s * (E_th-E_th_ref) / volume
        else:
            raise NotImplementedError("")

    def heat_capacity_v(self, mineral):
        """
        Returns heat capacity at constant volume. :math:`[J/K/mol]` 
        """
        debye_T = self._debye_T(mineral)
        return debye.heat_capacity_v(mineral.temperature, debye_T, mineral.params['n'])

    def heat_capacity_p(self, mineral):
        """
        Returns heat capacity at constant pressure. :math:`[J/K/mol]` 
        """
        alpha = mineral.thermal_expansivity()
        gr = mineral.grueneisen_parameter()
        C_v = mineral.heat_capacity_v()
        C_p = C_v*(1. + gr * alpha * mineral.temperature)
        return C_p

    def thermal_expansivity(self, mineral):
        """
        Returns thermal expansivity. :math:`[1/K]` 
        """
        C_v = mineral.heat_capacity_v()
        gr = mineral.grueneisen_parameter()
        K = mineral.isothermal_bulk_modulus()
        alpha = gr * C_v / K / mineral.molar_volume()
        return alpha

    def gibbs_free_energy(self, mineral):
        """
        Returns the Gibbs free energy at the pressure and temperature of the mineral [J/mol]
        """
        F = mineral.molar_helmholtz(mineral)
        G = F + mineral.pressure * mineral.molar_volume()
        return G

    def entropy( self, mineral):
        """
        Returns the entropy at the pressure and temperature of the mineral [J/K/mol]
        """
        Debye_T = self._debye_T(mineral)
        S = debye.entropy(mineral.temperature, Debye_T, mineral.params['n'] )
        return S 

    def enthalpy(self, mineral):
        """
        Returns the enthalpy at the pressure and temperature of the mineral [J/mol]
        """
        F = mineral.molar_helmholtz(mineral)
        entropy = mineral.molar_entropy()
        return F + mineral.temperature * entropy

    def helmholtz_free_energy(self, mineral):
        """
        Returns the Helmholtz free energy at the pressure and temperature of the mineral [J/mol]
        """
        x = mineral.params['V_0'] / mineral.molar_volume()
        f = 1./2. * (pow(x, 2./3.) - 1.)
        Debye_T = self._debye_T(mineral)
        F_0 = mineral.params['F_0']
        V_0 = mineral.params['V_0']
        K_0 = mineral.params['K_0']
        Kprime_0 = mineral.params['Kprime_0']
        n = mineral.params['n']

        F_quasiharmonic = debye.helmholtz_free_energy(mineral.temperature, Debye_T, n ) - \
                          debye.helmholtz_free_energy( 300., Debye_T, n )

        b_iikk= 9.*K_0 # EQ 28
        b_iikkmm= 27.*K_0*(Kprime_0-4.) # EQ 29

        F = F_0 + 0.5*b_iikk*f*f*V_0 \
            + (1./6.)*V_0*b_iikkmm*f*f*f + F_quasiharmonic

        return F

    def validate_parameters(self, params):
        """
        Check for existence and validity of the parameters
        """
        if 'T_0' not in params:
            params['T_0'] = 300.
        if 'P_0' not in params:
            params['P_0'] = 1.e5

        #if F, G and Gprime are not included this is presumably deliberate,
        #as we can model density and bulk modulus just fine without them,
        #so just add them to the dictionary as nans
        if 'F_0' not in params:
            params['F_0'] = float('nan')
        if 'G_0' not in params:
            params['G_0'] = float('nan')
        if 'Gprime_0' not in params:
            params['Gprime_0'] = float('nan')
        if 'eta_s_0' not in params:
            params['eta_s_0'] = float('nan')
  
        #check that all the required keys are in the dictionary
        expected_keys = ['V_0', 'K_0', 'Kprime_0', 'G_0', 'Gprime_0', 'molar_mass', 'n', 'Debye_0', 'grueneisen_0', 'q_0', 'eta_s_0']
        for k in expected_keys:
            if k not in params:
                raise KeyError('params object missing parameter : ' + k)
        
        #now check that the values are reasonable.  I mostly just
        #made up these values from experience, and we are only 
        #raising a warning.  Better way to do this? [IR]
        if params['V_0'] < 1.e-7 or params['V_0'] > 1.e-3:
            warnings.warn( 'Unusual value for V_0', stacklevel=2 )
        if params['K_0'] < 1.e9 or params['K_0'] > 1.e13:
            warnings.warn( 'Unusual value for K_0', stacklevel=2 )
        if params['Kprime_0'] < 0. or params['Kprime_0'] > 10.:
            warnings.warn( 'Unusual value for Kprime_0', stacklevel=2 )
        if params['G_0'] < 0. or params['G_0'] > 1.e13:
            warnings.warn( 'Unusual value for G_0', stacklevel=2 )
        if params['Gprime_0'] < 0. or params['Gprime_0'] > 10.:
            warnings.warn( 'Unusual value for Gprime_0', stacklevel=2 )
        if params['molar_mass'] < 0.001 or params['molar_mass'] > 1.:
            warnings.warn( 'Unusual value for molar_mass', stacklevel=2 )
        if params['n'] < 1. or params['n'] > 100. or not float(params['n']).is_integer():
            warnings.warn( 'Unusual value for n', stacklevel=2 )
        if params['Debye_0'] < 1. or params['Debye_0'] > 10000.:
            warnings.warn( 'Unusual value for Debye_0', stacklevel=2 )
        if params['grueneisen_0'] < 0. or params['grueneisen_0'] > 10.:
            warnings.warn( 'Unusual value for grueneisen_0' , stacklevel=2)
        if params['q_0'] < -10. or params['q_0'] > 10.:
            warnings.warn( 'Unusual value for q_0' , stacklevel=2)
        if params['eta_s_0'] < -10. or params['eta_s_0'] > 10.:
            warnings.warn( 'Unusual value for eta_s_0' , stacklevel=2)
            


class SLB3(SLBBase):
    """
    SLB equation of state with third order finite strain expansion for the
    shear modulus (this should be preferred, as it is more thermodynamically
    consistent.
    """
    def __init__(self):
        self.order=3


class SLB2(SLBBase):
    """
    SLB equation of state with second order finite strain expansion for the
    shear modulus.  In general, this should not be used, but sometimes
    shear modulus data is fit to a second order equation of state.  In that
    case, you should use this.  The moral is, be careful!
    """
    def __init__(self):
        self.order=2
