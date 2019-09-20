from __future__ import absolute_import
# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2017 by the BurnMan team, released under the GNU
# GPL v2 or later.


import numpy as np
import warnings

from . import modified_tait as mt
from . import anisotropic_equation_of_state as aeos
from scipy.constants import Avogadro
from scipy.special import hyp2f1
from . import einstein

def Sijkl_derivs_to_coeffs(derivatives):
    invS0, invSprime, invSprime_inf, invSdprime, dinvSdT, theta, T0 = derivatives
    a = invS0
    c = invSprime_inf
    try:
        b = invSprime - invS0*invSdprime/(2*(invSprime - invSprime_inf))
        d = -invSdprime/(2*(invSprime - invSprime_inf))
    except ZeroDivisionError:
        b = invSprime
        d = 0.
    zeta = -dinvSdT*theta/(invSprime*xi(T0, theta))
    return [a, b, c, d, zeta, theta, T0]

def Sijkl_coeffs_to_derivs(coefficients):
    a, b, c, d, zeta, theta, T0 = coefficients
    invS0 = a
    invSprime = b - a*d
    invSprime_inf = c
    invSdprime = 2*d*((a*d - b) + c)
    dinvSdT = -zeta*(invSprime*xi(T0, theta))/theta
    return [invS0, invSprime, invSprime_inf, invSdprime, dinvSdT, theta, T0]

def Pthoverzeta(T, theta, T0):
    u = theta/T
    u0 = theta/T0
    #xi = u*u*np.exp(u)/np.power(np.exp(u) - 1., 2.)
    return (1./(np.exp(u) - 1.) - 1./(np.exp(u0) - 1.))

def xi(T, theta):
    u = theta/T
    return u*u*np.exp(u)/np.power(np.exp(u) - 1., 2.)

def dxidT(T, theta):
    u = theta/T
    return (u*u*u*np.exp(u)*((u - 2.)*np.exp(u) + (u+2.))
            / (theta*np.power(np.exp(u) - 1., 3.)))

def S_ijkl(P, T, a, b, c, d, zeta, theta, T0):
    Pth = zeta*Pthoverzeta(T, theta, T0)
    Peff = P - Pth
    return (1. + d*Peff)/(a + b*Peff + c*d*Peff*Peff)

def dPeffdT(P, T, a, b, c, d, zeta, theta, T0):
    return -zeta / theta * xi(T, theta)

def d2PeffdT2(P, T, a, b, c, d, zeta, theta, T0):
    return -zeta / theta * dxidT(T, theta)

def a_ijkl(P, T, a, b, c, d, zeta, theta, T0):
    return -(S_ijkl(P, T, a, b, c, d, zeta, theta, T0)
             * dPeffdT(P, T, a, b, c, d, zeta, theta, T0))

def expintS(P, T, a, b, c, d, zeta, theta, T0):
    Pth = zeta*Pthoverzeta(T, theta, T0)
    Peff = P - Pth
    if np.abs(d) < 1.e-25:
        return np.power(1. + b*Peff/a, -1./b)
    else:
        f = (b + np.sqrt(b*b - 4.*a*c*d))/2.
        ai = c*d/f
        bi = (1. - f/c)/(f - a*c*d/f)
        ci = f/a
        di = (a*d/f - 1.)/(f - a*c*d/f)
        return np.power(1. + ai*Peff, bi) * np.power(1. + ci*Peff, di)

def intVoverV0dP(P1, P0, T, a, b, c, d, zeta, theta, T0):
    """
    Returns the integral of V/V0 between P0 and P1
    """
    f = (b + np.sqrt(b*b - 4.*a*c*d))/2.
    ai = c*d/f
    bi = (1. - f/c)/(f - a*c*d/f)
    ci = f/a
    di = (a*d/f - 1.)/(f - a*c*d/f)

    Pth = zeta*Pthoverzeta(T, theta, T0)
    Peff0 = P0 - Pth
    Peff1 = P1 - Pth
    g0 = np.power(1. + ai*Peff0, bi) * np.power(1. + ci*Peff0, di+1.)
    g1 = np.power(1. + ai*Peff1, bi) * np.power(1. + ci*Peff1, di+1.)
    int0 = (g0 * hyp2f1(-bi, 1. + di, 2. + di, (ai + ai*ci*Peff0)/(ai - ci)) /
            (ci*(1. + di)*np.power(-(ci*(1. + ai*Peff0)/(ai - ci)), bi)))
    int1 = (g1 * hyp2f1(-bi, 1. + di, 2. + di, (ai + ai*ci*Peff1)/(ai - ci)) /
            (ci*(1. + di)*np.power(-(ci*(1. + ai*Peff1)/(ai - ci)), bi)))
    return int1 - int0

class AEOS(aeos.AnisotropicEquationOfState):

    """
    Class for the Myhill (2019) anisotropic equation of state
    """

    def unit_cell_volume(self, pressure, temperature, params):
        return (expintS(pressure, temperature,
                        *params['beta_T_reuss']['coeffs'])
                * params['unit_cell_volume'])

    def molar_volume(self, pressure, temperature, params):
        return (expintS(pressure, temperature,
                        *params['beta_T_reuss']['coeffs'])
                * params['molar_volume'])

    def molar_gibbs(self, pressure, temperature, params):
        P_0 = 0.
        return (intVoverV0dP(pressure, P_0, temperature,
                             *params['beta_T_reuss']['coeffs'])
                * params['molar_volume'])

    def molar_entropy(self, pressure, temperature, params):
        # S0 - [V1 - V0]*dPeff/dT
        P_0 = 0.
        V_0 = self.molar_volume(P_0, temperature, params)
        V_1 = self.molar_volume(pressure, temperature, params)
        dpeffdT = dPeffdT(pressure, temperature, *params['beta_T_reuss']['coeffs'])
        return (V_0 - V_1) * dpeffdT

    def isothermal_compressibility_reuss(self, pressure, temperature, params):
        return S_ijkl(pressure, temperature, *params['beta_T_reuss']['coeffs'])

    def molar_heat_capacity_p(self, pressure, temperature, params):
        # Cp0 + [dVdPeff1 - dVdPeff0]*(dPeff/dT*dPeff/dT)
        P_0 = 0.
        V_0 = self.molar_volume(P_0, temperature, params)
        V_1 = self.molar_volume(pressure, temperature, params)
        beta_0 = self.isothermal_compressibility_reuss(P_0, temperature, params)
        beta_1 = self.isothermal_compressibility_reuss(pressure, temperature, params)
        dpeffdT = dPeffdT(pressure, temperature, *params['beta_T_reuss']['coeffs'])
        d2peffdT2 = d2PeffdT2(pressure, temperature, *params['beta_T_reuss']['coeffs'])
        d2GdT2a = (V_0*beta_0 - V_1*beta_1) * np.power(dpeffdT, 2.)
        d2GdT2b = (V_0 - V_1) * d2peffdT2
        return (d2GdT2b - d2GdT2a) * temperature

    def thermal_expansivity_tensor(self, pressure, temperature, params):
        alpha = np.zeros((6))
        alpha[2] = a_ijkl(pressure, temperature, *params['E_3']['coeffs'])
        alpha[1] = a_ijkl(pressure, temperature, *params['E_2']['coeffs'])
        alpha[0] = (a_ijkl(pressure, temperature, *params['beta_T_reuss']['coeffs'])
                    - alpha[1] - alpha[2])

        for i in [3, 4, 5]:
            for j in range(3): # we only need the elements S_ijkk under hydrostatic stress
                    try:
                        alpha[i] += a_ijkl(pressure, temperature, *params['S_T']['{0}{1}'.format(j+1, i+1)]['coeffs'])
                    except KeyError:
                        pass
        return alpha

    def isothermal_elastic_compliance_tensor(self, pressure, temperature, params):
        ST = np.zeros((6, 6))

        ST[0,0] = (S_ijkl(pressure, temperature, *params['beta_T_reuss']['coeffs'])
                   - S_ijkl(pressure, temperature, *params['E_2']['coeffs'])
                   - S_ijkl(pressure, temperature, *params['E_3']['coeffs'])
                   - S_ijkl(pressure, temperature, *params['S_T']['12']['coeffs'])
                   - S_ijkl(pressure, temperature, *params['S_T']['13']['coeffs']))

        ST[1,1] = (S_ijkl(pressure, temperature, *params['E_2']['coeffs'])
                   - S_ijkl(pressure, temperature, *params['S_T']['12']['coeffs'])
                   - S_ijkl(pressure, temperature, *params['S_T']['23']['coeffs']))
        ST[2,2] = (S_ijkl(pressure, temperature, *params['E_3']['coeffs'])
                   - S_ijkl(pressure, temperature, *params['S_T']['13']['coeffs'])
                   - S_ijkl(pressure, temperature, *params['S_T']['23']['coeffs']))

        for i in [3, 4, 5]:
            ST[i,i] = S_ijkl(pressure, temperature, *params['S_T']['{0}{1}'.format(i+1, i+1)]['coeffs'])

        for i in range(6):
            for j in range(i+1, 6):
                try:
                    ST[i,j] = S_ijkl(pressure, temperature,
                                     *params['S_T']['{0}{1}'.format(i+1,
                                                                    j+1)]['coeffs'])
                    ST[j,i] = ST[i,j]
                except KeyError:
                    pass
        return ST

    def isothermal_elastic_stiffness_tensor(self, pressure, temperature, params):
        ST = self.isothermal_elastic_compliance_tensor(pressure, temperature, params)
        return np.linalg.pinv(ST)

    def isothermal_bulk_modulus_reuss(self, pressure, temperature, params):
        beta_R = self.isothermal_compressibility_reuss(pressure,
                                                       temperature, params)
        return 1./beta_R

    def adiabatic_elastic_compliance_tensor(self, pressure, temperature, params):
        ST = self.isothermal_elastic_compliance_tensor(pressure, temperature, params)
        alpha = self.thermal_expansivity_tensor(pressure, temperature, params)
        molar_volume = self.molar_volume(pressure, temperature, params)
        molar_Cp = self.molar_heat_capacity_p(pressure, temperature, params)
        SN = ST - (np.einsum('i, j -> ij', alpha, alpha)
                   * molar_volume * temperature
                   / molar_Cp)
        return SN

    def adiabatic_elastic_stiffness_tensor(self, pressure, temperature, params):
        SN = self.isothermal_elastic_compliance_tensor(pressure, temperature, params)
        return np.linalg.pinv(SN)

    def volumetric_thermal_expansivity(self, pressure, temperature, params):
        alpha = self.thermal_expansivity_tensor(pressure, temperature, params)
        return sum(alpha[0:3])

    def validate_and_parse_parameters(self, params):
        """
        Check for existence and validity of the parameters and parse them
        """

        # Now check all the required keys for the
        # thermal part of the EoS are in the dictionary

        if params['symmetry_type'] == 'cubic':
            from copy import deepcopy
            E = deepcopy(params['beta_T_reuss'])
            for key in list(E.keys()):
                if key is not 'Theta':
                    E[key] *= 3.

            params['E_2'] = E
            params['E_3'] = E

        expected_keys = ['symmetry_type', 'unit_cell_vectors', 'Z',
                         'beta_T_reuss', 'E_2', 'E_3',
                         'S_T', 'T_0', 'n', 'molar_mass']
        for k in expected_keys:
            if k not in params:
                raise KeyError('params object missing key: ' + k)

        params['unit_cell_volume'] = np.linalg.det(np.array(params['unit_cell_vectors']))
        params['molar_volume'] = params['unit_cell_volume'] * Avogadro / params['Z']

        required_compliance_keys = ['12', '44']
        for k in required_compliance_keys:
            if k not in params['S_T']:
                raise KeyError('params object S_T missing key: ' + k)

        # Convert Sij derivatives into the static parameters
        p = params['beta_T_reuss']
        S_prms = [p['invS_0'], p['invSprime_0'], p['invSprime_inf'],
                  p['invSdprime_0'], p['dinvSdT_0'], p['Theta'], params['T_0']]
        params['beta_T_reuss']['coeffs'] = Sijkl_derivs_to_coeffs(S_prms)


        p = params['E_2']
        S_prms = [p['invS_0'], p['invSprime_0'], p['invSprime_inf'],
                  p['invSdprime_0'], p['dinvSdT_0'], p['Theta'], params['T_0']]
        params['E_2']['coeffs'] = Sijkl_derivs_to_coeffs(S_prms)

        p = params['E_3']
        S_prms = [p['invS_0'], p['invSprime_0'], p['invSprime_inf'],
                  p['invSdprime_0'], p['dinvSdT_0'], p['Theta'], params['T_0']]
        params['E_3']['coeffs'] = Sijkl_derivs_to_coeffs(S_prms)

        for key in params['S_T']:
            p = params['S_T'][key]
            S_prms = [p['invS_0'], p['invSprime_0'], p['invSprime_inf'],
                      p['invSdprime_0'], p['dinvSdT_0'], p['Theta'], params['T_0']]
            params['S_T'][key]['coeffs'] = Sijkl_derivs_to_coeffs(S_prms)

        # Fill in missing terms
        if params['symmetry_type'] == 'cubic':
            cubic_correspondences = [['12', ['13', '23']],
                                     ['44', ['55', '66']]]
            for parent, children in cubic_correspondences:
                for child in children:
                    if child not in params['S_T']:
                        params['S_T'][child] = params['S_T'][parent]

        # Check for a reasonable number of atoms and molar mass
        if params['n'] < 1. or params['n'] > 1000.:
            warnings.warn('Unusual value for n', stacklevel=2)
        if params['molar_mass'] < 0.001 or params['molar_mass'] > 10.:
            warnings.warn('Unusual value for molar_mass', stacklevel=2)
