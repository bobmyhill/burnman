from __future__ import absolute_import
# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit
# for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2019 by the BurnMan team, released under the GNU
# GPL v2 or later.

import numpy as np
import warnings

from . import einstein
from . import anisotropic_equation_of_state as anisotropiceos
from scipy.constants import Avogadro
from scipy.special import hyp2f1
from scipy.optimize import brentq
from ..tools import bracket

try:
    from numba import jit
except ImportError:
    def jit(fn):
        return fn


def Sijkl_derivs_to_coeffs(derivative_dict):
    dct = derivative_dict
    a = dct['invS_0']
    c = dct['invSprime_inf']
    try:
        b = dct['invSprime_0'] - (dct['invS_0'] * dct['invSdprime_0']
                                  / (2.*(dct['invSprime_0']
                                         - dct['invSprime_inf'])))
        d = -dct['invSdprime_0'] / (2*(dct['invSprime_0']
                                       - dct['invSprime_inf']))
    except ZeroDivisionError:
        b = dct['invSprime_0']
        d = 0.
    return [a, b, c, d, dct['grueneisen_0'], dct['q_0']]


def Sijkl_coeffs_to_derivs(coefficients):
    a, b, c, d, gr, q = coefficients
    invS0 = a
    invSprime = b - a*d
    invSprime_inf = c
    invSdprime = 2*d*((a*d - b) + c)
    return {'invS_0': invS0,
            'invSprime_0': invSprime,
            'invSprime_inf': invSprime_inf,
            'invSdprime_0': invSdprime,
            'grueneisen_0': gr,
            'q_0': q}


@jit
def S_ijkl_Tref(P, a, b, c, d):
    return (1. + d*P) / (a + b*P + c*d*P*P)


@jit
def xioverxi0_Tref(P, a, b, c, d):
    if np.abs(d) < 1.e-20:
        return np.power(1. + b*P/a, -1./b)
    else:
        f = (b + np.sqrt(b*b - 4.*a*c*d))/2.
        ai = c*d/f
        bi = (1. - f/c)/(f - a*c*d/f)
        ci = f/a
        di = (a*d/f - 1.)/(f - a*c*d/f)
        return np.power(1. + ai*P, bi) * np.power(1. + ci*P, di)


@jit
def intVoverV0dP_Tref(P1, P0, a, b, c, d):
    """
    Returns the integral of V/V0 between P0 and P1 at the reference temperature
    """
    f = (b + np.sqrt(b*b - 4.*a*c*d))/2.
    ai = c*d/f
    bi = (1. - f/c)/(f - a*c*d/f)
    ci = f/a
    di = (a*d/f - 1.)/(f - a*c*d/f)

    g0 = np.power(1. + ai*P0, bi) * np.power(1. + ci*P0, di+1.)
    g1 = np.power(1. + ai*P1, bi) * np.power(1. + ci*P1, di+1.)
    int0 = (g0 * hyp2f1(-bi, 1. + di, 2. + di, (ai + ai*ci*P0)/(ai - ci))
            / (ci*(1. + di)*np.power(-(ci*(1. + ai*P0)/(ai - ci)), bi)))
    int1 = (g1 * hyp2f1(-bi, 1. + di, 2. + di, (ai + ai*ci*P1)/(ai - ci))
            / (ci*(1. + di)*np.power(-(ci*(1. + ai*P1)/(ai - ci)), bi)))
    return int1 - int0


@jit
def _grueneisen_parameter_fast(VoverV0, grueneisen_0, q_0):
    """global function with plain parameters so jit will work"""
    x = 1. / VoverV0
    f = 1. / 2. * (pow(x, 2. / 3.) - 1.)
    a1_ii = 6. * grueneisen_0  # EQ 47
    a2_iikk = -12. * grueneisen_0 + 36. * \
        grueneisen_0 * grueneisen_0 - 18. * q_0 * grueneisen_0  # EQ 47
    nu_o_nu0_sq = 1. + a1_ii * f + (1. / 2.) * a2_iikk * f * f  # EQ 41
    return 1. / 6. / nu_o_nu0_sq * (2. * f + 1.) * (a1_ii + a2_iikk * f)


@jit
def _theta_fast(VoverV0, grueneisen_0, q_0, theta_0):
    """global function with plain parameters so jit will work"""
    x = 1. / VoverV0
    f = 1. / 2. * (pow(x, 2. / 3.) - 1.)
    a1_ii = 6. * grueneisen_0  # EQ 47
    a2_iikk = -12. * grueneisen_0 + 36. * \
        grueneisen_0 * grueneisen_0 - 18. * q_0 * grueneisen_0  # EQ 47
    nu_o_nu0_sq = 1. + a1_ii * f + (1. / 2.) * a2_iikk * f * f  # EQ 41
    # if nu_o_nu0_sq < 1.:
    #        raise Exception('At a volume of {0} V_0, nu/nu_0 is complex'.format(1./x, nu_o_nu0_sq))
    return np.sqrt(nu_o_nu0_sq) * theta_0


@jit
def _volume_dependent_q_fast(VoverV0, grueneisen_0, q_0):
    """
    Finite strain approximation for :math:`q`, the isotropic volume strain
    derivative of the grueneisen parameter.
    """
    x = 1. / VoverV0
    f = 1. / 2. * (pow(x, 2. / 3.) - 1.)
    a1_ii = 6. * grueneisen_0  # EQ 47
    a2_iikk = -12. * grueneisen_0 + 36. * pow(
        grueneisen_0, 2.) - 18. * q_0 * grueneisen_0  # EQ 47
    nu_o_nu0_sq = 1. + a1_ii * f + (1. / 2.) * a2_iikk * f * f  # EQ 41
    gr = 1. / 6. / nu_o_nu0_sq * (2. * f + 1.) * (a1_ii + a2_iikk * f)
    if np.abs(grueneisen_0) < 1.e-10:  # avoids divide by zero if gr_0 = 0.
        q = 1. / 9. * (18. * gr - 6.)
    else:
        q = 1. / 9. * \
            (18. * gr - 6. - 1. / 2. / nu_o_nu0_sq
             * (2. * f + 1.) * (2. * f + 1.) * a2_iikk / gr)
    return q


def thermal_pressure(VoverV0, T, Tref, V_0, theta_0, grueneisen_0, q_0, n):
    gr = _grueneisen_parameter_fast(VoverV0, grueneisen_0, q_0)
    theta = _theta_fast(VoverV0, grueneisen_0, q_0, theta_0)
    V = VoverV0 * V_0
    Eth = einstein.thermal_energy(T, theta, n)
    Eth0 = einstein.thermal_energy(Tref, theta, n)
    Pth = gr / V * (Eth - Eth0)
    return Pth


def delta_VoverV0(VoverV0, P, T, Tref, a_V, b_V, c_V, d_V,
                  V_0, theta_0, grueneisen_0, q_0, n):
    Pth = thermal_pressure(VoverV0, T, Tref,
                           V_0, theta_0, grueneisen_0, q_0, n)
    return xioverxi0_Tref(P - Pth, a_V, b_V, c_V, d_V) - VoverV0


def a_ijkloverS_ijkl(VoverV0, T, V_0, theta_0, grueneisen_0, q_0, n):
    gr = _grueneisen_parameter_fast(VoverV0, grueneisen_0, q_0)
    theta = _theta_fast(VoverV0, grueneisen_0, q_0, theta_0)
    V = VoverV0 * V_0
    Cv = einstein.molar_heat_capacity_v(T, theta, n)
    return gr*Cv/V


def dPthdP_Tref(VoverV0, T, Tref, theta_0, grueneisen_0, q_0, n,
                betaoverV_Tref):

    theta = _theta_fast(VoverV0, grueneisen_0, q_0, theta_0)
    gr = _grueneisen_parameter_fast(VoverV0, grueneisen_0, q_0)
    q = _volume_dependent_q_fast(VoverV0, grueneisen_0, q_0)

    Eth = einstein.thermal_energy(T, theta, n)
    Eth0 = einstein.thermal_energy(Tref, theta, n)

    Cv = einstein.molar_heat_capacity_v(T, theta, n)
    Cv0 = einstein.molar_heat_capacity_v(Tref, theta, n)

    return betaoverV_Tref*gr*((gr + 1. - q)*(Eth - Eth0)
                              - gr*(Cv*T - Cv0*Tref))


def thermal_properties_ijkl(P, T, a, b, c, d, gr_0, q_0, params):
    a_V, b_V, c_V, d_V, gr_V, q_V = params['beta_T_reuss']['coeffs']
    V_0 = params['V_0']
    T_0 = params['T_0']
    theta_0 = params['theta_0']
    n = params['n']

    # use gr and q from chi, everything else from volume
    args = (P, T, T_0, a_V, b_V, c_V, d_V, V_0, theta_0, gr_0, q_0, n)
    try:
        sol = bracket(delta_VoverV0, 1., 1.e-2, args)
    except ValueError:
        raise Exception('Cannot find a volume, perhaps you are outside '
                        'of the range of validity for the equation of state?')
    VoverV0_equiv = brentq(delta_VoverV0, sol[0], sol[1], args=args)
    Pth_equiv = thermal_pressure(VoverV0_equiv, T, T_0, V_0, theta_0,
                                 gr_0, q_0, n)
    betaoverV_Tref = (S_ijkl_Tref(P - Pth_equiv, a_V, b_V, c_V, d_V)
                      / (VoverV0_equiv * V_0))

    dPthdP0 = dPthdP_Tref(VoverV0_equiv, T, T_0, theta_0,
                          gr_0, q_0, n, betaoverV_Tref)

    xioverxi0_ijkl = xioverxi0_Tref(P - Pth_equiv, a, b, c, d)
    S_ijkl = S_ijkl_Tref(P - Pth_equiv, a, b, c, d) / (1. + dPthdP0)
    a_ijkl = a_ijkloverS_ijkl(VoverV0_equiv, T, V_0, theta_0,
                              gr_0, q_0, n) * S_ijkl

    return {'xi_rel': xioverxi0_ijkl, 'S': S_ijkl, 'a': a_ijkl,
            'P_th': Pth_equiv}


class AEOS(anisotropiceos.AnisotropicEquationOfState):
    """
    Class for the Myhill (2019) anisotropic equation of state
    """

    def _VoverV0(self, pressure, temperature, params):
        a, b, c, d, gr_0, q_0 = params['beta_T_reuss']['coeffs']  # f == 1

        args = (pressure, temperature, params['T_0'], a, b, c, d,
                params['V_0'], params['theta_0'],
                gr_0, q_0, params['n'])
        try:
            sol = bracket(delta_VoverV0, 1., 1.e-2, args)
        except ValueError:
            raise Exception(
                'Cannot find a volume, perhaps you are outside of the '
                'range of validity for the equation of state?')
        VoverV0 = brentq(delta_VoverV0, sol[0], sol[1], args=args)
        return VoverV0

    def unit_cell_volume(self, pressure, temperature, params):
        return (self._VoverV0(pressure, temperature, params)
                * params['unit_cell_volume_0'])

    def molar_volume(self, pressure, temperature, params):
        return (self._VoverV0(pressure, temperature, params)
                * params['V_0'])

    def unit_cell_vectors(self, pressure, temperature, params):
        if ((params['symmetry_type'] != 'monoclinic'
             and params['symmetry_type'] != 'triclinic')):  # orthotropic
            a, b, c, d, gr, q = params['beta_T_reuss']['coeffs']
            VoverV0 = thermal_properties_ijkl(pressure, temperature,
                                              a, b, c, d, gr, q, params)['xi_rel']
            a, b, c, d, gr, q = params['E_2']['coeffs']
            boverb0 = thermal_properties_ijkl(pressure, temperature,
                                              a, b, c, d, gr, q, params)['xi_rel']
            a, b, c, d, gr, q = params['E_3']['coeffs']
            coverc0 = thermal_properties_ijkl(pressure, temperature,
                                              a, b, c, d, gr, q, params)['xi_rel']
            aovera0 = VoverV0/boverb0/coverc0
            M = np.array([[aovera0, 0., 0.],
                          [0., boverb0, 0.],
                          [0., 0., coverc0]])
            return M.dot(params['unit_cell_vectors'])
        else:  # non-orthotropic
            raise Exception('Unit cell vector calculations'
                            'not yet implemented for '
                            'non orthotropic equation of state')

    def isothermal_compressibility_reuss(self, pressure, temperature, params):
        a, b, c, d, gr_0, q_0 = params['beta_T_reuss']['coeffs']
        VoverV0 = self._VoverV0(pressure, temperature, params)
        Pth = thermal_pressure(VoverV0, temperature,
                               params['T_0'], params['V_0'],
                               params['theta_0'], gr_0, q_0, params['n'])
        S_Tref = S_ijkl_Tref(pressure - Pth, a, b, c, d)
        betaoverV_Tref = S_Tref / (VoverV0 * params['V_0'])
        fac = 1. + dPthdP_Tref(self._VoverV0(pressure, temperature, params),
                               temperature, params['T_0'], params['theta_0'],
                               gr_0, q_0,
                               params['n'], betaoverV_Tref)
        return S_Tref/fac

    def isothermal_bulk_modulus_reuss(self, pressure, temperature, params):
        return 1./self.isothermal_compressibility_reuss(pressure, temperature, params)

    def molar_gibbs_free_energy(self, pressure, temperature, params):
        # First integrate cold path
        a, b, c, d, gr_0, q_0 = params['beta_T_reuss']['coeffs']
        VoverV0 = self._VoverV0(pressure, temperature, params)
        Pth = thermal_pressure(VoverV0, temperature,
                               params['T_0'], params['V_0'],
                               params['theta_0'], gr_0, q_0, params['n'])
        P_Tref = pressure - Pth
        Gc = intVoverV0dP_Tref(P_Tref, 0., a, b, c, d)

        # Quasiharmonic part
        theta = _theta_fast(VoverV0, gr_0, q_0, params['n'])
        Eth = einstein.thermal_energy(temperature, theta, params['n'])
        Eth0 = einstein.thermal_energy(params['T_0'], theta, params['n'])
        Gqh = Eth - Eth0 + VoverV0*params['V_0']*Pth
        return params['gibbs_0'] + Gc + Gqh

    def molar_entropy(self, pressure, temperature, params):
        a, b, c, d, gr_0, q_0 = params['beta_T_reuss']['coeffs']
        VoverV0 = self._VoverV0(pressure, temperature, params)
        einstein_T = _theta_fast(VoverV0,
                                 gr_0, q_0, params['theta_0'])
        return einstein.molar_entropy(temperature, einstein_T, params['n'])

    def molar_heat_capacity_v(self, pressure, temperature, params):
        a, b, c, d, gr_0, q_0 = params['beta_T_reuss']['coeffs']
        VoverV0 = self._VoverV0(pressure, temperature, params)
        einstein_T = _theta_fast(VoverV0, gr_0, q_0,
                                 params['theta_0'])
        return einstein.molar_heat_capacity_v(temperature, einstein_T,
                                              params['n'])

    def grueneisen_parameter(self, pressure, temperature, params):
        a, b, c, d, gr_0, q_0 = params['beta_T_reuss']['coeffs']
        VoverV0 = self._VoverV0(pressure, temperature, params)
        return _grueneisen_parameter_fast(VoverV0, gr_0, q_0)

    def volumetric_thermal_expansivity(self, pressure, temperature, params):
        """
        Returns thermal expansivity. :math:`[1/K]`
        """
        volume = self.molar_volume(pressure, temperature, params)
        C_v = self.molar_heat_capacity_v(pressure, temperature, params)
        gr = self.grueneisen_parameter(pressure, temperature, params)
        beta = self.isothermal_compressibility_reuss(pressure, temperature,
                                                     params)
        alpha = gr * C_v * beta / volume
        return alpha

    def molar_heat_capacity_p(self, pressure, temperature, params):

        volume = self.molar_volume(pressure, temperature, params)
        C_v = self.molar_heat_capacity_v(pressure, temperature, params)
        gr = self.grueneisen_parameter(pressure, temperature, params)
        beta = self.isothermal_compressibility_reuss(pressure, temperature,
                                                     params)

        return C_v * (1. + gr * gr * C_v * beta * temperature / volume)

    def thermal_expansivity_tensor(self, pressure, temperature, params):
        alpha = np.zeros((6))

        a, b, c, d, gr_0, q_0 = params['E_3']['coeffs']
        alpha[2] = thermal_properties_ijkl(pressure, temperature,
                                           a, b, c, d, gr_0, q_0, params)['a']
        a, b, c, d, gr_0, q_0 = params['E_2']['coeffs']
        alpha[1] = thermal_properties_ijkl(pressure, temperature,
                                           a, b, c, d, gr_0, q_0, params)['a']
        a, b, c, d, gr_0, q_0 = params['beta_T_reuss']['coeffs']
        a_V = thermal_properties_ijkl(pressure, temperature,
                                      a, b, c, d, gr_0, q_0, params)['a']
        alpha[0] = a_V - alpha[1] - alpha[2]

        # we only need to sum the elements S_ijkk, assuming hydrostatic stress
        for i in [3, 4, 5]:
            for j in range(3):
                try:
                    a, b, c, d, gr_0, q_0 = params['S_T']['{0}{1}'.format(j+1, i+1)]['coeffs']
                    alpha[i] = thermal_properties_ijkl(pressure, temperature,
                                                       a, b, c, d, gr_0, q_0,
                                                       params)['a']
                except KeyError:
                    pass

        return alpha

    def isothermal_elastic_compliance_tensor(self, pressure, temperature,
                                             params):
        ST = np.zeros((6, 6))
        for i in range(6):
            for j in range(i, 6):
                try:
                    a, b, c, d, gr_0, q_0 = params['S_T']['{0}{1}'.format(j+1, i+1)]['coeffs']
                    ST[i, j] = thermal_properties_ijkl(pressure, temperature,
                                                       a, b, c, d, gr_0, q_0,
                                                       params)['S']
                    ST[j, i] = ST[i, j]
                except KeyError:
                    pass

        # fill in non-diagonals of upper 3x3 block
        a, b, c, d, gr_0, q_0 = params['E_3']['coeffs']
        E_3 = thermal_properties_ijkl(pressure, temperature,
                                      a, b, c, d, gr_0, q_0, params)['S']
        a, b, c, d, gr_0, q_0 = params['E_2']['coeffs']
        E_2 = thermal_properties_ijkl(pressure, temperature,
                                      a, b, c, d, gr_0, q_0, params)['S']
        a, b, c, d, gr_0, q_0 = params['beta_T_reuss']['coeffs']
        E_1 = (thermal_properties_ijkl(pressure, temperature,
                                       a, b, c, d, gr_0, q_0, params)['S']
               - E_2 - E_3)

        ST[0, 1] = ((E_1 + E_2 - E_3) - (ST[0, 0] + ST[1, 1] - ST[2, 2])) / 2.
        ST[0, 2] = ((E_1 + E_3 - E_2) - (ST[0, 0] + ST[2, 2] - ST[1, 1])) / 2.
        ST[1, 2] = ((E_2 + E_3 - E_1) - (ST[1, 1] + ST[2, 2] - ST[0, 0])) / 2.
        ST[1, 0] = ST[0, 1]
        ST[2, 0] = ST[0, 2]
        ST[2, 1] = ST[1, 2]
        return ST

    def isothermal_elastic_stiffness_tensor(self, pressure, temperature,
                                            params):
        ST = self.isothermal_elastic_compliance_tensor(pressure, temperature,
                                                       params)
        return np.linalg.pinv(ST)

    def isentropic_elastic_compliance_tensor(self, pressure, temperature,
                                             params):
        ST = self.isothermal_elastic_compliance_tensor(pressure, temperature,
                                                       params)
        alpha = self.thermal_expansivity_tensor(pressure, temperature, params)
        molar_volume = self.molar_volume(pressure, temperature, params)
        molar_Cp = self.molar_heat_capacity_p(pressure, temperature, params)
        SN = ST - (np.einsum('i, j -> ij', alpha, alpha)
                   * molar_volume * temperature
                   / molar_Cp)
        return SN

    def isentropic_elastic_stiffness_tensor(self, pressure, temperature,
                                            params):
        SN = self.isentropic_elastic_compliance_tensor(pressure, temperature,
                                                       params)
        return np.linalg.pinv(SN)

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
                if key != 'grueneisen_0' and key != 'q_0':
                    E[key] *= 3.

            params['E_2'] = E
            params['E_3'] = E

        expected_keys = ['P_0', 'G_0', 'theta_0',
                         'symmetry_type', 'unit_cell_vectors', 'Z',
                         'beta_T_reuss', 'E_2', 'E_3',
                         'S_T', 'T_0', 'n', 'molar_mass']
        for k in expected_keys:
            if k not in params:
                raise KeyError('params object missing key: ' + k)

        params['unit_cell_volume_0'] = np.linalg.det(np.array(params['unit_cell_vectors']))
        params['V_0'] = (params['unit_cell_volume_0'] * Avogadro
                         / params['Z'])

        # nothing from the upper 3x3 non-diagonal block
        required_compliance_keys = ['11', '44']
        for k in required_compliance_keys:
            if k not in params['S_T']:
                raise KeyError('params object S_T missing key: ' + k)

        # Convert Sij derivatives into the static parameters
        for key in ['beta_T_reuss', 'E_2', 'E_3']:
            params[key]['coeffs'] = Sijkl_derivs_to_coeffs(params[key])

        for key in params['S_T']:
            params['S_T'][key]['coeffs'] = Sijkl_derivs_to_coeffs(params['S_T'][key])

        # Fill in missing terms
        if params['symmetry_type'] == 'cubic':
            cubic_correspondences = [['11', ['22', '33']],
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
