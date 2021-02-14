# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.

from __future__ import absolute_import

import numpy as np
import scipy.optimize as opt

from . import debye
from . import equation_of_state as eos
from .. import constants


class PitzerSterner(eos.EquationOfState):

    """
    Base class for the Pitzer and Sterner gas phase equation of state detailed
    in :cite:`PitzerSterner1994`.
    """

    def _compute_cs(self, c_coeffs, temperature):
        cs = np.empty(10)
        for i, ci in enumerate(c_coeffs):
            cs[i] = (ci[0]*np.power(temperature, -4.)
                     + ci[1]*np.power(temperature, -2.)
                     + ci[2]*np.power(temperature, -1.)
                     + ci[3]
                     + ci[4]*temperature
                     + ci[5]*np.power(temperature, 2))
        return cs

    def _delta_pressure(self, volume, pressure, temperature, params):
        return self.pressure(temperature, volume, params) - pressure

    def volume(self, pressure, temperature, params):
        """
        Returns molar volume. :math:`[m^3]`
        """
        return opt.brentq(self._delta_pressure, 1.e-7, 1.,
                          args=(pressure, temperature, params))

    def pressure(self, temperature, volume, params):
        """
        Returns the pressure of the mineral at a given
        temperature and volume [Pa]
        """
        # convert volume in m^3/mol to specific density in mol/cm^3
        rho_sp = 1.e-6 / volume
        c = self._compute_cs(params['c_coeffs'], temperature)
        reduced_P = (rho_sp
                     + c[0]*np.power(rho_sp, 2.)
                     - np.power(rho_sp, 2.)
                     * ((c[2]
                         + 2.*c[3]*rho_sp
                         + 3.*c[4]*np.power(rho_sp, 2.)
                         + 4.*c[5]*np.power(rho_sp, 3.))
                        / np.power(c[1]
                                   + c[2]*rho_sp
                                   + c[3]*np.power(rho_sp, 2.)
                                   + c[4]*np.power(rho_sp, 3.)
                                   + c[5]*np.power(rho_sp, 4.), 2.))
                     + c[6]*np.power(rho_sp, 2.)*np.exp(-c[7]*rho_sp)
                     + c[8]*np.power(rho_sp, 2.)*np.exp(-c[9]*rho_sp))

        # Expression returns reduced pressure in MPa, so multiply by 1.e6
        return reduced_P * constants.gas_constant * temperature * 1.e6

    def grueneisen_parameter(self, pressure, temperature, volume, params):
        """
        Returns grueneisen parameter :math:`[unitless]`
        """
        return NotImplementedError()

    def isothermal_bulk_modulus(self, pressure, temperature, volume, params):
        """
        Returns isothermal bulk modulus :math:`[Pa]`
        """
        return NotImplementedError()

    def adiabatic_bulk_modulus(self, pressure, temperature, volume, params):
        """
        Returns adiabatic bulk modulus. :math:`[Pa]`
        """
        return NotImplementedError()

    def shear_modulus(self, pressure, temperature, volume, params):
        """
        Returns shear modulus. :math:`[Pa]`
        Gas phases have a shear modulus of 0.
        """
        return 0.

    def molar_heat_capacity_v(self, pressure, temperature, volume, params):
        """
        Returns heat capacity at constant volume. :math:`[J/K/mol]`
        """
        return NotImplementedError()

    def molar_heat_capacity_p(self, pressure, temperature, volume, params):
        """
        Returns heat capacity at constant pressure. :math:`[J/K/mol]`
        """
        return NotImplementedError()

    def thermal_expansivity(self, pressure, temperature, volume, params):
        """
        Returns thermal expansivity. :math:`[1/K]`
        """
        return NotImplementedError()

    def gibbs_free_energy(self, pressure, temperature, volume, params):
        """
        Returns the Gibbs free energy at the pressure and temperature
        of the mineral [J/mol]
        """
        G = (self.helmholtz_free_energy(pressure, temperature, volume, params)
             + pressure * volume)
        return G

    def molar_internal_energy(self, pressure, temperature, volume, params):
        """
        Returns the internal energy at the pressure and temperature
        of the mineral [J/mol]
        """
        return (self.helmholtz_free_energy(pressure, temperature, volume,
                                           params)
                + (temperature
                   * self.entropy(pressure, temperature, volume, params)))

    def entropy(self, pressure, temperature, volume, params):
        """
        Returns the entropy at the pressure and temperature
        of the mineral [J/K/mol]
        """
        dT = 0.1
        F0 = self.helmholtz_free_energy(pressure, temperature - dT/2.,
                                        volume, params)
        F1 = self.helmholtz_free_energy(pressure, temperature + dT/2.,
                                        volume, params)
        return -(F1 - F0)/dT

    def enthalpy(self, pressure, temperature, volume, params):
        """
        Returns the enthalpy at the pressure and temperature
        of the mineral [J/mol]
        """

        return (self.helmholtz_free_energy(pressure, temperature, volume,
                                           params)
                + (temperature * self.entropy(pressure, temperature, volume,
                                              params)
                   + pressure * volume))

    def helmholtz_free_energy(self, pressure, temperature, volume, params):
        """
        Returns the Helmholtz free energy at the pressure and temperature
        of the mineral [J/mol]
        """
        # convert volume in m^3/mol to specific density in mol/cm^3
        rho_sp = 1.e-6 / volume
        c = self._compute_cs(params['c_coeffs'], temperature)
        reduced_F = (np.log(rho_sp)
                     + c[0]*rho_sp
                     + (1./(c[1]
                            + c[2]*rho_sp
                            + c[3]*np.power(rho_sp, 2.)
                            + c[4]*np.power(rho_sp, 3.)
                            + c[5]*np.power(rho_sp, 4.))
                        - 1./c[1])
                     - (c[6]/c[7])*(np.exp(-c[7]*rho_sp) - 1.)
                     - (c[8]/c[9])*(np.exp(-c[9]*rho_sp) - 1))

        F_debye = (params['Cv_0']*(temperature
                                   - temperature*np.log(temperature))
                   + debye.helmholtz_free_energy(temperature,
                                                 params['Debye_0'],
                                                 n=params['Debye_n']))
        return (reduced_F*constants.gas_constant*temperature
                + F_debye + params['F_0'])

    def validate_parameters(self, params):
        """
        Check for existence and validity of the parameters
        """

        # Now check all the required keys for the
        # thermal part of the EoS are in the dictionary
        expected_keys = ['molar_mass', 'n', 'formula',
                         'Debye_n', 'Debye_0', 'Cv_0', 'c_coeffs']
        for k in expected_keys:
            if k not in params:
                raise KeyError('params object missing parameter : ' + k)
