# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2017 by the BurnMan team, released under the GNU
# GPL v2 or later.

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import warnings

from .material import Material, material_property
from .mineral import Mineral
from . import averaging_schemes
from . import chemicalpotentials


def check_pairs(phases, amounts):
        if len(amounts) < 1:
            raise Exception('ERROR: we need at least one phase')

        if len(phases) != len(amounts):
            raise Exception(
                'ERROR: different array lengths for phases and amounts')

        total = sum(amounts)
        if abs(total - 1.0) > 1e-10:
            raise Exception(
                'ERROR: list of molar amounts does not add up to one')
        for p in phases:
            if not isinstance(p, Mineral):
                raise Exception(
                    'ERROR: object of type ''%s'' is not of type Mineral' % (type(p)))


# static composite of minerals/composites
class Composite(Material):

    """
    Base class for a composite material.
    The static phases can be minerals or materials,
    meaning composite can be nested arbitrarily.

    The amounts of the phases can be input
    as either 'molar' or 'mass' during instantiation,
    and modified (or initialised) after this point by
    using set_amounts.

    This class is available as ``burnman.Composite``.
    """

    def __init__(self, phases, amounts=None, amount_type='molar'):
        """
        Create a composite using a list of phases and their amounts.

        Parameters
        ----------
        phases: list of :class:`burnman.Material`
            list of phases.
        amounts: list of floats
            number of moles or kg for each phase.
        amount_type: 'molar' or 'mass' (optional, 'molar' as standard)
            specify whether molar or mass amounts are specified.
        """

        Material.__init__(self)

        assert(len(phases) > 0)
        self.phases = phases

        if amounts is not None:
            self.set_amounts(amounts, amount_type)
        else:
            self.molar_amounts = None

        self.set_averaging_scheme('VoigtReussHill')

    def set_amounts(self, amounts, amount_type='molar'):
        """
        Change the amounts of the phases of this Composite.

        Parameters
        ----------
        amounts: list of floats
            number of moles or kg for each phase.
        amount_type: 'molar' or 'mass'
            specify whether molar or mass amounts are specified.
        """
        assert(len(self.phases) == len(amounts))

        try:
            total = sum(amounts)
        except TypeError:
            raise Exception(
                "Since v0.8, burnman.Composite takes an array of Materials, then an array of amounts")

        for f in amounts:
            assert (f >= -1e-12)

        if amount_type == 'molar':
            molar_amounts = amounts
        elif amount_type == 'mass':
            molar_amounts = self._mass_to_molar_amounts(
                self.phases, amounts)
        else:
            raise Exception(
                "Amount type not recognised. Please use 'molar' or mass")

        # Set minimum value of a molar amount at 0.0 (rather than -1.e-12)
        self.molar_amounts = np.array([max(0.0, amount)
                                       for amount in molar_amounts])

    def set_method(self, method):
        """
        set the same equation of state method for all the phases in the composite
        """
        for phase in self.phases:
            phase.set_method(method)
        # Clear the cache on resetting method
        self.reset()

    def set_averaging_scheme(self, averaging_scheme):
        """
        Set the averaging scheme for the moduli in the composite.
        Default is set to VoigtReussHill, when Composite is initialized.
        """

        if type(averaging_scheme) == str:
            self.averaging_scheme = getattr(
                averaging_schemes, averaging_scheme)()
        else:
            self.averaging_scheme = averaging_scheme
        # Clear the cache on resetting averaging scheme
        self.reset()

    def set_state(self, pressure, temperature):
        """
        Update the material to the given pressure [Pa] and temperature [K].
        """
        Material.set_state(self, pressure, temperature)
        for phase in self.phases:
            phase.set_state(pressure, temperature)

    def debug_print(self, indent=""):
        print("%sComposite:" % indent)
        indent += "  "
        if self.molar_amounts is None:
            for i, phase in enumerate(self.phases):
                phase.debug_print(indent + "  ")
        else:
            for i, phase in enumerate(self.phases):
                print("%s%g of" % (indent, self.molar_amounts[i]))
                phase.debug_print(indent + "  ")

    def unroll(self):
        if self.molar_amounts is None:
            raise Exception(
                "Unroll only works if the composite has defined amounts.")
        phases = []
        amounts = []
        for i, phase in enumerate(self.phases):
            p_mineral, p_fraction = phase.unroll()
            check_pairs(p_mineral, p_fraction)
            amounts.extend([f * self.molar_amounts[i] for f in p_fraction])
            phases.extend(p_mineral)
        return phases, amounts

    def to_string(self):
        """
        return the name of the composite
        """
        return "'" + self.__class__.__name__ + "'"

    @material_property
    def internal_energy(self):
        """
        Returns internal energy of the mineral [J]
        Aliased with self.energy
        """
        U = sum(phase.internal_energy * molar_amount
                for (phase, molar_amount)
                in zip(self.phases, self.molar_amounts))
        return U

    @material_property
    def gibbs(self):
        """
        Returns Gibbs free energy of the composite [J]
        """
        G = sum(phase.molar_gibbs * molar_amount
                for (phase, molar_amount)
                in zip(self.phases, self.molar_amounts))
        return G

    @material_property
    def helmholtz(self):
        """
        Returns Helmholtz free energy of the composite [J]
        """
        F = sum(phase.molar_helmholtz * molar_amount
                for (phase, molar_amount)
                in zip(self.phases, self.molar_amounts))
        return F

    @material_property
    def volume(self):
        """
        Returns volume of the composite [m^3]
        Aliased with self.V
        """
        volumes = np.array([phase.molar_volume * molar_amount
                            for (phase, molar_amount)
                            in zip(self.phases, self.molar_amounts)])
        return np.sum(volumes)

    @material_property
    def mass(self):
        """
        Returns mass of the composite [kg]
        """
        return sum([phase.molar_mass * molar_amount
                    for (phase, molar_amount)
                    in zip(self.phases, self.molar_amounts)])

    @material_property
    def density(self):
        """
        Compute the density of the composite based on the molar volumes and masses
        Aliased with self.rho
        """
        densities = np.array([phase.density for phase in self.phases])
        volumes = np.array([phase.molar_volume * molar_amount
                            for (phase, molar_amount)
                            in zip(self.phases, self.molar_amounts)])
        return self.averaging_scheme.average_density(volumes, densities)

    @material_property
    def entropy(self):
        """
        Returns enthalpy of the mineral [J]
        Aliased with self.S
        """
        S = sum(phase.molar_entropy * molar_amount
                for (phase, molar_amount)
                in zip(self.phases, self.molar_amounts))
        return S

    @material_property
    def enthalpy(self):
        """
        Returns enthalpy of the mineral [J]
        Aliased with self.H
        """
        H = sum(phase.molar_enthalpy * molar_amount
                for (phase, molar_amount)
                in zip(self.phases, self.molar_amounts))
        return H

    @material_property
    def isothermal_bulk_modulus(self):
        """
        Returns isothermal bulk modulus of the composite [Pa]
        Aliased with self.K_T
        """
        V_amounts = np.array([phase.molar_volume * molar_amount
                              for (phase, molar_amount)
                              in zip(self.phases, self.molar_amounts)])
        K_ph = np.array([phase.isothermal_bulk_modulus
                         for phase in self.phases])
        G_ph = np.array([phase.shear_modulus
                         for phase in self.phases])

        return self.averaging_scheme.average_bulk_moduli(V_amounts, K_ph, G_ph)

    @material_property
    def adiabatic_bulk_modulus(self):
        """
        Returns adiabatic bulk modulus of the mineral [Pa]
        Aliased with self.K_S
        """
        V_amounts = np.array([phase.molar_volume * molar_amount
                              for (phase, molar_amount)
                              in zip(self.phases, self.molar_amounts)])
        K_ph = np.array([phase.adiabatic_bulk_modulus
                         for phase in self.phases])
        G_ph = np.array([phase.shear_modulus
                         for phase in self.phases])

        return self.averaging_scheme.average_bulk_moduli(V_amounts, K_ph, G_ph)

    @material_property
    def isothermal_compressibility(self):
        """
        Returns isothermal compressibility of the composite (or inverse isothermal bulk modulus) [1/Pa]
        Aliased with self.beta_T
        """
        return 1. / self.isothermal_bulk_modulus

    @material_property
    def adiabatic_compressibility(self):
        """
        Returns isothermal compressibility of the composite (or inverse isothermal bulk modulus) [1/Pa]
        Aliased with self.beta_S
        """
        return 1. / self.adiabatic_bulk_modulus

    @material_property
    def shear_modulus(self):
        """
        Returns shear modulus of the mineral [Pa]
        Aliased with self.G
        """
        V_amounts = np.array([phase.molar_volume * molar_amount for (
                           phase, molar_amount)
                              in zip(self.phases, self.molar_amounts)])
        K_ph = np.array([phase.adiabatic_bulk_modulus
                         for phase in self.phases])
        G_ph = np.array([phase.shear_modulus
                         for phase in self.phases])

        return self.averaging_scheme.average_shear_moduli(V_amounts, K_ph, G_ph)

    @material_property
    def p_wave_velocity(self):
        """
        Returns P wave speed of the composite [m/s]
        Aliased with self.v_p
        """
        return np.sqrt((self.adiabatic_bulk_modulus + 4. / 3. *
                        self.shear_modulus) / self.density)

    @material_property
    def bulk_sound_velocity(self):
        """
        Returns bulk sound speed of the composite [m/s]
        Aliased with self.v_phi
        """
        return np.sqrt(self.adiabatic_bulk_modulus / self.density)

    @material_property
    def shear_wave_velocity(self):
        """
        Returns shear wave speed of the composite [m/s]
        Aliased with self.v_s
        """
        return np.sqrt(self.shear_modulus / self.density)

    @material_property
    def grueneisen_parameter(self):
        """
        Returns grueneisen parameter of the composite [unitless]
        Aliased with self.gr
        """
        return self.thermal_expansivity * self.isothermal_bulk_modulus * self.volume / self.heat_capacity_v

    @material_property
    def thermal_expansivity(self):
        """
        Returns thermal expansion coefficient of the composite [1/K]
        Aliased with self.alpha
        """
        volumes = np.array([phase.molar_volume * molar_amount
                            for (phase, molar_amount)
                            in zip(self.phases, self.molar_amounts)])
        alphas = np.array([phase.thermal_expansivity for phase in self.phases])
        return self.averaging_scheme.average_thermal_expansivity(volumes, alphas)

    @material_property
    def heat_capacity_v(self):
        """
        Returns heat capacity at constant volume of the composite [J/K]
        Aliased with self.C_v
        """
        c_v = np.array([phase.heat_capacity_v for phase in self.phases])
        return self.averaging_scheme.average_heat_capacity_v(self.molar_amounts, c_v)

    @material_property
    def heat_capacity_p(self):
        """
        Returns heat capacity at constant pressure of the composite [J/K]
        Aliased with self.C_p
        """
        c_p = np.array([phase.heat_capacity_p for phase in self.phases])
        return self.averaging_scheme.average_heat_capacity_p(self.molar_amounts, c_p)

    def _mass_to_molar_amounts(self, phases, mass_amounts):
        """
        Converts a set of mass amounts for phases into a set of molar amounts.
        Not normalised!!

        Parameters
        ----------
        phases : list of :class:`burnman.Material`
        The list of phases for which amounts should be converted.

        mass_amounts : list of floats
        The list of mass amounts of the input phases.

        Returns
        -------
        molar_amounts : list of floats
        The list of molar amounts corresponding to the input molar amounts
        """
        molar_amounts = np.array([mass_amount / phase.molar_mass
                                  for mass_amount, phase
                                  in zip(mass_amounts, phases)])
        return molar_amounts
