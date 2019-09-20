# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2019 by the BurnMan team, released under the GNU
# GPL v2 or later.


class AnisotropicEquationOfState(object):

    """
    This class defines the interface for an anisotropic equation of state
    that a mineral uses to determine its properties at a
    given :math:`P, T`.  In order define a new equation of state, you
    should define these functions.

    All functions should accept and return values in SI units.

    In general these functions are functions of pressure and
    temperature, as well as a "params" object,
    which is a Python dictionary that stores the material
    parameters of the mineral, such as reference unit cell parameters,
    reference moduli, etc.
    """

    def molar_volume(self, pressure, temperature, params):
        """
        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the equation of state. :math:`[Pa]`
        temperature : float
            Temperature at which to evaluate the equation of state. :math:`[K]`
        params : dictionary
            Dictionary containing material parameters required by the equation of state.

        Returns
        -------
            Molar volume of the mineral. :math:`[m^3]`
        """
        raise NotImplementedError("")

    def pressure(self, temperature, volume, params):
        """
        Parameters
        ----------
        volume : float
            Molar volume at which to evaluate the equation of state. [m^3]
        temperature : float
            Temperature at which to evaluate the equation of state. [K]
        params : dictionary
            Dictionary containing material parameters required by the equation of state.

        Returns
        -------
        pressure : float
            Pressure of the mineral, including cold and thermal parts. [m^3]
        """
        raise NotImplementedError("")

    def density(self, volume, params):
        """
        Calculate the density of the mineral :math:`[kg/m^3]`.
        The params object must include a "molar_mass" field.

        Parameters
        ----------
        volume : float
        Molar volume of the mineral.  For consistency this should be calculated
        using :func:`volume`. :math:`[m^3]`
        params : dictionary
            Dictionary containing material parameters required by the equation of state.

        Returns
        -------
        density : float
            Density of the mineral. :math:`[kg/m^3]`
        """
        return params["molar_mass"] / volume

    def unit_cell_vectors(self, pressure, temperature, params):
        """
        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the equation of state. :math:`[Pa]`
        temperature : float
            Temperature at which to evaluate the equation of state. :math:`[K]`
        params : dictionary
            Dictionary containing material parameters required by the equation of state.

        Returns
        -------
        cell_vectors : list of 3 numpy arrays
            Vectors defining the primitive unit cell. :math:`[m]`
        """
        raise NotImplementedError("")

    def unit_cell_volume(self, pressure, temperature, params):
        """
        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the equation of state. :math:`[Pa]`
        temperature : float
            Temperature at which to evaluate the equation of state. :math:`[K]`
        params : dictionary
            Dictionary containing material parameters required by the equation of state.

        Returns
        -------
        primitive_cell_volume : float
            The primitive unit cell volume. :math:`[m^3]`
        """
        raise NotImplementedError("")

    def grueneisen_parameter_tensor(self, pressure, temperature, params):
        """
        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the equation of state. :math:`[Pa]`
        temperature : float
            Temperature at which to evaluate the equation of state. :math:`[K]`
        params : dictionary
            Dictionary containing material parameters required by the equation of state.

        Returns
        -------
        gamma : 2D numpy array of floats
            2D Grueneisen parameter tensor of the mineral. :math:`[unitless]`
        """
        raise NotImplementedError("")

    def isothermal_elastic_stiffness_tensor(self, pressure, temperature, params):
        """
        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the equation of state. :math:`[Pa]`
        temperature : float
            Temperature at which to evaluate the equation of state. :math:`[K]`
        params : dictionary
            Dictionary containing material parameters required by the equation of state.

        Returns
        -------
        :math:`C^T_{ijkl}` : 2D numpy array of floats
            Isothermal elastic stiffness tensor of the mineral (Voigt notation). :math:`[Pa]`
        """
        raise NotImplementedError("")

    def adiabatic_elastic_stiffness_tensor(self, pressure, temperature, params):
        """
        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the equation of state. :math:`[Pa]`
        temperature : float
            Temperature at which to evaluate the equation of state. :math:`[K]`
        params : dictionary
            Dictionary containing material parameters required by the equation of state.

        Returns
        -------
        :math:`C^S_{ijkl}` : 2D numpy array of floats
            Isothermal elastic stiffness tensor of the mineral (Voigt notation). :math:`[Pa]`
        """
        raise NotImplementedError("")

    def isothermal_elastic_compliance_tensor(self, pressure, temperature, params):
        """
        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the equation of state. :math:`[Pa]`
        temperature : float
            Temperature at which to evaluate the equation of state. :math:`[K]`
        params : dictionary
            Dictionary containing material parameters required by the equation of state.

        Returns
        -------
        :math:`S^T_{ijkl}` : 2D numpy array of floats
            Isothermal elastic compliance tensor of the mineral (Voigt notation). :math:`[Pa]`
        """
        raise NotImplementedError("")

    def adiabatic_elastic_compliance_tensor(self, pressure, temperature, params):
        """
        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the equation of state. :math:`[Pa]`
        temperature : float
            Temperature at which to evaluate the equation of state. :math:`[K]`
        params : dictionary
            Dictionary containing material parameters required by the equation of state.

        Returns
        -------
        :math:`S^S_{ijkl}` : 2D numpy array of floats
            Isothermal elastic compliance tensor of the mineral (Voigt notation). :math:`[Pa]`
        """
        raise NotImplementedError("")

    def molar_heat_capacity_v(self, pressure, temperature, params):
        """
        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the equation of state. :math:`[Pa]`
        temperature : float
            Temperature at which to evaluate the equation of state. :math:`[K]`
        params : dictionary
            Dictionary containing material parameters required by the equation of state.

        Returns
        -------
        C_V : float
            Heat capacity at constant volume of the mineral. :math:`[J/K/mol]`
        """
        raise NotImplementedError("")

    def molar_heat_capacity_p(self, pressure, temperature, params):
        """
        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the equation of state. :math:`[Pa]`
        temperature : float
            Temperature at which to evaluate the equation of state. :math:`[K]`
        params : dictionary
            Dictionary containing material parameters required by the equation of state.

        Returns
        -------
        C_P : float
            Heat capacity at constant pressure of the mineral. :math:`[J/K/mol]`
        """
        raise NotImplementedError("")

    def thermal_expansivity_tensor(self, pressure, temperature, params):
        """
        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the equation of state. :math:`[Pa]`
        temperature : float
            Temperature at which to evaluate the equation of state. :math:`[K]`
        params : dictionary
            Dictionary containing material parameters required by the equation of state.

        Returns
        -------
        alpha : 2D numpy array of floats
            Thermal expansivity tensor of the mineral. :math:`[1/K]`
        """
        raise NotImplementedError("")

    def molar_gibbs_free_energy(self, pressure, temperature, params):
        """
        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the equation of state. [Pa]
        temperature : float
            Temperature at which to evaluate the equation of state. [K]
        params : dictionary
            Dictionary containing material parameters required by the equation of state.

        Returns
        -------
        G : float
            Gibbs free energy of the mineral
        """
        raise NotImplementedError("")

    def molar_helmholtz_free_energy(self, pressure, temperature, params):
        """
        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the equation of state. [Pa]
        temperature : float
            Temperature at which to evaluate the equation of state. [K]
        params : dictionary
            Dictionary containing material parameters required by the equation of state.

        Returns
        -------
        F : float
            Helmholtz free energy of the mineral
        """
        raise NotImplementedError("")

    def molar_entropy(self, pressure, temperature, params):
        """
        Returns the entropy at the pressure and temperature of the mineral [J/K/mol]
        """

        raise NotImplementedError("")

    def molar_enthalpy(self, pressure, temperature, params):
        """
        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the equation of state. [Pa]
        temperature : float
            Temperature at which to evaluate the equation of state. [K]
        params : dictionary
            Dictionary containing material parameters required by the equation of state.

        Returns
        -------
        H : float
            Enthalpy of the mineral
        """
        raise NotImplementedError("")

    def molar_internal_energy(self, pressure, temperature, params):
        """
        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the equation of state. [Pa]
        temperature : float
            Temperature at which to evaluate the equation of state. [K]
        params : dictionary
            Dictionary containing material parameters required by the equation of state.

        Returns
        -------
        U : float
            Internal energy of the mineral
        """
        raise NotImplementedError("")

    def validate_parameters(self, params):
        """
        The params object is just a dictionary associating mineral physics parameters
        for the equation of state.  Different equation of states can have different parameters,
        and the parameters may have ranges of validity.  The intent of this function is
        twofold. First, it can check for the existence of the parameters that the
        equation of state needs, and second, it can check whether the parameters have reasonable
        values.  Unreasonable values will frequently be due to unit issues (e.g., supplying
        bulk moduli in GPa instead of Pa). In the base class this function does nothing,
        and an equation of state is not required to implement it.  This function will
        not return anything, though it may raise warnings or errors.

        Parameters
        ----------
        params : dictionary
            Dictionary containing material parameters required by the equation of state.
        """
        pass
