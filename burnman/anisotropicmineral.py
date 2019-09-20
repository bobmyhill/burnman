# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2017 by the BurnMan team, released under the GNU
# GPL v2 or later.


from __future__ import absolute_import
from __future__ import print_function
import warnings

import numpy as np

from .material import Material, material_property
from . import eos
from .tools import copy_documentation
from .tools import unit_normalize


class AnisotropicMineral(Material):

    """
    This is the base class for all anisotropic minerals. States of the mineral
    can only be queried after setting the pressure and temperature
    using set_state(). The method for computing properties of
    the material is set using set_method(). This is done during
    initialisation if the param 'equation_of_state' has been defined.
    The method can be overridden later by the user.

    This class is available as ``burnman.AnisotropicMineral``.

    If deriving from this class, set the properties in self.params
    to the desired values. For more complicated materials you
    can overwrite set_state(), change the params and then call
    set_state() from this class.

    All the material parameters are expected to be in plain SI units.  This
    means that the elastic moduli should be in Pascals and NOT Gigapascals,
    and the Debye temperature should be in K not C.
    Additionally, the primitive unit cell vectors should be in m and
    'n' should be the number of atoms per molecule.
    To convert this to m^3/(mol of molecule) you should multiply by 10^(-30) *
    N_a / Z, where N_a is Avogadro's number and Z is the number of formula units per
    unit cell. You can look up Z in many places, including www.mindat.org

    See :cite:`Mainprice2011` Geological Society of London Special Publication
    and https://materialsproject.org/wiki/index.php/Elasticity_calculations
    for mathematical descriptions of each function.
    """

    def __init__(self, params=None, property_modifiers=None):
        Material.__init__(self)
        if params is not None:
            self.params = params
        elif 'params' not in self.__dict__:
            self.params = {}

        if property_modifiers is not None:
            self.property_modifiers = property_modifiers
        elif 'property_modifiers' not in self.__dict__:
            self.property_modifiers = []

        self.method = None
        if 'equation_of_state' in self.params:
            self.set_method(self.params['equation_of_state'])
        if 'name' in self.params:
            self.name = self.params['name']

    def set_method(self, equation_of_state):
        """
        Set the equation of state to be used for this mineral.
        Takes a string corresponding to any of the predefined
        anisotropic equations of state, currently just 'aeos'.
        Alternatively, you can pass a user defined
        class which derives from the equation_of_state base class.
        After calling set_method(), any existing derived properties
        (e.g., elastic parameters or thermodynamic potentials) will be out
        of date, so set_state() will need to be called again.
        """

        if equation_of_state is None:
            self.method = None
            return

        new_method = eos.create(equation_of_state)
        if self.method is not None and 'equation_of_state' in self.params:
            self.method = eos.create(self.params['equation_of_state'])

        if type(new_method).__name__ == 'instance':
            raise Exception(
                "Please derive your method from object (see python old style classes)")

        if self.method is not None and type(new_method) is not type(self.method):

            # Warn user that they are changing the EoS
            warnings.warn('Warning, you are changing the method to ' + new_method.__class__.__name__ +
                          ' even though the material is designed to be used with the method ' +
                          self.method.__class__.__name__ +
                          '.  This does not overwrite any mineral attributes', stacklevel=2)
            self.reset()

        self.method = new_method

        # Validate and parse the params object on the requested EOS.
        try:
            self.method.validate_and_parse_parameters(self.params)
        except Exception as e:
            print('Mineral ' + self.to_string() +
                  ' failed to validate parameters with message : \" ' + e.message + '\"')
            raise

        # Invalidate the cache upon resetting the method
        self.reset()

    def to_string(self):
        """
        Returns the name of the mineral class
        """
        return "'" + self.__class__.__module__.replace(".minlib_", ".") + "." + self.__class__.__name__ + "'"

    def debug_print(self, indent=""):
        print("%s%s" % (indent, self.to_string()))

    def unroll(self):
        return ([self], [1.0])

    @copy_documentation(Material.set_state)
    def set_state(self, pressure, temperature):
        Material.set_state(self, pressure, temperature)
        self._property_modifiers = eos.property_modifiers.calculate_property_modifications(
            self)

        if self.method is None:
            raise AttributeError(
                "no method set for mineral, or equation_of_state given in mineral.params")

    """
    Properties from equations of state
    We choose the P, T properties (e.g. Gibbs(P, T) rather than Helmholtz(V, T)),
    as it allows us to more easily apply corrections to the free energy
    """
    @material_property
    @copy_documentation(Material.molar_gibbs)
    def molar_gibbs(self):
        return self.method.molar_gibbs_free_energy(self.pressure, self.temperature, self.params) \
            + self._property_modifiers['G']

    @material_property
    def _molar_volume_unmodified(self):
        return self.method.molar_volume(self.pressure, self.temperature, self.params)

    @material_property
    @copy_documentation(Material.molar_volume)
    def molar_volume(self):
        return self._molar_volume_unmodified \
            + self._property_modifiers['dGdP']

    @material_property
    @copy_documentation(Material.molar_entropy)
    def molar_entropy(self):
        return self.method.molar_entropy(self.pressure, self.temperature, self.params) \
            - self._property_modifiers['dGdT']

    @material_property
    @copy_documentation(Material.molar_heat_capacity_p)
    def molar_heat_capacity_p(self):
        return self.method.molar_heat_capacity_p(self.pressure,
                                                 self.temperature,
                                                 self.params) \
            - self.temperature * self._property_modifiers['d2GdT2']

    @material_property
    @copy_documentation(Material.thermal_expansivity)
    def thermal_expansivity(self):
        return (
            (self.method.volumetric_thermal_expansivity(self.pressure,
                                                        self.temperature,
                                                        self.params)
             * self._molar_volume_unmodified)
            + self._property_modifiers['d2GdPdT']) / self.molar_volume



    def _voigt_index_to_ij(self, m):
        """
        Returns the ij (or kl) indices of the
        stiffness tensor which correspond to those
        of the Voigt notation m (or n).
        """
        if m == 3:
            return 1, 2
        elif m == 4:
            return 0, 2
        elif m == 5:
            return 0, 1
        else:
            return m, m

    def _voigt_notation_to_fourth_rank_tensor(self, voigt_notation):
        """
        Converts a tensor in Voigt notation (6x6 matrix)
        to the full fourth rank tensor (3x3x3x3 matrix).
        """
        tensor = np.zeros([3, 3, 3, 3])
        for m in range(6):
            i, j = self._voigt_index_to_ij(m)
            for n in range(6):
                k, l = self._voigt_index_to_ij(n)
                tensor[i][j][k][l] = voigt_notation[m][n]
                tensor[j][i][k][l] = voigt_notation[m][n]
                tensor[i][j][l][k] = voigt_notation[m][n]
                tensor[j][i][l][k] = voigt_notation[m][n]
        return tensor


    # Tensor properties
    @material_property
    def unit_cell_vectors(self):
        return self.method.unit_cell_vectors(self.pressure,
                                             self.temperature,
                                             self.params)

    @material_property
    def thermal_expansivity_tensor(self):
        alpha = self.method.thermal_expansivity_tensor(self.pressure,
                                                       self.temperature,
                                                       self.params)

        return alpha

    @material_property
    def isothermal_elastic_compliance_tensor(self):
        return self.method.isothermal_elastic_compliance_tensor(self.pressure,
                                                                self.temperature,
                                                                self.params)

    @material_property
    def isentropic_elastic_compliance_tensor(self):
        return self.method.isentropic_elastic_compliance_tensor(self.pressure,
                                                                self.temperature,
                                                                self.params)

    @material_property
    def isothermal_elastic_stiffness_tensor(self):
        return self.method.isothermal_elastic_stiffness_tensor(self.pressure,
                                                               self.temperature,
                                                               self.params)

    @material_property
    def isentropic_elastic_stiffness_tensor(self):
        return self.method.isentropic_elastic_stiffness_tensor(self.pressure,
                                                               self.temperature,
                                                               self.params)

    @material_property
    def full_isothermal_elastic_stiffness_tensor(self):
        return self._voigt_notation_to_stiffness_tensor(self.isothermal_elastic_stiffness_tensor)

    @material_property
    def full_isentropic_elastic_stiffness_tensor(self):
        return self._voigt_notation_to_stiffness_tensor(self.isentropic_elastic_stiffness_tensor)

    @material_property
    def full_isothermal_elastic_compliance_tensor(self):
        try:  # numpy.block was new in numpy version 1.13.0.
            block = np.block([[np.ones((3, 3)), 2.*np.ones((3, 3))],
                              [2.*np.ones((3, 3)), 4.*np.ones((3, 3))]])
        except:
            block = np.array(np.bmat([[[[1.]*3]*3, [[2.]*3]*3], [[[2.]*3]*3, [[4.]*3]*3]] ))
        return self._voigt_notation_to_stiffness_tensor(np.divide(self.isothermal_elastic_compliance_tensor, block))

    @material_property
    def full_isentropic_elastic_compliance_tensor(self):
        try: # numpy.block was new in numpy version 1.13.0.
            block = np.block([[np.ones((3, 3)), 2.*np.ones((3, 3))],
                              [2.*np.ones((3, 3)), 4.*np.ones((3, 3))]])
        except:
            block = np.array(np.bmat([[[[1.]*3]*3, [[2.]*3]*3], [[[2.]*3]*3, [[4.]*3]*3]] ))
        return self._voigt_notation_to_stiffness_tensor(np.divide(self.isentropic_elastic_compliance_tensor, block))

    # Scalar properties (derived from tensors above)
    @material_property
    def isothermal_compressibility_reuss(self):
        beta_T_reuss = np.sum(np.sum(self.isothermal_elastic_compliance_tensor[:3,:3]))
        return beta_T_reuss

    @material_property
    def isentropic_compressibility_reuss(self):
        beta_T_reuss = np.sum(np.sum(self.isentropic_elastic_compliance_tensor[:3,:3]))
        return beta_T_reuss

    @material_property
    def isothermal_bulk_modulus_reuss(self):
        #return 1./self.isothermal_compressibility_reuss
        return self.method.isothermal_bulk_modulus_reuss(self.pressure,
                                                         self.temperature,
                                                         self.params)

    @material_property
    def isentropic_bulk_modulus_reuss(self):
        return 1./self.isentropic_compressibility_reuss

    @material_property
    def isothermal_bulk_modulus_voigt(self):
        """
        Computes the isothermal bulk modulus (Voigt bound)
        """
        K = np.sum(np.sum(self.isothermal_elastic_stiffness_tensor[:3, :3]))/9.
        return K

    @material_property
    def isentropic_bulk_modulus_voigt(self):
        """
        Computes the isentropic bulk modulus (Voigt bound)
        """
        K = np.sum(np.sum(self.isentropic_elastic_stiffness_tensor[:3, :3]))/9.
        return K

    @material_property
    def isothermal_compressibility_voigt(self):
        return 1./self.isothermal_bulk_modulus_voigt

    @material_property
    def isentropic_compressibility_voigt(self):
        return 1./self.isentropic_bulk_modulus_voigt

    @material_property
    def isothermal_bulk_modulus_vrh(self):
        """
        Computes the isothermal bulk modulus (Voigt-Reuss-Hill average)
        """
        return 0.5*(self.isothermal_bulk_modulus_voigt
                    + self.isothermal_bulk_modulus_reuss)

    @material_property
    def isentropic_bulk_modulus_vrh(self):
        """
        Computes the isothermal bulk modulus (Voigt-Reuss-Hill average)
        """
        return 0.5*(self.isentropic_bulk_modulus_voigt
                    + self.isentropic_bulk_modulus_reuss)

    @material_property
    def shear_modulus_voigt(self):
        """
        Computes the shear modulus (Voigt bound)
        """
        G = (np.sum([self.isentropic_elastic_stiffness_tensor[i][i]
                     for i in [0, 1, 2]])
             + np.sum([self.isentropic_elastic_stiffness_tensor[i][i]
                       for i in [3, 4, 5]])*3.
             - (self.isentropic_elastic_stiffness_tensor[0][1]
                + self.isentropic_elastic_stiffness_tensor[1][2]
                + self.isentropic_elastic_stiffness_tensor[2][0])) / 15.
        return G

    @material_property
    def shear_modulus_reuss(self):
        """
        Computes the shear modulus (Reuss bound)
        """
        beta = (np.sum([self.isentropic_elastic_compliance_tensor[i][i]
                        for i in [0, 1, 2]])*4.
                + np.sum([self.isentropic_elastic_compliance_tensor[i][i]
                          for i in [3, 4, 5]])*3.
                - (self.isentropic_elastic_compliance_tensor[0][1]
                   + self.isentropic_elastic_compliance_tensor[1][2]
                   + self.isentropic_elastic_compliance_tensor[2][0])*4.) / 15.
        return 1./beta

    @material_property
    def shear_modulus_vrh(self):
        """
        Computes the shear modulus (Voigt-Reuss-Hill average)
        """
        return 0.5*(self.shear_modulus_voigt + self.shear_modulus_reuss)

    @material_property
    def p_wave_velocity_voigt(self):
        return np.sqrt((self.isentropic_bulk_modulus_voigt
                        + 4. / 3. * self.shear_modulus_voigt) / self.density)

    @material_property
    def bulk_sound_velocity_voigt(self):
        return np.sqrt(self.isentropic_bulk_modulus_voigt / self.density)

    @material_property
    def shear_wave_velocity_voigt(self):
        return np.sqrt(self.shear_modulus_voigt / self.density)

    @material_property
    def p_wave_velocity_reuss(self):
        return np.sqrt((self.isentropic_bulk_modulus_reuss
                        + 4. / 3. * self.shear_modulus_reuss) / self.density)

    @material_property
    def bulk_sound_velocity_reuss(self):
        return np.sqrt(self.isentropic_bulk_modulus_reuss / self.density)

    @material_property
    def shear_wave_velocity_reuss(self):
        return np.sqrt(self.shear_modulus_reuss / self.density)

    @material_property
    def p_wave_velocity_vrh(self):
        return np.sqrt((self.isentropic_bulk_modulus_vrh
                        + 4. / 3. * self.shear_modulus_vrh) / self.density)

    @material_property
    def bulk_sound_velocity_vrh(self):
        return np.sqrt(self.isentropic_bulk_modulus_vrh / self.density)

    @material_property
    def shear_wave_velocity_vrh(self):
        return np.sqrt(self.shear_modulus_vrh / self.density)

    @material_property
    def universal_elastic_anisotropy(self):
        """
        Compute the universal (isentropic) elastic anisotropy
        """
        return (5.*(self.shear_modulus_voigt/self.shear_modulus_reuss)
                + (self.isentropic_bulk_modulus_voigt
                   / self.isentropic_bulk_modulus_reuss) - 6.)

    @material_property
    def isotropic_poisson_ratio(self):
        """
        Compute mu, the isotropic Poisson ratio
        (a description of the laterial response to loading)
        """
        return ((3.*self.isentropic_bulk_modulus_vrh
                 - 2.*self.shear_modulus_vrh)
                / (6.*self.isentropic_bulk_modulus_vrh
                   + 2.*self.shear_modulus_vrh))

    @material_property
    def grueneisen_parameter(self):
        return (self.thermal_expansivity
                * self.isothermal_bulk_modulus_reuss
                * self.molar_volume
                / self.molar_heat_capacity_v)

    def christoffel_tensor(self, propagation_direction):
        """
        Computes the Christoffel tensor from an elastic stiffness
        tensor and a propagation direction for a seismic wave
        relative to the stiffness tensor

        T_ik = C_ijkl n_j n_l
        """
        propagation_direction = unit_normalize(propagation_direction)
        Tik = np.tensordot(np.tensordot(self.full_isentropic_elastic_stiffness_tensor,
                                        propagation_direction,
                                        axes=([1], [0])),
                           propagation_direction,
                           axes=([2], [0]))
        return Tik

    def linear_compressibility(self, direction):
        """
        Computes the linear compressibility in a given direction
        relative to the stiffness tensor
        """
        direction = unit_normalize(direction)
        Sijkk = np.einsum('ijkk', self.full_isentropic_elastic_compliance_tensor)
        beta = Sijkk.dot(direction).dot(direction)
        return beta

    def youngs_modulus(self, direction):
        """
        Computes the Youngs modulus in a given direction
        relative to the stiffness tensor
        """
        direction = unit_normalize(direction)
        Sijkl = self.full_isentropic_elastic_compliance_tensor
        S = Sijkl.dot(direction).dot(direction).dot(direction).dot(direction)
        return 1./S

    def shear_modulus(self, plane_normal, shear_direction):
        """
        Computes the shear modulus on a plane in a given
        shear direction relative to the stiffness tensor
        """
        plane_normal = unit_normalize(plane_normal)
        shear_direction = unit_normalize(shear_direction)

        assert np.abs(plane_normal.dot(shear_direction)) < np.finfo(np.float).eps, 'plane_normal and shear_direction must be orthogonal'
        Sijkl = self.full_isentropic_elastic_compliance_tensor
        G = Sijkl.dot(shear_direction).dot(plane_normal).dot(shear_direction).dot(plane_normal)
        return 0.25/G

    def poissons_ratio(self,
                       axial_direction,
                       lateral_direction):
        """
        Computes the poisson ratio given loading and response
        directions relative to the stiffness tensor
        """

        axial_direction = unit_normalize(axial_direction)
        lateral_direction = unit_normalize(lateral_direction)
        assert np.abs(axial_direction.dot(lateral_direction)) < np.finfo(np.float).eps, 'axial_direction and lateral_direction must be orthogonal'

        Sijkl = self.full_isentropic_elastic_compliance_tensor
        x = axial_direction
        y = lateral_direction
        nu = -(Sijkl.dot(y).dot(y).dot(x).dot(x)
               / Sijkl.dot(x).dot(x).dot(x).dot(x))
        return nu

    def wave_velocities(self, propagation_direction):
        """
        Computes the compressional wave velocity, and two
        shear wave velocities in a given propagation direction

        Returns two lists, containing the wave speeds and
        directions of particle motion relative to the stiffness tensor
        """
        propagation_direction = unit_normalize(propagation_direction)

        Tik = self.christoffel_tensor(propagation_direction)

        eigenvalues, eigenvectors = np.linalg.eig(Tik)

        idx = eigenvalues.argsort()[::-1]
        eigenvalues = np.real(eigenvalues[idx])
        eigenvectors = eigenvectors[:, idx]
        velocities = np.sqrt(eigenvalues/self.rho)

        return velocities, eigenvectors

    """
    Properties from mineral parameters,
    Legendre transformations
    or Maxwell relations
    """
    @material_property
    def formula(self):
        """
        Returns the chemical formula of the Mineral class
        """
        if 'formula' in self.params:
            return self.params['formula']
        else:
            raise ValueError(
                'No formula parameter for mineral {0}.'.format(self.to_string))

    @material_property
    @copy_documentation(Material.molar_mass)
    def molar_mass(self):
        if 'molar_mass' in self.params:
            return self.params['molar_mass']
        else:
            raise ValueError('No molar_mass parameter for mineral '
                             '{0}.'.format(self.to_string))

    @material_property
    @copy_documentation(Material.density)
    def density(self):
        return self.molar_mass / self.molar_volume

    @material_property
    @copy_documentation(Material.molar_internal_energy)
    def molar_internal_energy(self):
        return (self.molar_gibbs
                - self.pressure * self.molar_volume
                + self.temperature * self.molar_entropy)

    @material_property
    @copy_documentation(Material.molar_helmholtz)
    def molar_helmholtz(self):
        return self.molar_gibbs - self.pressure * self.molar_volume

    @material_property
    @copy_documentation(Material.molar_enthalpy)
    def molar_enthalpy(self):
        return self.molar_gibbs + self.temperature * self.molar_entropy

    @material_property
    @copy_documentation(Material.molar_heat_capacity_v)
    def molar_heat_capacity_v(self):
        return (self.molar_heat_capacity_p
                - self.molar_volume * self.temperature
                * self.thermal_expansivity * self.thermal_expansivity
                * self.isothermal_bulk_modulus_reuss)
