# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU GPL v2 or later.


import warnings

import numpy as np

from burnman.material import Material
import burnman.eos as eos


class Mineral(Material):
    """
    This is the base class for all minerals. States of the mineral
    can only be queried after setting the pressure and temperature
    using set_state(). The method for computing properties of
    the material is set using set_method(). This is done during 
    initialisation if the param 'equation_of_state' has been defined.
    The method can be overridden later by the user.

    This class is available as ``burnman.Mineral``.

    If deriving from this class, set the properties in self.params
    to the desired values. For more complicated materials you
    can overwrite set_state(), change the params and then call
    set_state() from this class.

    All the material parameters are expected to be in plain SI units.  This
    means that the elastic moduli should be in Pascals and NOT Gigapascals,
    and the Debye temperature should be in K not C.  Additionally, the
    reference volume should be in m^3/(mol molecule) and not in unit cell
    volume and 'n' should be the number of atoms per molecule.  Frequently in
    the literature the reference volume is given in Angstrom^3 per unit cell.
    To convert this to m^3/(mol of molecule) you should multiply by 10^(-30) *
    N_a / Z, where N_a is Avogadro's number and Z is the number of formula units per
    unit cell. You can look up Z in many places, including www.mindat.org
    """

    def __init__(self):
        if 'params' not in self.__dict__:
            self.params = {}
        self.method = None
        if 'equation_of_state' in self.params:
            self.set_method(self.params['equation_of_state'])
        self.eos_data = {}

    def set_method(self, equation_of_state):
        """
        Set the equation of state to be used for this mineral.
        Takes a string corresponding to any of the predefined
        equations of state:  'bm2', 'bm3', 'mgd2', 'mgd3', 'slb2', 'slb3',
        'mt', 'hp_tmt', or 'cork'.  Alternatively, you can pass a user defined
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
            raise Exception("Please derive your method from object (see python old style classes)")

        if self.method is not None and type(new_method) is not type(self.method):

            # Warn user that they are changing the EoS
            warnings.warn('Warning, you are changing the method to ' + new_method.__class__.__name__ + ' even though the material is designed to be used with the method ' + self.method.__class__.__name__ + '.  This does not overwrite any mineral attributes other than temperature and pressure, which are set to nan. Stale attributes will be preserved unless they are refreshed (for example, by set_state).', stacklevel=2)

            try:
                self.pressure=self.temperature=float('nan')
            except AttributeError:
                pass

        self.method = new_method

        #Validate the params object on the requested EOS. 
        try:
            self.method.validate_parameters(self.params)
        except Exception as e:
            print 'Mineral ' + self.to_string() + ' failed to validate parameters with message : \" ' + e.message + '\"'
            raise

    def to_string(self):
        """
        Returns the name of the mineral class
        """
        return "'" + self.__class__.__module__.replace(".minlib_",".") + "." + self.__class__.__name__ + "'"

    def debug_print(self, indent=""):
        print "%s%s" % (indent, self.to_string())

    def unroll(self):
        return ([1.0],[self])

    def eos_pressure(self, temperature, volume):
        return self.method.pressure(temperature, volume, self.params)

    def set_state(self, pressure, temperature):
        """
        Update the material to the given pressure [Pa] and temperature [K].

        This updates the other properties of this class (v_s, v_p, ...).
        """

        self.pressure = pressure
        self.temperature = temperature
        self.old_params = self.params
        self.eos_data = {}

        if self.method is None:
            raise AttributeError, "no method set for mineral, or equation_of_state given in mineral.params"

        # Wipe attributes
        self.V = self.gr = self.K_T = self.K_S \
        = self.G = self.C_v = self.C_p = self.alpha \
        = self.gibbs = self.helmholtz = self.S = self.H = None

        del self.V
        del self.gr
        del self.K_T
        del self.K_S
        del self.G
        del self.C_v
        del self.C_p
        del self.alpha
        del self.gibbs
        del self.helmholtz
        del self.S
        del self.H
            
        self.V = self.method.volume(self)
        self.K_T = self.method.isothermal_bulk_modulus(self)
        self.K_S = self.method.adiabatic_bulk_modulus(self)

        self.gr = self.method.grueneisen_parameter(self)
        self.alpha = self.method.thermal_expansivity(self)

        self.G = self.method.shear_modulus(self)

        self.C_p = self.method.heat_capacity_p(self)
        self.C_v = self.method.heat_capacity_v(self)

        # Attempt to calculate the gibbs free energy and helmholtz free energy, 
        # but don't complain if the equation of state does not calculate it, 
        # or if the mineral params do not have the requisite entries.
        try:
            self.gibbs = self.method.gibbs_free_energy(self)
        except (KeyError, NotImplementedError):
            self.gibbs = float('nan')
        try:
            self.helmholtz = self.method.helmholtz_free_energy(self)
        except (KeyError, NotImplementedError):
            self.helmholtz = float('nan')
        try:
            self.S = self.method.entropy(self)
        except (KeyError, NotImplementedError):
            self.S = float('nan')
        try:
            self.H = self.method.enthalpy(self)
        except (KeyError, NotImplementedError):
            self.H = float('nan')


    # The following gibbs function avoids having to calculate a bunch of unnecessary parameters over P-T space. This will be useful for gibbs minimisation.
    def calcgibbs(self, pressure, temperature):
        return self.method.gibbs_free_energy(pressure, temperature, self.params)

    def molar_mass(self):
        """
        Returns molar mass of the mineral [kg/mol]
        """
        return self.params['molar_mass']

    def density(self):
        """
        Returns density of the mineral [kg/m^3]
        """
        return  self.params['molar_mass'] / self.V

    def molar_volume(self):
        """
        Returns molar volume of the mineral [m^3/mol]
        """
        return self.V

    def grueneisen_parameter(self):
        """
        Returns grueneisen parameter of the mineral [unitless]
        """
        try:
            return self.gr
        except:
            self.gr = self.method.grueneisen_parameter(self)
            return self.gr


    def isothermal_bulk_modulus(self):
        """
        Returns isothermal bulk modulus of the mineral [Pa]
        """
        try:
            return self.K_T 
        except:
            self.K_T = self.method.isothermal_bulk_modulus(self)
            return self.K_T

    def compressibility(self):
        """
        Returns isothermal bulk modulus of the mineral [Pa]
        """
        return 1./self.isothermal_bulk_modulus()

    def adiabatic_bulk_modulus(self):
        """
        Returns adiabatic bulk modulus of the mineral [Pa]
        """
        try:
            return self.K_S 
        except:
            self.K_S = self.method.adiabatic_bulk_modulus(self)
            return self.K_S
    
    def shear_modulus(self):
        """
        Returns shear modulus of the mineral [Pa]
        """
        try:
            return self.G 
        except:
            self.G = self.method.shear_modulus(self)
            return self.G

    def thermal_expansivity(self):
        """
        Returns thermal expansion coefficient of the mineral [1/K]
        """
        try:
            return self.alpha 
        except:
            self.alpha = self.method.thermal_expansivity(self)
            return self.alpha

    def heat_capacity_v(self):
        """
        Returns heat capacity at constant volume of the mineral [J/K/mol]
        """
        try:
            return self.C_v
        except:
            self.C_v = self.method.heat_capacity_v(self)
            return self.C_v

    def heat_capacity_p(self):
        """
        Returns heat capacity at constant pressure of the mineral [J/K/mol]
        """
        try: 
            return self.C_p
        except:
            self.C_p = self.method.heat_capacity_p(self)
            return self.C_p

    def v_s(self):
        """
        Returns shear wave speed of the mineral [m/s]
        """
        return np.sqrt(self.shear_modulus() / \
            self.density())
    def v_p(self):
        """
        Returns P wave speed of the mineral [m/s]
        """
        return np.sqrt((self.adiabatic_bulk_modulus() + 4. / 3. * \
            self.shear_modulus()) / self.density())

    def v_phi(self):
        """
        Returns bulk sound speed of the mineral [m/s]
        """
        return np.sqrt(self.adiabatic_bulk_modulus() / self.density())

    def molar_gibbs(self):
        """
        Returns Gibbs free energy of the mineral [J]
        """
        self.gibbs = self.gibbs or self.method.gibbs_free_energy(self)
        return self.gibbs

    def molar_helmholtz(self):
        """
        Returns Helmholtz free energy of the mineral [J]
        """
        try: 
            return self.helmholtz
        except:
            self.helmholtz = self.method.helmholtz_free_energy(self)
            return self.helmholtz

    def molar_enthalpy(self):
        """
        Returns enthalpy of the mineral [J]
        """
        try:
            return self.H 
        except:
            self.H = self.method.enthalpy(self)
            return self.H

    def molar_entropy(self):
        """
        Returns enthalpy of the mineral [J]
        """
        try:
            return self.S
        except:
            self.S = self.method.entropy(self)
            return self.S
