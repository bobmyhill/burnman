# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.optimize as opt
from scipy.interpolate import interp2d

from ..tools.math import bracket
from itertools import chain
from ..tools.chemistry import formula_mass
from . import equation_of_state as eos


def _read_1D_sesame(filename):
    """
    Utility function to read a Sesame table containing 1D data.
    This is an file containing only floats in scientific format.
    There are a variable number of floats on each line.

    Parameters
    ----------
    filename : string
        The name of the Sesame file.

    Returns
    -------
    value_array : numpy array
        Values contained within the file.
    """
    with open(filename, 'r') as f:
        datastream = f.read()
        lines = [[float(item.replace('D', 'E'))
                  for item in line.strip().split()]
                 for line in datastream.split('\n') if line.strip()]
        return np.array(list(chain.from_iterable(lines)))


def _read_2D_sesame(filename, factor, densities, temperatures, kind):
    '''
    Utility function to read in a structured Sesame file containing 2D data
    into an interpolation function.

    Parameters
    ----------
    filename : string
        The filename to read

    factor : float
        A factor by which to multiple all the values
        (used to convert to SI units).

    densities : numpy array
        The densities at which the file records values. Read from the
        appropriate 1D Sesame table.

    temperatures : numpy array
        The temperatures at which the file records values. Read from the
        appropriate 1D Sesame table.

    kind : string
        The kind of interpolation to be used by scipy.interpolate.interp2d.

    Returns
    -------
    values : scipy.interpolate.interp2d function
    '''
    with open(filename, 'r') as f:
        datastream = f.read()
        lines = [[item.replace('D', 'E') for item in line.strip().split()]
                 for line in datastream.split('\n') if line.strip()]
    values = []
    for line in lines:
        if line[0][0] == 'T':
            values.append([])
        else:
            values[-1].extend([float(i) for i in line])
    return interp2d(densities, temperatures, np.array(values)*factor, kind=kind)


class Sesame(eos.EquationOfState):
    """
    This is the base class for a sesame equation of state.

    Instances of this class are initialised with
    directory 301 of a sesame equation of state and a chemical formula.
    The directory should contain the following files:
    'A', 'E', 'P', 'R', 'T', where these files contain
    the Helmholtz free energy, internal energy, pressure
    (all 2D files in MJ/kg, MJ/kg and GPa), and densities and temperatures
    (1D files, in Mg/m^3 and K).

    Properties of the material are determined by quintic spline interpolation
    from the grids created from the input files. They are returned in
    SI units on a molar basis, even though the files are not in these units.
    """

    def validate_parameters(self, params):
        expected_keys = ['formula', 'directory']
        for k in expected_keys:
            if k not in params:
                raise KeyError('params object missing parameter : ' + k)

        if 'molar_mass' not in params:
            params['molar_mass'] = formula_mass(params['formula'])
        if 'n' not in params:
            params['n'] = sum(params['formula'].values())

        self._property_interpolators = self._read_sesame(params['directory'],
                                                         params['molar_mass'],
                                                         'quintic')

    def _read_sesame(self, directory, molar_mass, kind):
        """
        Reads the files in the Sesame 301 directory.

        Parameters
        ----------
        directory : string
            The name of the directory containing the Sesame files

        molar_mass : float
            The molar mass of the material

        kind : string
            The kind of interpolation to be used by scipy.interpolate.interp2d.
        """
        temperatures = _read_1D_sesame(directory+'/T')  # K
        densities = _read_1D_sesame(directory+'/R')*1.e3  # Mg/m^3
        f_energy = _read_2D_sesame(directory+'/E', molar_mass*1.e6,
                                   densities, temperatures, kind)  # MJ/kg
        f_helmholtz_energy = _read_2D_sesame(directory+'/A', molar_mass*1.e6,
                                             densities, temperatures,
                                             kind)  # MJ/kg
        f_pressure = _read_2D_sesame(directory+'/P', 1.e9,
                                     densities, temperatures, kind)

        property_interpolators = {'E': f_energy,
                                  'A': f_helmholtz_energy,
                                  'P': f_pressure}

        return property_interpolators

    """
    Properties obtained by linear interpolation of Sesame files
    """

    def density(self, pressure, temperature):
        def _delta_pressure(rho):
            return (pressure
                    - self._property_interpolators['P']([rho],
                                                        [temperature])[0])
        # we need to have a sign change in [a,b] to find a zero.
        # Let us start with a conservative guess for the density:
        try:
            sol = bracket(_delta_pressure, 1.e5, 1.e1)
        except ValueError:
            raise Exception(
                'Cannot find a volume, perhaps you are outside of the range '
                'of validity for the equation of state?')
        return opt.brentq(_delta_pressure, sol[0], sol[1])

    def volume(self, pressure, temperature, params):
        return params['molar_mass']/self.density(pressure, temperature)

    def molar_internal_energy(self, pressure, temperature, volume, params):
        density = params['molar_mass'] / volume
        return self._property_interpolators['E']([density],
                                                 [temperature])[0]

    def molar_helmholtz(self, pressure, temperature, volume, params):
        density = params['molar_mass'] / volume
        return self._property_interpolators['A']([density],
                                                 [temperature])[0]

    def gibbs_free_energy(self, pressure, temperature, volume, params):
        return (self.molar_helmholtz(pressure, temperature, volume, params)
                + pressure*volume)

    def isothermal_bulk_modulus(self, pressure, temperature, volume, params):
        density = params['molar_mass'] / volume
        return (density * self._property_interpolators['P']([density],
                                                            [temperature],
                                                            dx=1)[0])

    def entropy(self, pressure, temperature, volume, params):
        density = params['molar_mass'] / volume
        return -self._property_interpolators['A']([density],
                                                  [temperature], dy=1)[0]

    def molar_heat_capacity_v(self, pressure, temperature, volume, params):
        density = params['molar_mass'] / volume
        return (-temperature
                * self._property_interpolators['A']([density],
                                                    [temperature],
                                                    dy=2)[0])

    def thermal_expansivity(self, pressure, temperature, volume, params):
        density = params['molar_mass'] / volume
        return (self._property_interpolators['P']([density],
                                                  [temperature],
                                                  dy=1)[0]
                / self.isothermal_bulk_modulus(pressure, temperature,
                                               volume, params))

    '''
    Derived properties
    '''

    def molar_heat_capacity_p(self, pressure, temperature, volume, params):
        C_v = self.molar_heat_capacity_v(pressure, temperature, volume, params)
        alpha = self.thermal_expansivity(pressure, temperature, volume, params)
        K_T = self.isothermal_bulk_modulus(pressure, temperature, volume,
                                           params)

        return (C_v + volume * temperature * alpha * alpha * K_T)

    def shear_modulus(self, pressure, temperature, volume, params):
        """
        Not implemented.
        Returns 0.
        """
        return 0.
