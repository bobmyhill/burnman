# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.

from __future__ import absolute_import
from __future__ import print_function
import warnings


from subprocess import Popen, PIPE, STDOUT
from os import rename


import numpy as np
import scipy.optimize as opt
from scipy.interpolate import interp2d 

from .mineral import Mineral, material_property
from . import eos
from .tools import bracket
from .tools import copy_documentation

from .processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses = read_masses()

def _read_1D_sesame(filename):
    with open(filename, 'r') as f:
        datastream = f.read()
        lines = np.array([[float(item.replace('D', 'E')) for item in line.strip().split()]
                          for line in datastream.split('\n') if line.strip()])
        values = []
        for line in lines:
            values.extend(line)
            
        return np.array(values)

def _read_2D_sesame(filename, factor, densities, temperatures, kind):
    '''
    Returns an interp2d function
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

class SesameMineral(Mineral):
    """
    This is the base class for a sesame mineral. States of the material
    can only be queried after setting the pressure and temperature
    using set_state(). 

    Instances of this class are initialised with 
    directory 301 of a sesame equation of state and a chemical formula. 
    The directory should contain the following files: 
    'A', 'E', 'P', 'R', 'T', where these files contain
    the Helmholtz free energy, internal energy, pressure 
    (all 2D files in MJ/kg, MJ/kg and GPa), and densities and temperatures
    (1D files, in Mg/m^3 and K). 

    Properties of the material are determined by cubic spline interpolation 
    from the grids created from the input files. They are all returned in 
    SI units on a molar basis, even though the files are not in these units.

    This class is available as ``burnman.SesameMineral``.
    """
    def __init__(self, directory, formula, interp_method='quintic'):
        formula = dictionarize_formula(formula)
        molar_mass = formula_mass(formula, atomic_masses)
        self.params = {'name': directory,
                       'formula': formula,
                       'n': sum(formula.values()),
                       'molar_mass': molar_mass}
        self._property_interpolators = self._read_sesame(directory, molar_mass, interp_method)
        Mineral.__init__(self)
        
        class SesameMethod(object):
            """
            Dummy class because Mineral.set_state() needs a method
            """
            pass
        self.method = SesameMethod()

    def _read_sesame(self, directory, molar_mass, kind):
        temperatures = _read_1D_sesame(directory+'/T') # K
        densities = _read_1D_sesame(directory+'/R')*1.e3 # Mg/m^3
        f_energy = _read_2D_sesame(directory+'/E', molar_mass*1.e6, densities, temperatures, kind) # MJ/kg
        f_helmholtz_energy = _read_2D_sesame(directory+'/A', molar_mass*1.e6, densities, temperatures, kind) # MJ/kg
        f_pressure = _read_2D_sesame(directory+'/P', 1.e9, densities, temperatures, kind)
        
        property_interpolators = {'E': f_energy,
                                  'A': f_helmholtz_energy,
                                  'P': f_pressure}
        
        return property_interpolators

    def set_state(self, pressure, temperature):
        Mineral.set_state(self, pressure, temperature)
        #self.density = self._density
        
    """
    Properties by linear interpolation of Perple_X output
    """
    @material_property
    def molar_mass(self):
        if 'molar_mass' in self.params:
            return self.params['molar_mass']
        else:
            raise ValueError(
                "No molar_mass parameter for mineral " + self.to_string + ".")

    @material_property
    def density(self):

        _delta_pressure = lambda rho: (self.pressure -
                                       self._property_interpolators['P']([rho],
                                                                         [self.temperature])[0])

        # we need to have a sign change in [a,b] to find a zero.
        # Let us start with a conservative guess for the density:
        try:
            sol = bracket(_delta_pressure, 1.e5, 1.e1)
        except ValueError:
            raise Exception(
                'Cannot find a volume, perhaps you are outside of the range of validity for the equation of state?')
        return opt.brentq(_delta_pressure, sol[0], sol[1])

    
    @material_property
    def molar_volume(self):
        return self.molar_mass/self.density

    @material_property
    def internal_energy(self):
        return self._property_interpolators['E']([self.density], [self.temperature])[0]

    @material_property
    def molar_helmholtz(self):
        return self._property_interpolators['A']([self.density], [self.temperature])[0]

    @material_property
    def molar_enthalpy(self):
        return self.internal_energy + self.pressure*self.molar_volume
    
    @material_property
    def molar_gibbs(self):
        return self.molar_helmholtz + self.pressure*self.molar_volume

    @material_property
    def isothermal_bulk_modulus(self):
        return self.density * \
            self._property_interpolators['P']([self.density], [self.temperature], dx=1)[0]

    @material_property
    def isothermal_compressibility(self):
        return 1. / self.isothermal_bulk_modulus

    @material_property
    def molar_entropy(self):
        return -self._property_interpolators['A']([self.density], [self.temperature], dy=1)[0]

    @material_property
    def heat_capacity_v(self):
        return -self.temperature * \
            self._property_interpolators['A']([self.density], [self.temperature], dy=2)[0]
    
    @material_property
    def thermal_expansivity(self):
        return self.isothermal_compressibility * \
            self._property_interpolators['P']([self.density], [self.temperature], dy=1)[0]

    '''
    Derived properties
    '''

    @material_property
    def heat_capacity_p(self):
        return self.heat_capacity_v + self.molar_volume * self.temperature \
            * self.thermal_expansivity * self.thermal_expansivity \
            * self.isothermal_bulk_modulus

    @material_property
    def adiabatic_bulk_modulus(self):
        return self.isothermal_bulk_modulus * self.heat_capacity_v / self.heat_capacity_p
    
    @material_property
    def adiabatic_compressibility(self):
        return 1. / self.adiabatic_bulk_modulus

    @material_property
    def grueneisen_parameter(self):
        return ( self.thermal_expansivity *
                 self.isothermal_bulk_modulus  /
                 (self.heat_capacity_v * self.density) )

    @material_property
    def bulk_sound_velocity(self):
        return np.sqrt( self.isothermal_bulk_modulus  /
                        self.density )
    
    @material_property
    def shear_modulus(self):
        raise NotImplementedError("sesame model has no implementation of shear modulus")
    
    @material_property
    def p_wave_velocity(self):
        raise NotImplementedError("sesame model has no implementation of p wave velocity")
        
    @material_property
    def shear_wave_velocity(self):
        raise NotImplementedError("sesame model has no implementation of s wave velocity")
