from burnman import constants
from burnman.processchemistry import *

from scipy.misc import factorial
from scipy.optimize import fsolve

import numpy as np

atomic_masses=read_masses()

# Equations are as implemented in de Koker and Stixrude (2013; Supplementary Materials)
# and in the PhD thesis of de Koker (2008)

# Atomic momenta
def _partition_function(mass, temperature): # eq. S7; qi/V. Note the typo - there should be temperature in this expression
    # Should V=Vi, rather than Vensemble?
    return np.power(mass*constants.Boltzmann*temperature/(2*np.pi*constants.Dirac*constants.Dirac), 3./2.)

def _atomic_momenta(volume, temperature, params): # F_ig, eq. S6
    # ideal gas
    entropy_sum=0.
    for element, N in params['formula'].iteritems(): # N is a.p.f.u
        q=(volume/constants.Avogadro)*_partition_function((atomic_masses[element]/constants.Avogadro), temperature) # masses are in kg/mol
        entropy_sum+=N*(1. + np.log(q/N)) # see also eq. 16.72 of Callen., 1985; p. 373
    return -constants.gas_constant*temperature*entropy_sum

'''
def _atomic_momenta(volume, temperature, params): # F_ig, thesis
    # total number of atoms
    n_atoms=0
    for element, N in params['formula'].iteritems():
        n_atoms += N
    # sum contributions
    F_ig=0.
    for element, N in params['formula'].iteritems(): # N is a.p.f.u
        F_iga = -constants.gas_constant*temperature \
            * (np.log(volume/n_atoms/constants.Avogadro) + np.log(_partition_function(atomic_masses[element]/constants.Avogadro, temperature)) + 1.)
        F_ig += N*F_iga + constants.gas_constant*temperature \
            * N*np.log(N/n_atoms)
    return F_ig
'''

def _atomic_entropy(volume, temperature, params): # F_ig, eq. S6
    # ideal gas
    entropy_sum=0.
    for element, N in params['formula'].iteritems(): # N is a.p.f.u
        q=(volume/constants.Avogadro)*_partition_function((atomic_masses[element]/constants.Avogadro), temperature) # masses are in kg/mol
        entropy_sum+=N*(5./2. + np.log(q/N)) # see also eq. 16.72 of Callen., 1985; p. 373
    return constants.gas_constant*entropy_sum


def _atomic_pressure(volume, temperature, params): # PV = nRT
    n_atoms=0
    for element, N in params['formula'].iteritems():
        n_atoms += N
    return n_atoms*constants.gas_constant*temperature / volume


# Finite strain
def _finite_strain(volume, temperature, params): # f(V), eq. S3a
    return (1./2.)*(np.power(params['V_0']/volume, 2./3.) - 1.0)        

def _dfdV(volume, temperature, params): # f(V), eq. S3a
    return (-1./3.)*np.power(params['V_0']/volume, 2./3.)/volume  


# Temperature
def _theta(volume, temperature, params): # theta, eq. S3b
    return np.power(temperature/params['T_0'], params['m']) - 1.

def _dthetadT(volume, temperature, params):
    return params['m']*np.power(temperature/params['T_0'], params['m']) \
        / temperature

# Electronic component
def _zeta(volume, temperature, params): # eq. S5a, beta in deKoker thesis (3.34)
    return params['zeta_0']*(np.power(volume/params['el_V_0'], params['xi']))

def _dzetadV(volume, temperature, params):
    return params['xi']*params['zeta_0']*(np.power(volume/params['el_V_0'], params['xi']))/volume

def _Tel(volume, temperature, params): # eq. S5b
    return params['Tel_0']*(np.power(volume/params['el_V_0'], params['eta']))
                            
def _dTeldV(volume, temperature, params):
    return params['eta']*params['Tel_0']*(np.power(volume/params['el_V_0'], params['eta']))/volume
                  
def _electronic_excitation_energy(volume, temperature, params): # F_el
    temperature_el=_Tel(volume, temperature, params)
    if temperature < temperature_el:
        F_el = 0
    else:
        F_el = -_zeta(volume, temperature, params)*(0.5*(temperature*temperature - temperature_el*temperature_el) - temperature*temperature_el*np.log(temperature/temperature_el))
    return F_el

def _electronic_excitation_entropy(volume, temperature, params): # P_el
    temperature_el=_Tel(volume, temperature, params)
    if temperature < temperature_el:
        S_el = 0
    else:
        S_el = _zeta(volume, temperature, params)*( temperature - temperature_el - temperature_el*np.log(temperature/temperature_el))
    return S_el

def _electronic_excitation_pressure(volume, temperature, params): # P_el
    temperature_el=_Tel(volume, temperature, params)
    if temperature < temperature_el:
        P_el = 0
    else:
        P_el =  _dzetadV(volume, temperature, params) * (0.5*(temperature*temperature - temperature_el*temperature_el) - temperature*temperature_el*np.log(temperature/temperature_el))
        P_el += _zeta(volume, temperature, params)*_dTeldV(volume, temperature, params)*((temperature-temperature_el) - temperature*np.log(temperature/temperature_el))
    return P_el


# Weird element order correcting:
def vector_to_array(a, Of, Otheta):
    array=np.empty([Of+1, Otheta+1])
    for i in range(Of+1):
        for j in range(Otheta+1):
            n=int((i+j)*((i+j)+1.)/2. + j)
            array[i][j]=a[n]
    return array

# Bonding energy
def _bonding_energy(volume, temperature, params): # F_xs, eq. S2
    f=_finite_strain(volume, temperature, params)
    theta=_theta(volume, temperature, params)
    energy = 0.
    for i in range(len(params['a'])):
        ifact=factorial(i, exact=False)
        for j in range(len(params['a'][0])):
            jfact=factorial(j, exact=False)
            energy += params['a'][i][j]*np.power(f, i)*np.power(theta, j)/ifact/jfact         
    return energy

def _bonding_entropy(volume, temperature, params): # F_xs, eq. 3.18
    f=_finite_strain(volume, temperature, params)
    theta=_theta(volume, temperature, params)
    entropy = 0.
    for i in range(len(params['a'])):
        ifact=factorial(i, exact=False)
        for j in range(len(params['a'][0])):
            if j > 0:
                jfact=factorial(j, exact=False)
                entropy += j*params['a'][i][j]*np.power(f, i)*np.power(theta, j-1.)/ifact/jfact         
    return -_dthetadT(volume, temperature, params)*entropy

def _bonding_pressure(volume, temperature, params): # P_xs, eq. 3.17 of de Koker thesis
    f=_finite_strain(volume, temperature, params)
    theta=_theta(volume, temperature, params)
    pressure=0.
    for i in range(len(params['a'])):
        ifact=factorial(i, exact=False)
        if i > 0:
            for j in range(len(params['a'][0])):
                jfact=factorial(j, exact=False)
                pressure += float(i)*params['a'][i][j]*np.power(f, float(i)-1.)*np.power(theta, float(j))/ifact/jfact
    return -_dfdV(volume, temperature, params)*pressure

# Pressure
def _pressure_liquid(volume, temperature, params):
    return _atomic_pressure(volume, temperature, params) + _electronic_excitation_pressure(volume, temperature, params) + _bonding_pressure(volume, temperature, params)

def _volume_at_pressure(volume, pressure, temperature, params):
    return pressure - _pressure_liquid(volume[0], temperature, params)

def volume_liquid(pressure, temperature, params):
    return fsolve(_volume_at_pressure, 0.8e-6, args=(pressure, temperature, params), xtol=1e-12, full_output=True)[0][0]

def helmholtz_liquid(volume, temperature, params): # F(V,T), eq. S1
    return _atomic_momenta(volume, temperature, params) + _electronic_excitation_energy(volume, temperature, params) + _bonding_energy(volume, temperature, params)

def gibbs_liquid(pressure, temperature, params):
    volume=fsolve(_volume_at_pressure, params['V_0']/2., args=(pressure, temperature, params))[0]
    return helmholtz_liquid(volume, temperature, params) + pressure*volume
  
def entropy_liquid(volume, temperature, params):
    return _atomic_entropy(volume, temperature, params) + _electronic_excitation_entropy(volume, temperature, params) + _bonding_entropy(volume, temperature, params)

def energy_liquid(volume, temperature, params):
    return helmholtz_liquid(volume, temperature, params) \
        + temperature*entropy_liquid(volume, temperature, params)
