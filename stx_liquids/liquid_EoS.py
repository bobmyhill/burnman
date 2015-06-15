import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.processchemistry import *
from burnman import constants
import numpy as np
from scipy.misc import factorial

atomic_masses=read_masses()

'''
# V_alpha and mass_a are the volume and mass of one particle of type alpha
def _partial_atomic_momenta(temperature, volume, element, N): # from deKoker thesis, eq. 3.14
    return -constants.gas_constant*temperature*(np.log(V_alpha) * 1.5*np.log(mass*constants.Boltzmann*temperature/(2*np.pi*constants.Dirac*constants.Dirac)) + 1.)

def _atomic_momenta(temperature, volume, params): # from deKoker thesis, eq. 3.13
    sumN=0.
    for element, N in params['formula'].iteritems():
        sumN+=N
        
    fig_sum=0.
    for element, N in params['formula'].iteritems():
        fig_sum+=N*_partial_atomic_momenta(temperature, volume, element, N) + constants.gas_constant*temperature*N*np.log(N/sumN)
        
'''

# Equations are as implemented in deKoker and Stixrude (supplementary materials)
# Boltzmann's constant is in J/K
def _partition_function(mass): # eq. S7; qi/V
    return np.power(mass*constants.Boltzmann/(2*np.pi*constants.Dirac*constants.Dirac), 1.5)

def _atomic_momenta(temperature, volume, params): # F_ig, eq. S6
    # ideal gas
    entropy_sum=0.
    for element, N in params['formula'].iteritems(): # N is a.p.f.u
        #print element, N
        q=volume/constants.Avogadro*_partition_function(atomic_masses[element]/constants.Avogadro) # masses are in kg/mol
        entropy_sum+=N*(1. + np.log(q/N)) # eq. 16.72 of Callen., 1985; p. 373
    return -constants.gas_constant*temperature*entropy_sum # note boltzmann is for individual particles


def _finite_strain(temperature, volume, params): # f(V), eq. S3a
    return 0.5*(np.power(params['V_0']/volume, 2./3.) - 1.0)        

def _theta(temperature, volume, params): # theta, eq. S3b
    return np.power(temperature/params['T_0'], params['m']) - 1.

def _zeta(temperature, volume, params): # eq. S5a, beta in deKoker thesis (3.34)
    return params['zeta_0']*(np.power(volume/params['V_0'], params['xi']))

def _Tel(temperature, volume, params): # eq. S5b
    return params['Tel_0']*(np.power(volume/params['V_0'], params['eta']))
                            
                            
def _electronic_excitation_energy(temperature, volume, params): # F_el
    temperature_el=_Tel(temperature, volume, params)
    return -_zeta(temperature, volume, params)*(0.5*(temperature*temperature + temperature_el*temperature_el) - temperature*temperature_el*np.log(temperature/temperature_el))

'''
def _bonding_energy(temperature, volume, params): # F_xs, eq. S2
    for i in range(params['O_f']):
        for j in range(params['O_theta']):
            params['a'][i][j]/(np.math.factorial(i)*np.math.factorial(j)) * np.power(_finite_strain(temperature, volume, params), i) * np.power(_theta(temperature, volume, params), j)
'''

# Weird element order correcting:
def vector_to_array(a, Of, Otheta):
    array=np.empty([Of, Otheta])
    for i in range(Of):
        for j in range(Otheta):
            n=int(i*(i+1.)/2. + j) # (i=0, 0; i=1, 1; i=2, 3)
            array[i][j]=a[n]
    return array


def _bonding_energy(temperature, volume, params): # F_xs, eq. S2
    f=_finite_strain(temperature, volume, params)
    theta=_theta(temperature, volume, params)
    energy = 0.
    for i in range(len(params['a'])):
        ifact=factorial(i, exact=False)
        for j in range(len(params['a'][0])):
            jfact=factorial(j, exact=False)
            energy += params['a'][i][j]*np.power(f, i)*np.power(theta, j)/ifact/jfact
            
    return energy

'''
def bonding_pressure(temperature, volume, params): # P_xs, eq. 3.17 of de Koker thesis
    return -dfdv(temperature, volume, params)
'''

def helmholtz_free_energy(temperature, volume, params): # F(V,T), eq. S1
    return _atomic_momenta(temperature, volume, params) + _electronic_excitation_energy(temperature, volume, params) + _bonding_energy(temperature, volume, params)

class SiO2_liquid():
    params={'formula': {'Si': 1., 'O': 2.},
                 'V_0': 0.2780000000E+02, # F_cl (1,1) # should be m^3/mol
                 'T_0': 0.3000000000E+04, # F_cl (1,2) # K
                 'E_0': -.2360007614E+04, # F_cl (1,3) # J/mol
                 'S_0': -.1380253514E+00, # F_cl (1,4) # J/K/mol
                 'm':0.91, # F_cl (5)
                 'a': vector_to_array([-.1945931560E+04, -.2266835978E+03, 0.4550286309E+03, 0.2015652870E+04, -.2005850460E+03, -.2166028187E+03, 0.4836972992E+05, 0.4415340414E+03, 0.7307765325E+02, 0.0000000000E+00, -.6515876520E+06, 0.2070169954E+05, 0.8921220900E+03, 0.0000000000E+00, 0.0000000000E+00, 0.4100181286E+07, -.1282587237E+06, -.1228478753E+04, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00], 2, 5),
                 'zeta_0':0.4266056389E-03, # F_el (1)
                 'xi':0.8639433047E+00, # F_el (2)
                 'Tel_0':0.5651204964E+04, # F_el (3)
                 'eta':-.2783503528E+00 
            } # F_el (4)

SiO2_liq=SiO2_liquid()

temperature=3000.
volume=0.2780000000E+02 # should be m^3/mol
print 'Helmholtz:', helmholtz_free_energy(temperature, volume, SiO2_liq.params) # should be J/mol
print 'F_xs:', _bonding_energy(temperature, volume, SiO2_liq.params) # should be J/mol
print 'F_el:', _electronic_excitation_energy(temperature, volume, SiO2_liq.params) # should be J/mol
print 'F_ig:', _atomic_momenta(temperature, volume, SiO2_liq.params) # should be J/mol


