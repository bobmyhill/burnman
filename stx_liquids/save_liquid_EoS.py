import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.processchemistry import *
from burnman import constants
import numpy as np


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

def energy_excess_coefficients(f, theta):
    bf=np.empty([36])
    bf[0]  = 1.     # 0 0
    bf[1]  = f    # 1 0  volume functional [finite strain)
    bf[2]  = theta   # 0 1  temperature functional

    bf[3]  = (1./2.)*(np.power(f,2.))   # 2 0
    bf[4]  = (f)*(theta)       # 1 1
    bf[5]  = (1./2.)*(np.power(theta,2.))  # 0 2
    
    bf[6]  = (1./6.)*(np.power(f,3.))           # 3 0
    bf[7]  = (1./2.)*(np.power(f,2.))*(theta)   # 2 1
    bf[8]  = (1./2.)*(f)*(np.power(theta,2.))   # 1 2
    bf[9] = (1./6.)*(np.power(theta,3.))          # 0 3
    
    bf[10] = (1./24.)*(np.power(f,4.))               # 4 0
    bf[11] = (1./ 6.)*(np.power(f,3.))*(theta)       # 3 1
    bf[12] = (1./ 4.)*(np.power(f,2.))*(np.power(theta,2.))   # 2 2
    bf[13] = (1./ 6.)*(f)*(np.power(theta,3.))       # 1 3
    bf[14] = (1./24.)*(np.power(theta,4.))              # 0 4
    
    bf[15] = (1./120.)*(np.power(f,5.))               # 5 0
    bf[16] = (1./ 24.)*(np.power(f,4.))*(theta)       # 4 1
    bf[17] = (1./ 12.)*(np.power(f,3.))*(np.power(theta,2.))   # 3 2
    bf[18] = (1./ 12.)*(np.power(f,2.))*(np.power(theta,3.))   # 2 3
    bf[19] = (1./ 24.)*(f)*(np.power(theta,4.))       # 1 4
    bf[20] = (1./120.)*(np.power(theta,5.))              # 0 5
    
    bf[21] = (1./720.)*(np.power(f,6.))               # 6 0
    bf[22] = (1./120.)*(np.power(f,5.))*(theta)       # 5 1
    bf[23] = (1./ 48.)*(np.power(f,4.))*(np.power(theta,2.))   # 4 2
    bf[24] = (1./ 36.)*(np.power(f,3.))*(np.power(theta,3.))   # 3 3
    bf[25] = (1./ 48.)*(np.power(f,2.))*(np.power(theta,4.))   # 2 4
    bf[26] = (1./120.)*(f)*(np.power(theta,5.))       # 1 5
    bf[27] = (1./720.)*(np.power(theta,6.))              # 0 6
    
    bf[28] = (1./5040.)*(np.power(f,7.))               # 7 0
    bf[29] = (1./ 720.)*(np.power(f,6.))*(theta)       # 6 1
    bf[30] = (1./ 240.)*(np.power(f,5.))*(np.power(theta,2.))   # 5 2
    bf[31] = (1./ 144.)*(np.power(f,4.))*(np.power(theta,3.))   # 4 3
    bf[32] = (1./ 144.)*(np.power(f,3.))*(np.power(theta,4.))   # 3 4
    bf[33] = (1./ 240.)*(np.power(f,2.))*(np.power(theta,5.))   # 2 5
    bf[34] = (1./ 720.)*(f)*(np.power(theta,6.))       # 1 6
    bf[35] = (1./5040.)*(np.power(theta,7.))              # 0 7
    
    return bf


def _bonding_energy(temperature, volume, params): # F_xs, eq. S2
    f=_finite_strain(temperature, volume, params)
    theta=_theta(temperature, volume, params)
    bf=energy_excess_coefficients(f, theta)

    ffun_fcl340 = 0.
    for i in range(len(params['a'])):
        ffun_fcl340 += params['a'][i]*bf[i]
        
    return ffun_fcl340

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
                 'O_f':2, # F_cl (3) # expansion order
                 'm':0.91, # F_cl (5)
                 'O_theta':5, # expansion order
                 'a': [-.1945931560E+04, -.2266835978E+03, 0.4550286309E+03, 0.2015652870E+04, -.2005850460E+03, -.2166028187E+03, 0.4836972992E+05, 0.4415340414E+03, 0.7307765325E+02, 0.0000000000E+00, -.6515876520E+06, 0.2070169954E+05, 0.8921220900E+03, 0.0000000000E+00, 0.0000000000E+00, 0.4100181286E+07, -.1282587237E+06, -.1228478753E+04, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00], # element order for a is a little weird; see function "energy_excess_coefficients", above
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


