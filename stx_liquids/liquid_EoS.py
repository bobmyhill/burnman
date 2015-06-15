import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.processchemistry import *
from burnman import constants
import numpy as np
from scipy.misc import factorial
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
        q=volume/constants.Avogadro*_partition_function(atomic_masses[element]/constants.Avogadro, temperature) # masses are in kg/mol
        entropy_sum+=N*(1. + np.log(q/N)) # eq. 16.72 of Callen., 1985; p. 373
    return -constants.gas_constant*temperature*entropy_sum # note boltzmann is for individual particles

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
def _pressure(volume, temperature, params):
    return _atomic_pressure(volume, temperature, params) + _electronic_excitation_pressure(volume, temperature, params) + _bonding_pressure(volume, temperature, params)

def _pressure_residual(volume, pressure, temperature, params):
    return pressure - _pressure(volume[0], temperature, params)

def _volume(pressure, temperature, params):
    return fsolve(_pressure_residual, 0.8e-6, args=(pressure, temperature, params), xtol=1e-12, full_output=True)[0][0]

def helmholtz_free_energy(volume, temperature, params): # F(V,T), eq. S1
    return _atomic_momenta(volume, temperature, params) + _electronic_excitation_energy(volume, temperature, params) + _bonding_energy(volume, temperature, params)

class SiO2_liquid():
    params={'formula': {'Si': 1., 'O': 2.},
                 'V_0': 0.2780000000E+02*1e-6, # F_cl (1,1) # should be m^3/mol
                 'T_0': 0.3000000000E+04, # F_cl (1,2) # K
                 'E_0': -.2360007614E+04 * 1e3, # F_cl (1,3) # J/mol
                 'S_0': -.1380253514E+00, # F_cl (1,4) # J/K/mol
                 'O_theta': 2,
                 'O_f':5,
                 'm': 0.91, # F_cl (5)
                 'a': [-.1945931560E+04, -.2266835978E+03, 0.4550286309E+03, 0.2015652870E+04, -.2005850460E+03, -.2166028187E+03, 0.4836972992E+05, 0.4415340414E+03, 0.7307765325E+02, 0.0000000000E+00, -.6515876520E+06, 0.2070169954E+05, 0.8921220900E+03, 0.0000000000E+00, 0.0000000000E+00, 0.4100181286E+07, -.1282587237E+06, -.1228478753E+04, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00],
                 'zeta_0':0.4266056389E-03, # [J K^-2] F_el (1)
                 'xi':0.8639433047E+00, # [], F_el (2)
                 'Tel_0':0.5651204964E+04, # [K], F_el (3)
                 'eta':-.2783503528E+00, # []
                 'el_V_0':0.1000000000E+01 * 1e-6 # [m^3/mol]
            } # F_el (4)
    params['a'] = vector_to_array(params['a'], params['O_f'], params['O_theta'])*1e3 # [J/mol]

SiO2_liq=SiO2_liquid()

# see page 15 of supplementary material for SiO2 volume-pressure and volume-energy benchmark
# For a volume of 11 cm^3/mol and temperature of 7000 K, pressure should be ca. 170 GPa  
# Energy (E, not F or G) should be ca. -1560 kJ/mol
temperature=7000. # K
volume=11e-6 # m^3/mol
print _pressure(volume, temperature, SiO2_liq.params)/1e9

temperature=3000. # K
volume=0.2780000000E+02*1e-6 # m^3/mol
print _pressure(volume, temperature, SiO2_liq.params)/1e9


fig1 = mpimg.imread('figures/SiO2_liquid_PVT.png')
plt.imshow(fig1, extent=[9, 30, -10, 220], aspect='auto')

temperatures=np.linspace(2000., 7000., 6)
#temperatures=np.linspace(3000., 7000., 2)
volumes=np.linspace(9e-6, 30e-6, 101)
pressures=np.empty_like(volumes)
for temperature in temperatures:
    for i, volume in enumerate(volumes):
        pressures[i]=_pressure(volume, temperature, SiO2_liq.params)/1e9
    plt.plot(volumes*1e6, pressures, linewidth=2, label=str(temperature)+'K')

plt.legend(loc='upper right')
plt.xlim(9,30)
plt.ylim(-10,220)
plt.show()

fig1 = mpimg.imread('figures/SiO2_liquid_SelVT.png')
plt.imshow(fig1, extent=[9, 30, -0.03, 0.75], aspect='auto')

temperatures=np.linspace(2000., 7000., 6)
volumes=np.linspace(9e-6, 30e-6, 101)
entropies=np.empty_like(volumes)
for temperature in temperatures:
    for i, volume in enumerate(volumes):
        entropies[i]=_electronic_excitation_entropy(volume, temperature, SiO2_liq.params)
    plt.plot(volumes*1e6, entropies/3./constants.gas_constant, linewidth=2, label=str(temperature)+'K')

plt.legend(loc='upper right')
plt.xlim(9,30)
plt.show()

fig1 = mpimg.imread('figures/SiO2_liquid_EVT.png')
plt.imshow(fig1, extent=[9, 30, -2400, -1200], aspect='auto')

temperatures=np.linspace(2000., 7000., 6)
volumes=np.linspace(9e-6, 30e-6, 101)
energies=np.empty_like(volumes)
for temperature in temperatures:
    for i, volume in enumerate(volumes):
        dT=1e-4
        F = helmholtz_free_energy(volume, temperature, SiO2_liq.params)
        S = (F - helmholtz_free_energy(volume, temperature+dT, SiO2_liq.params))/dT
        energies[i]=F + temperature*S
    plt.plot(volumes*1e6, energies/1e3, linewidth=2, label=str(temperature)+'K')

plt.legend(loc='upper right')
plt.xlim(9,30)
plt.show()






pressure=14.e9 # Pa
temperature=3120. # K
stv=burnman.minerals.SLB_2011.stishovite()
stv.set_state(pressure, temperature)
volume=_volume(pressure, temperature, SiO2_liq.params)
print 'volume:', volume, 'should be ca. 18.5 cm^3/mol'
print 'PV:', pressure*volume/1e3, 'kJ/mol'
print temperature*SiO2_liq.params['S_0'], temperature*SiO2_liq.params['S_0']*1000., SiO2_liq.params['E_0']
print -temperature*SiO2_liq.params['S_0']*1000. + SiO2_liq.params['E_0']
print 'Gibbs (stv):', stv.gibbs/1e3, 'kJ/mol'
print 'Gibbs (SiO2_liq):', (helmholtz_free_energy(volume, temperature, SiO2_liq.params) + pressure*volume)/1e3, 'kJ/mol'

print 'Helmholtz (liq):', helmholtz_free_energy(volume, temperature, SiO2_liq.params)/1.e3
print 'Helmholtz (stv):', (stv.gibbs - pressure*stv.V)/1.e3

print 'Energy, V (stv):', (stv.gibbs - pressure*stv.V + temperature*stv.S)/1.e3, stv.V

'''
volume=_volume(pressure, temperature, SiO2_liq.params) # should be m^3/mol
print 'Helmholtz:', helmholtz_free_energy(volume, temperature, SiO2_liq.params), 'J/mol'

print 'F_xs:', _bonding_energy(volume, temperature, SiO2_liq.params)/1e3 # should be J/mol
print 'F_el:', _electronic_excitation_energy(volume, temperature, SiO2_liq.params)/1e3 # should be J/mol
print 'F_ig:', _atomic_momenta(volume, temperature, SiO2_liq.params)/1e3 # should be J/mol
print 'PV:', pressure*volume/1e3
print 'stv volume:', stv.V

pressure = _pressure(volume, temperature, SiO2_liq.params)
print 'Volume:', volume, 'Pressure:', pressure, 'Pa'
volume=_volume(pressure, temperature, SiO2_liq.params)
pressure = _pressure(volume, temperature, SiO2_liq.params)
print 'Volume:', volume, 'Pressure:', pressure, 'Pa'
'''

'''
dT=1e-4
pressure=0.e9 # Pa
temperature=3000. # K
volume=_volume(pressure, temperature, SiO2_liq.params)
print 'Entropy (SiO2_liq):', -(helmholtz_free_energy(volume, temperature+dT, SiO2_liq.params) - (helmholtz_free_energy(volume, temperature, SiO2_liq.params)))/dT, 'should be ca. 8.92*N*R = 222.5'

print helmholtz_free_energy(volume, temperature, SiO2_liq.params), helmholtz_free_energy(volume, temperature+dT, SiO2_liq.params)
'''
