from liquid_EoS import _finite_strain

import numpy as np
from scipy.optimize import fsolve

def helmholtz_solid(volume, temperature, params):
    return params['E_0'] - params['T_0']*params['S_0'] \
        + F_cmp(volume, temperature, params) \
        + F_th(volume, temperature, params)

def F_th(volume, temperature, params):
    return -params['S_0']*(temperature - params['T_0'])  \
        - params['C_V']*(temperature*np.log(temperature/params['T_0']) \
                         - (temperature - params['T_0'])) \
        - int_aKt_dV(volume, temperature, params) * (temperature - params['T_0'])

#Evaluate the integral of aK_T(V,To) from V1 to V2, assuming gamma = gamma_0*(V/Vo)^q
def int_aKt_dV(volume, temperature, params):
    return params['C_V']*params['gamma_0']/params['q_0']*(np.power(volume/params['V_0'], params['q_0']) - 1.)

def F_cmp(volume, temperature, params):
    f = _finite_strain(volume, temperature, params)
    K_0 = params['K_0']
    K_prime_0 = params['K_prime_0']
    K_dprime_0 = params['K_dprime_0']
    a3 = 3.*(K_prime_0 - 4.)
    a4 = 9.* ( K_0 * K_dprime_0 + K_prime_0 * (K_prime_0 - 7.) ) + 143.
    return 9.*K_0*params['V_0']*(f*f/2. + a3*f*f*f/6. + a4*f*f*f*f/24.)


def pressure_solid(volume, temperature, params):
    f = _finite_strain(volume, temperature, params)
    n = params['n']
    K_0 = params['K_0']
    K_prime_0 = params['K_prime_0']
    K_dprime_0 = params['K_dprime_0']
    a3 = 3. * ( K_prime_0 - 4. )
    a4 = 9. * ( K_0 * K_dprime_0 + K_prime_0 * (K_prime_0 - 7.) ) + 143.
    return 3.*params['K_0']*np.power(1+2.*f, 2.5)*(f + a3*f*f/2. + a4/6.*f*f*f) \
        + params['C_V'] * (temperature - params['T_0']) \
        * gamma(volume, temperature, params) / volume

def _volume_at_pressure(volume, pressure, temperature, params):
    return pressure - pressure_solid(volume[0], temperature, params)
    
def volume_solid(pressure, temperature, params):
    return fsolve(_volume_at_pressure, 0.8e-6, args=(pressure, temperature, params), xtol=1e-12, full_output=True)[0][0]

def gamma(volume, temperature, params):
    return params['gamma_0']*np.power(volume/params['V_0'], params['q_0'])
    
# E = F + TS = F - T(dF/dT)|v
# There is a typo in both deKoker and Stixrude papers (2009, 2013) where a factor of (T-T0)
# is multiplied with the aK integral. This should read T0.
def energy_solid(volume, temperature, params):
    return params['E_0'] + F_cmp(volume, temperature, params) \
        + params['C_V']*(temperature - params['T_0']) \
        + int_aKt_dV(volume, temperature, params) * (params['T_0'])

def entropy_solid(volume, temperature, params):
    return params['S_0'] + int_aKt_dV(volume, temperature, params) + params['C_V']*np.log(temperature/params['T_0'])

def gibbs_solid(pressure, temperature, params):
    volume=fsolve(_volume_at_pressure, params['V_0']/2., args=(pressure, temperature, params))[0]
    return helmholtz_solid(volume, temperature, params) + pressure*volume
    
'''
# An alternative way of calculating the internal energy
def energy_2(volume, temperature, params):
    return helmholtz_solid(volume, temperature, params) + temperature*entropy_solid(volume, temperature, params)
'''