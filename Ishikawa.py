import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

Avogadro = 6.0223e23
gas_constant = 8.31446
def pressure(volume, temperature, params):
    return pressure_reference_T(volume, params) + thermal_pressure(volume, temperature, params)

def pressure_reference_T(volume, params):
    x = np.power(volume/params['V_0'], 1./3.)
    return 3.*params['K_0']/(x*x)*(1-x)*np.exp(1.5*(params['Kprime_0'] - 1.)*(1. - x))

def thermal_pressure(volume, temperature, params):
    return grueneisen(volume, params)/volume*(thermal_energy(volume, temperature, params) - thermal_energy(volume, params['T_0'], params))

def thermal_energy(volume, temperature, params):
    return 3.*params['n']*gas_constant*(temperature + params['e_0']*np.power(volume/params['V_0'], params['g'])*temperature*temperature)

def grueneisen(volume, params):
    return params['grueneisen_0']*(1. + params['a']*(np.power(volume/params['V_0'], params['b']) - 1.))

def heat_capacity_v(volume, temperature, params):
    return 3.*params['n']*gas_constant*(1. + 2.*params['e_0']*np.power(volume/params['V_0'], params['g'])*temperature)

# S = \int Cv/T dT
def entropy(volume, temperature, params):
    return 3.*params['n']*gas_constant*( np.log(temperature) + 2.*params['e_0']*np.power(volume/params['V_0'], params['g'])*temperature)

# F = F_0 - intSdT | V - intPdV | T0
'''
def helmholtz_free_energy_old(volume, temperature, params):
    intSdT = 3.*params['n']*gas_constant*( temperature*(np.log(temperature) - 1.)
                                           - params['T_0']*(np.log(params['T_0']) - 1.)
                                           + params['e_0']*np.power(volume/params['V_0'], params['g'])
                                           * ( temperature*temperature - params['T_0']*params['T_0'] ))
    
    x = np.power(volume/params['V_0'], 1./3.)
    intPdV = (2.*params['K_0']*params['V_0']*(5.-3.*x+3.*params['Kprime_0']*(x - 1.))) / ((params['Kprime_0'] - 1.)*(params['Kprime_0'] - 1.)*np.exp((3.*(params['Kprime_0'] - 1.)*(x - 1.))/2.)) - (4.*params['K_0']*params['V_0'])/((params['Kprime_0'] - 1.)*(params['Kprime_0'] - 1.))
    return params['F_0'] - intSdT - intPdV
'''

def helmholtz_free_energy(volume, temperature, params):

    x = np.power(volume/params['V_0'], 1./3.)
    intPdV = (2.*params['K_0']*params['V_0']*(5.-3.*x+3.*params['Kprime_0']*(x - 1.))) \
             / ((params['Kprime_0'] - 1.)*(params['Kprime_0'] - 1.)*np.exp((3.*(params['Kprime_0'] - 1.)*(x - 1.))/2.)) \
             - (4.*params['K_0']*params['V_0'])/((params['Kprime_0'] - 1.)*(params['Kprime_0'] - 1.))
    
    thermal_part = 3.*params['grueneisen_0']*params['n']*gas_constant*((params['a']*(temperature - params['T_0'])*np.power(volume/params['V_0'], params['b']))/params['b']
                                                                       + ((params['a'] - 1.)*params['e_0']*(temperature*temperature - params['T_0']*params['T_0'])
                                                                          * np.power(volume/params['V_0'], params['g']))/params['g']
                                                                       + (params['a']*params['e_0']*(temperature*temperature - params['T_0']*params['T_0'])
                                                                          * np.power(volume/params['V_0'], params['b'] + params['g']))
                                                                       /(params['b'] + params['g']) + (params['a'] - 1.)*(temperature - params['T_0'])*np.log(volume))

    thermal_part0 = 3.*params['grueneisen_0']*params['n']*gas_constant*((params['a']*(temperature - params['T_0']))/params['b']
                                                                         + ((params['a'] - 1.)*params['e_0']*(temperature*temperature - params['T_0']*params['T_0']))/params['g']
                                                                         + (params['a']*params['e_0']*(temperature*temperature - params['T_0']*params['T_0']))
                                                                         /(params['b'] + params['g']) + (params['a'] - 1.)*(temperature - params['T_0'])*np.log(params['V_0']))

    return intPdV + thermal_part - thermal_part0

def gibbs_free_energy(volume, temperature, params):
    return helmholtz_free_energy(volume, temperature, params) \
        + pressure(volume, temperature, params) * volume


def volume(P, T, params):
    pdiff = lambda V: P - pressure(V, T, params)
    return opt.fsolve(pdiff, [0.5*params['V_0']], args=())[0]
               
params = {
    'T_0': 7000.,
    'F_0': 0.,
    'V_0': 17.87*1.e-30*Avogadro,
    'K_0': 24.6e9,
    'Kprime_0': 6.65,
    'grueneisen_0': 1.85,
    'a': 1.,
    'b': 0.35,
    'e_0': 0.314e-4,
    'g': -0.4,
    'n': 1.
    }

temperatures = np.linspace(1000., 7000., 21)
volumes = np.empty_like(temperatures)
entropies = np.empty_like(temperatures)
p = 1.e5
for i, T in enumerate(temperatures):
    volumes[i] = volume(p, T, params)
    entropies[i] = entropy(volumes[i], T, params)
    
#plt.plot(temperatures, entropies)
#plt.show()



dT = 1.
dP = 1000.

P = 0.e9
T = 1.

gibbs0 = gibbs_free_energy(volume(P, T, params), T, params)
gibbs1 = gibbs_free_energy(volume(P, T+dT, params), T+dT, params)

S = -(gibbs1 - gibbs0)/dT
print S, entropy(volume(P, T, params), T, params)




gibbs0 = gibbs_free_energy(volume(P, T, params), T, params)
gibbs1 = gibbs_free_energy(volume(P+dP, T, params), T, params)

V = (gibbs1 - gibbs0)/dP
print V, volume(P, T, params)



dV = 1.e-10
V = 0.8*params['V_0']
T = 1.


helmholtz0 = helmholtz_free_energy(V, T, params)
helmholtz1 = helmholtz_free_energy(V, T+dT, params)
S = -(helmholtz1 - helmholtz0)/dT
print 'Entropy:', S, entropy(V, T, params), S - entropy(V, T, params)


helmholtz0 = helmholtz_free_energy(V, T, params)
helmholtz1 = helmholtz_free_energy(V+dV, T, params)
P = (helmholtz1 - helmholtz0)/dV
print 'Pressure:', P/1.e9, pressure(V, T, params)/1.e9
exit()

pressures = np.linspace(1.e9, 500.e9, 101)
volumes = np.empty_like(pressures)
for T in [4000., 5000., 6000., 7000.]:
    print T
    for i, p in enumerate(pressures):
        volumes[i] = volume(p, T, params)
    plt.plot(pressures, volumes/(1.e-30*Avogadro))
plt.ylim(6., 11.)
plt.show()



temperatures = np.linspace(10., 8000., 101)
thermal_pressures = np.empty_like(temperatures)
for p in [136.e9, 200.e9, 265.e9, 330.e9]:
    Pth0 = thermal_pressure(volume(p, 0., params), 0., params)
    for i, T in enumerate(temperatures):
        thermal_pressures[i] =  thermal_pressure(volume(p, T, params), T, params) - Pth0

    plt.plot(temperatures, thermal_pressures/1.e9)
plt.ylim(0., 80.)
plt.show()


pressures = np.linspace(100.e9, 400.e9, 101)
heat_capacities = np.empty_like(pressures)
for T in [4000., 5000., 6000., 7000.]:
    print T
    for i, p in enumerate(pressures):
        volumes[i] = volume(p, T, params)
    plt.plot(pressures, volumes/(1.e-30*Avogadro))
plt.ylim(30., 55.)
plt.show()




