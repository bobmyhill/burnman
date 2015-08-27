# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

"""

example_thermodynamic_beginner
----------------------
    
This example is for absolute beginners wishing to use burnman to do 
thermodynamics. 

*Uses:*

* :doc:`mineral_database`


*Demonstrates:*

* Ways to query endmember properties
* How to set a reference temperature

"""

import os, sys, numpy as np, matplotlib.pyplot as plt
from scipy import optimize
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
import matplotlib.pyplot as plt

Cp_LT = 1000.
Cp_HT = 1700.

T_ref = 1809.
P_ref = 49.9999999e9

T_obs = 2809.
P_obs = 50.e9


mineral=minerals.Myhill_calibration_iron.liquid_iron()

mineral.set_state(50.e9, 7000.)
print mineral.gibbs

# First, let's fit the heat capacity at the new reference pressure
# over the calibration range 
def Cp(T, a, b, c, d):
    return a + b*T + c/T/T + d/np.sqrt(T)

mineral.set_state(P_ref, T_ref)

Cps = []
temperatures = np.linspace(Cp_LT, Cp_HT, 100)
for T in temperatures:
    mineral.set_state(P_ref, T)
    Cps.append(mineral.C_p)
    
Cp_new = optimize.curve_fit(Cp, temperatures, Cps)[0]
a, b, c, d = Cp_new

# Let's just check that the conversion is ok...
plt.plot(temperatures, Cps)
plt.plot(temperatures, Cp(temperatures, a, b, c, d))
plt.ylabel('Cp')
plt.show()

# Now let's record several properties at the PT of comparison
mineral.set_state(P_obs, T_obs)
print mineral.gibbs, mineral.H, mineral.S, mineral.C_p

temperatures = np.linspace(300., 5000., 100)
Cp0 = []
alphas_0 = []
volumes_0 = []
gibbs_0 = []
for T in temperatures:
    mineral.set_state(P_obs, T)
    Cp0.append(mineral.C_p)
    alphas_0.append(mineral.alpha)
    volumes_0.append(mineral.V)
    gibbs_0.append(mineral.gibbs)


pressures = np.linspace(1.e5, 100.e9, 100)
eos_0 = []
for P in pressures:
    mineral.set_state(P, T_obs)
    eos_0.append(mineral.V)

# Now comes the work of finding all the values at the reference pressure and temperature
dP = 100000.
mineral.set_state(P_ref-dP, T_ref)

K0 = mineral.K_T 
mineral.set_state(P_ref, T_ref)
K1 = mineral.K_T
mineral.set_state(P_ref+dP, T_ref)
K2 = mineral.K_T

grad0 = (K1 - K0)/dP
grad1 = (K2 - K1)/dP

mineral.set_state(P_ref, T_ref)
mineral.params['T_0'] = T_ref
mineral.params['P_0'] = P_ref
mineral.params['H_0'] = mineral.H
mineral.params['S_0'] = mineral.S
mineral.params['V_0'] = mineral.V
mineral.params['Cp'] = [a, b, c, d]
mineral.params['K_0'] = mineral.K_T 
mineral.params['a_0'] = mineral.alpha
mineral.params['Kprime_0'] = (K2 - K0)/(2.*dP)
mineral.params['Kdprime_0'] = (grad1 - grad0)/dP


# Now print the new values
mineral.set_state(P_obs, T_obs)
print mineral.gibbs, mineral.H, mineral.S, mineral.C_p

temperatures = np.linspace(300., 5000., 100)
Cp1 = []
alphas_1 = []
volumes_1 = []
gibbs_1 = []
for T in temperatures:
    mineral.set_state(P_obs, T)
    Cp1.append(mineral.C_p)
    alphas_1.append(mineral.alpha)
    volumes_1.append(mineral.V)
    gibbs_1.append(mineral.gibbs)
    
pressures = np.linspace(1.e5, 100.e9, 100)
eos_1 = []
for P in pressures:
    mineral.set_state(P, T_obs)
    eos_1.append(mineral.V)


# Finally, let's do some plotting

plt.plot(temperatures, np.array(Cp0), label='old')
plt.plot(temperatures, np.array(Cp1), label='new')
plt.legend(loc='lower right')
plt.show()

plt.plot(temperatures, np.array(Cp0), label='old')
plt.plot(temperatures, np.array(Cp1), label='new')
plt.xlim(Cp_LT, Cp_HT)
plt.ylim(0., 100.)
plt.legend(loc='lower right')
plt.show()


plt.plot(temperatures, np.array(alphas_0), label='Tref = room temperature')
plt.plot(temperatures, np.array(alphas_1), label='Tref = '+str(T_ref)+' K')

plt.title('Thermal expansion at '+str(P_obs/1.e9)+' GPa')
plt.legend(loc='lower right')
plt.xlabel('Temperatures (K)')
plt.ylabel('Thermal expansivity (m^3/m^3/K)')
plt.show()        

plt.plot(temperatures, np.array(volumes_0), label='Tref = room temperature')
plt.plot(temperatures, np.array(volumes_1), label='Tref = '+str(T_ref)+' K')
plt.title('Thermal expansion')
plt.legend(loc='lower right')
plt.xlabel('Temperatures (K)')
plt.ylabel('Volumes (m^3/mol)')
plt.show()

plt.plot(pressures/1.e9, np.array(eos_0), label='Tref = room temperature')
plt.plot(pressures/1.e9, np.array(eos_1), label='Tref = '+str(T_ref)+' K')

plt.title('Compression at '+str(T_obs)+' K')
plt.legend(loc='lower right')
plt.xlabel('Pressures (GPa)')
plt.ylabel('Volumes (m^3/mol)')
plt.show()


plt.plot(temperatures, np.array(gibbs_0) - np.array(gibbs_1))

plt.title('Compression at '+str(T_obs)+' K')
plt.xlabel('Temperatures (K)')
plt.ylabel('Delta Gibbs (J/mol)')
plt.ylim(-100., 100.)
plt.show()



mineral.set_state(50.e9, 7000.)
print mineral.gibbs


from math import log10, floor, isnan
round_to_n = lambda x, n: round(x, -int(floor(log10(abs(x)))) + (n - 1))

for p in mineral.params.keys():
    if isinstance(mineral.params[p], float) and \
            p != 'H_0':
        if isnan(mineral.params[p]):
            del mineral.params[p] 
        else:
            mineral.params[p] = round_to_n(mineral.params[p], 5)
    if p == 'Cp':
        for i, value in enumerate(mineral.params[p]):
            mineral.params[p][i] = round_to_n(value, 5)
    if p == 'H_0':
        mineral.params[p] = round(mineral.params[p], 0)

print mineral.params

mineral.set_state(50.e9, 7001.)
mineral.set_state(50.e9, 7000.)
print mineral.gibbs
