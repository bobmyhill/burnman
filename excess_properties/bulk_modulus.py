# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.

from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import minerals
'''
hlt = minerals.HP_2011_ds62.hlt()

T = 300.
pressures = np.linspace(1.5, 200.e9, 101)
volumes = np.empty_like(pressures)
for i, P in enumerate(pressures):
    hlt.set_state(P, T)
    volumes[i] = hlt.V

f = 1.01
V_0 = hlt.params['V_0']
K_0 = hlt.params['K_0']
hlt.params['V_0'] = V_0*f
hlt.params['K_0'] = np.power((1./f), 7./3.)*K_0 
#hlt.params['Kprime_0'] = 4.975

print(K_0/1.e9, hlt.params['K_0']/1.e9)
volumes2 = np.empty_like(pressures)
for i, P in enumerate(pressures):
    hlt.set_state(P, T)
    volumes2[i] = hlt.V


plt.plot(pressures, volumes2 - volumes)
plt.show()



# K' = 4.
mineral_list = [minerals.HP_2011_ds62.ep, minerals.HP_2011_ds62.syv]

for mineral in mineral_list:
    
    orig = mineral()
    orig2 = mineral()
    
    f = 1.05
    V_0 = orig.params['V_0']
    K_0 = orig2.params['K_0']
    orig2.params['V_0'] = f*V_0
    orig2.params['K_0'] = np.power(f, -6.)*K_0
    
    pressures = np.linspace(1.e5, 500.e9, 101)
    vdiff = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        orig.set_state(P, T)
        orig2.set_state(P, T)
        vdiff[i] = orig2.V - orig.V
    
    plt.plot(pressures, vdiff)
    plt.show()
    

# Check equation for ideal bulk modulus

fo = minerals.HP_2011_ds62.fo()
fa = minerals.HP_2011_ds62.fa()

fofa = minerals.HP_2011_ds62.fo()

Vfofa = 0.5*(fo.params['V_0'] + fa.params['V_0'])

dVdP = -0.5*(fo.params['V_0']/fo.params['K_0'] + fa.params['V_0']/fa.params['K_0'])

Kfofa = Vfofa/(-dVdP)

fofa.params['V_0'] = Vfofa
fofa.params['K_0'] = Kfofa
fofa.params['Kprime_0'] = fofa.params['Kprime_0'] + 0.44

pressures = np.linspace(1.e5, 5.e9, 101)
for i, P in enumerate(pressures):
    fo.set_state(P, T)
    fa.set_state(P, T)
    fofa.set_state(P, T)

    vdiff[i] = fofa.V - 0.5*(fo.V + fa.V)

plt.plot(pressures, vdiff)
plt.show()
'''

# Test of MT-like EoS for excess properties at standard temperature

def volume_excess(Vexcess, Videal, Kideal, alphaideal, Kprime, theta, pressure, temperature):
    Vnonideal = Videal + Vexcess
    Knonideal = Kideal*np.power(Videal/Vnonideal, Kprime)
    T_0 = 300.
    u_0 = theta/T_0
    u = theta/temperature
    ksi_0 = u_0*u_0*np.exp(u_0)/((np.exp(u_0) - 1.)*(np.exp(u_0) - 1.))
    Pth = alphaideal*Kideal*theta/ksi_0*((1./(np.exp(u) - 1.)) - (1./(np.exp(u_0) - 1.)))
    return Vnonideal*np.power((1.+Kprime*(pressure-Pth)/Knonideal), -1./Kprime) \
      - Videal*np.power((1.+Kprime*(pressure-Pth)/Kideal), -1./Kprime)

Vexcess = 1.e-6
Videal = 100.e-6
Kideal = 150.e9
theta = 10636./(200./20. + 6.44)
alphaideal = 1.e-5

T = 300.
pressures = np.linspace(0., 20.e9, 101)
Vxs = np.empty_like(pressures)
Vxs2 = np.empty_like(pressures)
print theta
for i, P in enumerate(pressures):
    Vxs[i] = volume_excess(Vexcess, Videal, Kideal, alphaideal, 4., theta, P, T)
    Vxs2[i] = volume_excess(Vexcess, Videal, Kideal, alphaideal, 10., theta, P, T)


plt.plot(pressures, Vxs, label='4')
plt.plot(pressures, Vxs2, label='10')
plt.legend(loc='upper right')
plt.show()

P = 1.e5 
temperatures = np.linspace(0.1, 6000., 101)
Vxs = np.empty_like(temperatures)
Vxs2 = np.empty_like(temperatures)
print theta
for i, T in enumerate(temperatures):
    Vxs[i] = volume_excess(Vexcess, Videal, Kideal, alphaideal, 10., theta, 1.e5, T)
    Vxs2[i] = volume_excess(Vexcess, Videal, Kideal, alphaideal, 10., theta, 10.e9, T)


plt.plot(temperatures, Vxs, label='1 bar')
plt.plot(temperatures, Vxs2, label='10 GPa')
plt.legend(loc='upper right')
plt.show()
