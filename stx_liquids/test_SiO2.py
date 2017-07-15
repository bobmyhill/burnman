import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import DKS_2013_solids, DKS_2013_liquids, DKS_2013_liquids_tweaked, SLB_2011
from burnman import constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
SOLIDS
'''

stv = DKS_2013_solids.stishovite()
stv_2 = SLB_2011.stishovite()
coe = SLB_2011.coesite()
liquid = DKS_2013_liquids.SiO2_liquid()
liquid_2 = DKS_2013_liquids_tweaked.SiO2_liquid()


P = 1.e5
temperatures = np.linspace(1., 2000., 101)
Cp = np.empty_like(temperatures)
Cp_2 = np.empty_like(temperatures)
S = np.empty_like(temperatures)
S_2 = np.empty_like(temperatures)


Ts, Cps, Ss, Hs, phis = np.loadtxt('data/stv_LT_Cp.dat', unpack=True)
plt.subplot(121)
plt.plot(Ts, Cps, marker='o', label='Yong et al., 2012')
plt.subplot(122)
plt.plot(Ts, Ss, marker='o', label='Yong et al., 2012')
for i, T in enumerate(temperatures):
    stv.set_state(P, T-0.5)
    S0 = stv.S
    stv.set_state(P, T+0.5)
    S1 = stv.S
    Cp[i] = T*(S1 - S0)

    
    stv.set_state(P, T)
    stv_2.set_state(P, T)
    Cp_2[i] = stv_2.heat_capacity_p

    S[i] = stv.S
    S_2[i] = stv_2.S


plt.subplot(121)
plt.plot(temperatures, Cp, label='DKS')
plt.plot(temperatures, Cp_2, label='SLB')

plt.subplot(122)
plt.plot(temperatures, S, label='DKS')
plt.plot(temperatures, S_2, label='SLB')

plt.ylim(0., 150)
plt.legend(loc='lower right')
plt.show()

T = 2973.15
P = 13.e9

stv.set_state(P, T)
stv_2.set_state(P, T)
liquid.set_state(P, T)
liquid_2.set_state(P, T)

#print liquid.S - liquid_2.S
#print stv.S - stv_2.S



print liquid_2.gibbs - stv_2.gibbs

pressures = np.linspace(1.e5, 50.e9, 51)
T1 = np.empty_like(pressures)
T2 = np.empty_like(pressures)
for i, P in enumerate(pressures):
    try:
        T1[i] = burnman.tools.equilibrium_temperature([stv_2, liquid_2], [1.0, -1.0], P, 2000.)
    except:
        T1[i] = 0.

    try:
        T2[i] = burnman.tools.equilibrium_temperature([coe, liquid_2], [1.0, -1.0], P, 2000.)
    except:
        T2[i] = 0.
        
plt.plot(pressures/1.e9, T1)
plt.plot(pressures/1.e9, T2)
plt.ylim(0., 6000.)
plt.show()
