#!/usr/python
import os, sys
import numpy as np
sys.path.insert(1,os.path.abspath('..'))

from scipy.optimize import fsolve, minimize, fmin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Liquid model
from models import *
from SS1985_functions import *

# Benchmarks for the solid solution class
import burnman
from burnman.minerals import DKS_2013_liquids_tweaked
from burnman import tools
from burnman.processchemistry import *
from burnman.chemicalpotentials import *



# 13 GPa, stv
n_cations=1. # number of cations
r=2. # Oxygens available for bonding
K0 = lambda T: 0.00000000001
K1 = lambda T: 1.
Kinf = lambda T: 100000000000.

G = lambda T: 0. - 120.*(T-1300.)
K = lambda T: np.exp(-(G(T))/(R*T))
Wsh = lambda T: 00000.
Whs = lambda T: 00000.


pressure = 13.e9
anhydrous_phase=DKS_2013_liquids_tweaked.stishovite()
liquid=MgO_SiO2_liquid()
liquid.set_composition([0., 1.])

Tmelt = fsolve(delta_gibbs, 2000., args=(pressure, anhydrous_phase, liquid, 1., 1.))[0]
print Tmelt

compositions=np.linspace(0.0001, 0.99, 101)
Gex=np.empty_like(compositions)
Gex_2=np.empty_like(compositions)
for i, X in enumerate(compositions):
    T0 = 1000.
    Gex[i]=(1-X)*excesses_nonideal(X, T0, r, K(T0), Wsh(T0), Whs(T0))[0] + X*excesses_nonideal(X, T0, r, K(T0), Wsh(T0), Whs(T0))[1]
    Gex_2[i]=(1-X)*excesses_nonideal(X, Tmelt, r, K(Tmelt), Wsh(Tmelt), Whs(Tmelt))[0] + X*excesses_nonideal(X, Tmelt, r, K(Tmelt), Wsh(Tmelt), Whs(Tmelt))[1]

plt.plot( compositions, Gex, '-', linewidth=2., label='model at '+str(T0)+' K')
plt.plot( compositions, Gex_2, '-', linewidth=2., label='model at Tmelt')
plt.ylabel("Excess Gibbs (J/mol)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()



fn0=lambda T: 0.
temperatures=np.linspace(600., Tmelt, 101)
compositions0=np.empty_like(temperatures)
compositions1=np.empty_like(temperatures)
compositionsinf=np.empty_like(temperatures)
compositions=np.empty_like(temperatures)

for i, T in enumerate(temperatures):
    compositions0[i]=fsolve(solve_composition, 0.001, args=(T, pressure, r, K0, fn0, fn0, anhydrous_phase, liquid, 1., 1.))
    compositions1[i]=fsolve(solve_composition, 0.001, args=(T, pressure, r, K1, fn0, fn0, anhydrous_phase, liquid, 1., 1.))
    compositionsinf[i]=fsolve(solve_composition, 0.001, args=(T, pressure, r, Kinf, fn0, fn0, anhydrous_phase, liquid, 1., 1.))
    compositions[i]=fsolve(solve_composition, 0.001, args=(T, pressure, r, K, Wsh, Whs, anhydrous_phase, liquid, 1., 1.))


plt.plot( compositions, temperatures, linewidth=1, label='stv')
plt.plot( compositions0, temperatures, linewidth=1, label='K=0')
plt.plot( compositions1, temperatures, linewidth=1, label='K=1')
plt.plot( compositionsinf, temperatures, linewidth=1, label='K=inf')



###################
# CALCULATE LIQUIDUS SPLINE
from scipy.interpolate import UnivariateSpline
Xs=[0.0, 0.21, 0.35, 0.48]
Ts=[Tmelt-273.15, 2200., 1890., 1660.] # in C (Zhang et al., 1996 for dry melting)
spline_Zhang = UnivariateSpline(Xs, Ts, s=1)

Xs_liquidus = np.linspace(0.0, 0.52, 101)
plt.plot(Xs_liquidus, spline_Zhang(Xs_liquidus)+273.15)
###################


stishovite = []
s_stishovite = []
superliquidus=[]
for line in open('data/13GPa_SiO2-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[2] == 's':
            stishovite.append([float(content[0])+273.15, 1.-float(content[1])])
        if content[2] == 'ss':
            s_stishovite.append([float(content[0])+273.15, 1.-float(content[1])])
        if content[2] == 'l':
            superliquidus.append([float(content[0])+273.15, 1.-float(content[1])])

stishovite=np.array(zip(*stishovite))
s_stishovite=np.array(zip(*s_stishovite))
superliquidus=np.array(zip(*superliquidus))
add_T = 30. # 
plt.plot( stishovite[1], stishovite[0] + add_T, marker='.', linestyle='none', label='stv+liquid')
plt.plot( s_stishovite[1], s_stishovite[0] + add_T, marker='.', linestyle='none', label='(stv)+liquid')
plt.plot( superliquidus[1], superliquidus[0] + add_T, marker='.', linestyle='none', label='superliquidus')

plt.ylim(1000.,3000.)
plt.xlim(0.,1.)
plt.ylabel("Temperature (K)")
plt.xlabel("X")
plt.legend(loc='upper right')
plt.show()

####################
# a-X relationships
compositions = np.linspace(0., 0.6, 101)
activities = np.empty_like(temperatures)
for i, composition in enumerate(compositions):
    temperature = spline_Zhang(composition)+273.15
    activities[i] =  np.exp( delta_gibbs([temperature], pressure, anhydrous_phase, liquid, 1., 1.) / (constants.gas_constant*temperature))

   
plt.plot(compositions, activities)
plt.title('Stishovite')
plt.xlim(0., 1.)
plt.ylim(0., 1.)
plt.show()
####################


data=[[compositions, temperatures],[compositions0, temperatures],[compositions1, temperatures],[compositionsinf, temperatures]]

for datapair in data:
    print '>> -W1,black'
    compositions, temperatures=datapair
    for i, X in enumerate(compositions):
        print compositions[i], temperatures[i]
