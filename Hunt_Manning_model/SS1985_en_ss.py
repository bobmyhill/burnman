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
from burnman.minerals import SLB_2011
from burnman import tools
from burnman.processchemistry import *
from burnman.chemicalpotentials import *


# 13 GPa, fo
r=4./3. # Oxygens available for bonding (one cation basis)
n_cations = 1.
Kinf = lambda T: 100000000000.
K0 = lambda T: 0.00000000001
K1 = lambda T: 1
G = lambda T: 0. - 120.*(T-1800.)
K = lambda T: np.exp(-(G(T))/(R*T))
Wsh = lambda T: 00000.
Whs = lambda T: 00000.

pressure = 13.e9
anhydrous_phase=SLB_2011.hp_clinoenstatite()
liquid=MgO_SiO2_liquid()
liquid.set_composition([1./2., 1./2.])

Tmelt = fsolve(delta_gibbs, 2000., args=(pressure, anhydrous_phase, liquid, 1./4., 1.))[0]
print Tmelt

compositions=np.linspace(0.0001, 0.99, 101)
Gex=np.empty_like(compositions)
Gex_2=np.empty_like(compositions)
for i, X in enumerate(compositions):
    T0 = 1000.    
    Gex[i]=(1-X)*excesses_nonideal(X, T0, r, K(T0), Wsh(T0), Whs(T0))[0] + X*excesses_nonideal(X, T0, r, K(T0), Wsh(T0), Whs(T0))[1]
    Gex_2[i]=(1-X)*excesses_nonideal(X, Tmelt, r, K(Tmelt), Wsh(Tmelt), Whs(Tmelt))[0] + X*excesses_nonideal(X, Tmelt, r, K(Tmelt), Wsh(Tmelt), Whs(Tmelt))[1]

plt.plot( compositions, Gex, '-', linewidth=2., label='model at 1500 K')
plt.plot( compositions, Gex_2, '-', linewidth=2., label='model at 3000 K')


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
    compositions0[i]=fsolve(solve_composition, 0.001, args=(T, pressure, r, K0, fn0, fn0, anhydrous_phase, liquid, 1./4., 1.))
    compositions1[i]=fsolve(solve_composition, 0.001, args=(T, pressure, r, K1, fn0, fn0, anhydrous_phase, liquid, 1./4., 1.))
    compositionsinf[i]=fsolve(solve_composition, 0.001, args=(T, pressure, r, Kinf, fn0, fn0, anhydrous_phase, liquid, 1./4., 1.))
    compositions[i]=fsolve(solve_composition, 0.001, args=(T, pressure, r, K, Wsh, Whs, anhydrous_phase, liquid, 1./4., 1.))


plt.plot( compositions, temperatures, linewidth=1, label='hen')
plt.plot( compositions0, temperatures, linewidth=1, label='K=0')
plt.plot( compositions1, temperatures, linewidth=1, label='K=1')
plt.plot( compositionsinf, temperatures, linewidth=1, label='K=inf')



###################
# CALCULATE LIQUIDUS SPLINE
from scipy.interpolate import UnivariateSpline
add_T = 30. # K

Xs=[0.0, 0.2, 0.5, 0.6]
Ts=[2186., 1770.+add_T, 1495.+add_T, 1440.+add_T] # in C (Kato and Kumazawa, 1986a for dry melting)
spline_KK1986 = UnivariateSpline(Xs, Ts, s=1)

Xs=[0.0, 0.2, 0.5, 0.6]
Ts=[2513.28 - 273.15, 1800.+add_T, 1495.+add_T, 1440.+add_T] # in C (Presnall and Gasparik, 1990 for dry melting)
spline_PG1990 = UnivariateSpline(Xs, Ts, s=1)

Xs_liquidus = np.linspace(0.0, 0.6, 101)
plt.plot(Xs_liquidus, spline_KK1986(Xs_liquidus)+273.15)
plt.plot(Xs_liquidus, spline_PG1990(Xs_liquidus)+273.15)
###################

enstatite=[]
superliquidus=[]
y_liquid=[]
for line in open('data/13GPa_en-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[2] == 'e' or content[2] == 'se' or content[2] == 'e_davide':
            enstatite.append([float(content[0])+273.15, (100. - float(content[1])*2.)/100.])
        if content[2] == 'l' or content[2] == 'l_davide':
            superliquidus.append([float(content[0])+273.15,(100. - float(content[1])*2.)/100.])
        if content[2] == 'l_Yamada':
            y_liquid.append([float(content[0])+273.15,(100. - float(content[1])*2.)/100.])

enstatite=np.array(zip(*enstatite))
superliquidus=np.array(zip(*superliquidus))
y_liquid=np.array(zip(*y_liquid))
plt.plot( y_liquid[1], y_liquid[0], linewidth=1, label='liquidus (Yamada et al., 2004)')
plt.plot( enstatite[1], enstatite[0] + add_T, marker='.', linestyle='none', label='en+liquid')
plt.plot( superliquidus[1], superliquidus[0] + add_T, marker='.', linestyle='none', label='superliquidus')

plt.ylim(1000.,3000.)
plt.xlim(0.,1.)
plt.ylabel("Temperature (K)")
plt.xlabel("X")
plt.legend(loc='upper right')
plt.show()

####################
# a-X relationships (1 cation basis)
compositions = np.linspace(0., 0.6, 101)
activities = np.empty_like(temperatures)
for i, composition in enumerate(compositions):
    temperature = spline_PG1990(composition)+273.15
    activities[i] =  np.exp( delta_gibbs([temperature], pressure, anhydrous_phase, liquid, 1./4., 1.)  / (constants.gas_constant*temperature))

   
plt.plot(compositions, activities)
plt.title('Enstatite')
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
