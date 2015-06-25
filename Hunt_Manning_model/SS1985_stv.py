#!/usr/python
import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    DKS_2013_liquids_tweaked, \
    SLB_2011
from burnman import constants
import numpy as np
from scipy.optimize import fsolve, curve_fit
from scipy.interpolate import UnivariateSpline, interp1d, splrep, splev
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

R=8.31446 # from wiki


# Gmelt - Gstv = Smelt(Tmelt - T) = -RT*(rlogX0)
def find_temperature(T, P, Xs, r, K, solid, liquid):
    Xb=Xs/(Xs + r*(1.-Xs)) # eq. 5.3b
    X0=1.-Xb-(0.5 - np.sqrt(0.25 - (K(T)-4.)/K(T)*(Xb-Xb*Xb)))/((K(T)-4)/K(T)) # eq. 5.3
    
    solid.set_state(P, T[0])
    liquid.set_state(P, T[0])
    return (liquid.gibbs - solid.gibbs) + R*T[0]*r*np.log(X0)

'''
def eqm_with_L(X, Tmelt, Smelt, r, K): # X is mole fraction H2O
    Xb=X/(X + r*(1.-X)) # eq. 5.3b
    X0=1.-Xb-(0.5 - np.sqrt(0.25 - (K-4.)/K*(Xb-Xb*Xb)))/((K-4)/K) # eq. 5.3

    factor=1.-(r*R*np.log(X0))/Smelt
    T=Tmelt/factor # eq. 13
    
    return T



def solveT(T, Xs, Tmelt, Smelt, r, K):
    return T-eqm_with_L(Xs, Tmelt, Smelt, r, K(T))
'''


'''
# 6 GPa, en
r=3.0 # Oxygens available for bonding
Kinf = lambda T: 100000000000.
K0 = lambda T: 0.00000000001
K = lambda T: np.exp(-6000./T + 4.5) # equilibrium constant
Smelt=40. # J/K/mol
Tmelt=2010. # C
f=0.5 # ternary is 0.5*MgO + 0.5*SiO2 = 0.5*MgSiO3
Msil=100.3887
MH2O=18.01528
'''

# 13 GPa, stv
r=2. # Oxygens available for bonding
Kinf = lambda T: 100000000000.
K0 = lambda T: 0.00000000001
K = lambda T: np.exp(-(35000-20.*T)/(R*T))
Smelt=30. #120. # J/K/mol # reasonable range: 8--19 (corresponding to K=inf, 0)
Tmelt=2600. # C, 2700 is a reasonable metastable stishovite melting point at 13 GPa
f=1.0
Msil=60.08
MH2O=18.01528



SiO2_liq = DKS_2013_liquids_tweaked.SiO2_liquid()
stv = SLB_2011.stishovite()
P=12.5e9

H2Omolfraction=np.linspace(0.0001, 0.65, 101)
temperatures=np.empty_like(H2Omolfraction)
temperaturesinf=np.empty_like(H2Omolfraction)
temperatures0=np.empty_like(H2Omolfraction)
H2Owtfraction=np.empty_like(H2Omolfraction)
ternaryH2Omolfraction=np.empty_like(H2Omolfraction)
for i, Xs in enumerate(H2Omolfraction):

    temperatures[i] = fsolve(find_temperature, 2000., args=(P, Xs, r, K, stv, SiO2_liq))
    temperaturesinf[i]=fsolve(find_temperature, 2000., args=(P, Xs, r, Kinf, stv, SiO2_liq))
    temperatures0[i]=fsolve(find_temperature, 2000., args=(P, Xs, r, K0, stv, SiO2_liq))

    print Xs, temperatures0[i], temperatures[i], temperaturesinf[i]

    H2Owtfraction[i]=(H2Omolfraction[i]*MH2O)/(H2Omolfraction[i]*MH2O + (1.-H2Omolfraction[i])*Msil)
    ternaryH2Omolfraction[i]=H2Omolfraction[i]/(H2Omolfraction[i] + (1.-H2Omolfraction[i])/f)


stishovite=[]
liquid=[]
for line in open('data/13GPa_SiO2-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[2] == 's':
            stishovite.append([float(content[0])+273.15, 1.-float(content[1])])
        if content[2] == 'l':
            liquid.append([float(content[0])+273.15, 1.-float(content[1])])

stishovite=zip(*stishovite)
liquid=zip(*liquid)
plt.plot( stishovite[1], stishovite[0], marker='.', linestyle='none', label='stv+liquid')
plt.plot( liquid[1], liquid[0], marker='.', linestyle='none', label='superliquidus')

plt.plot( H2Omolfraction, temperatures, '-', linewidth=2., label='r='+str(r)+'; K=K(T)')
plt.plot( H2Omolfraction, temperaturesinf, '-', linewidth=2., label='r='+str(r)+'; K=inf')
plt.plot( H2Omolfraction, temperatures0, '-', linewidth=2., label='r='+str(r)+'; K=0')


plt.ylabel("Temperature (K)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()




dTdP = 1.19423269439e-07
P = 13.7e9 # GPa
T = 3073.15 #K

P2=13.e9

T2=T - (P-P2)*dTdP


print T2-273.15, 'C'
