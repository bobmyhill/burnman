#!/usr/python

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

R=8.31446 # from wiki


def eqm_with_L(X, Tmelt, Smelt, r, K): # X is mole fraction H2O
    Xb=X/(X + r*(1.-X)) # eq. 5.3b
    X0=1.-Xb-(0.5 - np.sqrt(0.25 - (K-4.)/K*(Xb-Xb*Xb)))/((K-4)/K) # eq. 5.3

    factor=1.-(r*R*np.log(X0))/Smelt
    T=Tmelt/factor # eq. 13
    return T


def solveT(T, Xs, Tmelt, Smelt, r, K):
    return T-eqm_with_L(Xs, Tmelt, Smelt, r, K(T))

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
Smelt=13.5 #120. # J/K/mol # reasonable range: 8--19 (corresponding to K=inf, 0)
Tmelt=2700. # C
f=1.0
Msil=60.08
MH2O=18.01528

H2Omolfraction=np.linspace(0.0001, 0.65, 101)
temperatures=np.empty_like(H2Omolfraction)
temperaturesinf=np.empty_like(H2Omolfraction)
temperatures0=np.empty_like(H2Omolfraction)
H2Owtfraction=np.empty_like(H2Omolfraction)
ternaryH2Omolfraction=np.empty_like(H2Omolfraction)
for i, Xs in enumerate(H2Omolfraction):
    #temperatures[i]=eqm_with_L(Xs, Tmelt, Smelt, r, K)
    temperatures[i]=fsolve(solveT, 1400., args=(Xs, Tmelt, Smelt, r, K))
    temperaturesinf[i]=fsolve(solveT, 1400., args=(Xs, Tmelt, Smelt, r, Kinf))
    temperatures0[i]=fsolve(solveT, 1400., args=(Xs, Tmelt, Smelt, r, K0))

    print Xs, temperatures0[i], temperatures[i], temperaturesinf[i]

    H2Owtfraction[i]=(H2Omolfraction[i]*MH2O)/(H2Omolfraction[i]*MH2O + (1.-H2Omolfraction[i])*Msil)
    ternaryH2Omolfraction[i]=H2Omolfraction[i]/(H2Omolfraction[i] + (1.-H2Omolfraction[i])/f)


stishovite=[]
liquid=[]
for line in open('../figures/13GPa_SiO2-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[2] == 's':
            stishovite.append([float(content[0]), 1.-float(content[1])])
        if content[2] == 'l':
            liquid.append([float(content[0]), 1.-float(content[1])])

stishovite=zip(*stishovite)
liquid=zip(*liquid)
plt.plot( stishovite[1], stishovite[0], marker='.', linestyle='none', label='stv+liquid')
plt.plot( liquid[1], liquid[0], marker='.', linestyle='none', label='superliquidus')

plt.plot( H2Omolfraction, temperatures, '-', linewidth=2., label='r='+str(r)+'; K=K(T)')
plt.plot( H2Omolfraction, temperaturesinf, '-', linewidth=2., label='r='+str(r)+'; K=inf')
plt.plot( H2Omolfraction, temperatures0, '-', linewidth=2., label='r='+str(r)+'; K=0')


plt.ylabel("Temperature (C)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()

