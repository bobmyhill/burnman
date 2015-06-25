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


def activities(X, r, K): # X is mole fraction H2O
    Xb=X/(X + r*(1.-X)) # eq. 5.3b
    XO=1.-Xb-(0.5 - np.sqrt(0.25 - (K-4.)/K*(Xb-Xb*Xb)))/((K-4)/K) # eq. 5.3
    XH2O=XO + 2*Xb - 1.0
    XOH=2.*(Xb-XH2O)
    return np.power(XO,r), XH2O, XOH


def solveT(T, Xs, Tmelt, Smelt, r, K):
    return T-eqm_with_L(Xs, Tmelt, Smelt, r, K(T))

'''
# 6 GPa, en
r=3.0 # Oxygens available for bonding
K = lambda T: np.exp(-6000./T + 4.5) # equilibrium constant
Smelt=40. # J/K/mol
Tmelt=2010. # C
f=0.5 # ternary is 0.5*MgO + 0.5*SiO2 = 0.5*MgSiO3
Msil=100.3887
MH2O=18.01528
'''

# 13 GPa, per
r=1. # Oxygens available for bonding
Kinf = lambda T: 100000000000.
K0 = lambda T: 0.00000000001
K = lambda T: np.exp(-(-72000.-15.*T)/(R*T))
Smelt=22. # Cohen and Gong
Tmelt=5100. # C, Cohen and Gong
f=1.0
Msil=40.3044
MH2O=18.01528

Xbr=0.67 # composition of fluid in eqm with Xbr
#Xbr=0.76 
Tbr=1210. # C




XH2O = activities(Xbr, r, K(Tbr))[1]
dGH2O= R*(Tbr+273.15)*np.log(XH2O)


dGH2O_HP=1000.*((-829.82 - -547.59)- -274.10)
XH2O_HP=np.exp(dGH2O_HP/(R*(Tbr+273.15)))
print XH2O_HP, XH2O
print dGH2O_HP, dGH2O


compositions=np.linspace(0.0001, 0.99, 1001)
Gex=np.empty_like(compositions)
for i, X in enumerate(compositions):
    XS, XH2O, XOH = activities(X, r, K(Tbr))
    Gex[i]=(1-X)*R*(Tbr+273.15)*np.log(XS) + (X)*R*(Tbr+273.15)*np.log(XH2O)

plt.plot( compositions, Gex, '-', linewidth=2., label='model')
#plt.plot ( 0.0, Smelt*(Tbr-Tmelt), marker='o', label='model per')
plt.plot ( [1.0], [dGH2O_HP], marker='o', label='HP H2O activity')
plt.plot ( [0.0, 1.0], [R*(Tbr+273.15)*np.log(activities(Xbr, r, K(Tbr))[0]), R*(Tbr+273.15)*np.log(activities(Xbr, r, K(Tbr))[1])], marker='o', label='model H2O activity')
plt.ylabel("Excess Gibbs (J/mol)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()



H2Omolfraction=np.linspace(0.0001, 0.8, 1001)
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
    H2Owtfraction[i]=(H2Omolfraction[i]*MH2O)/(H2Omolfraction[i]*MH2O + (1.-H2Omolfraction[i])*Msil)
    ternaryH2Omolfraction[i]=H2Omolfraction[i]/(H2Omolfraction[i] + (1.-H2Omolfraction[i])/f)


periclase=[]
brucite=[]
liquid=[]
for line in open('../figures/13GPa_per-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[2] == 'p' or content[2] == 'sp':
            periclase.append([float(content[0]), float(content[1])/100.])
        if content[2] == 'l':
            liquid.append([float(content[0]), float(content[1])/100.])
        if content[2] == 'b':
            brucite.append([float(content[0]), float(content[1])/100.])

periclase=zip(*periclase)
brucite=zip(*brucite)
liquid=zip(*liquid)
plt.plot( periclase[1], periclase[0], marker='.', linestyle='none', label='per+liquid')
plt.plot( brucite[1], brucite[0], marker='.', linestyle='none', label='br+liquid')
plt.plot( liquid[1], liquid[0], marker='.', linestyle='none', label='superliquidus')

plt.plot( H2Omolfraction, temperatures, '-', linewidth=2., label='r='+str(r)+'; K=K(T)')
plt.plot( H2Omolfraction, temperaturesinf, '-', linewidth=2., label='r='+str(r)+'; K=inf')
plt.plot( H2Omolfraction, temperatures0, '-', linewidth=2., label='r='+str(r)+'; K=0')


plt.ylabel("Temperature (C)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()

