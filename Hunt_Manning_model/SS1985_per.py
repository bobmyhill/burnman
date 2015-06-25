#!/usr/python
import os, sys
import numpy as np

from scipy.optimize import fsolve, minimize, fmin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if not os.path.exists('burnman'):
    sys.path.insert(1,os.path.abspath('/home/rm438/projects/burnman/'))


# Benchmarks for the solid solution class
import burnman
from burnman import minerals
from burnman import tools
from burnman.mineral import Mineral
from burnman.minerals import Myhill_calibration_iron
from burnman.processchemistry import *
from burnman.chemicalpotentials import *
from burnman import constants
atomic_masses=read_masses()

R=8.31446 # from wiki


X=0.67
r=1.
P=13.e9
T=1300.


def eqm_with_L(X, Tmelt, Smelt, r, K): # X is mole fraction H2O
    Xb=X/(X + r*(1.-X)) # eq. 5.3b
    XO=1.-Xb-(0.5 - np.sqrt(0.25 - (K-4.)/K*(Xb-Xb*Xb)))/((K-4)/K) # eq. 5.3

    factor=1.-(r*R*np.log(XO))/Smelt
    T=Tmelt/factor # eq. 13
    return T


def excesses_nonideal(X, T, r, K, Wsh, Whs): # X is mole fraction H2O
    Xb=X/(X + r*(1.-X)) # eq. 5.3b
    XO=1.-Xb-(0.5 - np.sqrt(0.25 - (K-4.)/K*(Xb-Xb*Xb)))/((K-4)/K) # eq. 5.3

    activity_anhydrous_phase=np.power(XO,r)
    activity_H2O=XO + 2*Xb - 1.0

    partial_excess_anhydrous_phase=R*T*np.log(activity_anhydrous_phase)
    partial_excess_H2O=R*T*np.log(activity_H2O)

    #partial_excess_anhydrous_phase+=Xb*Xb*W
    #partial_excess_H2O+=(1.-Xb*Xb)*W

    Xs=1.-Xb
    partial_excess_anhydrous_phase+= 2.*Xb*Xb*(1.-Xb)*Whs - Xb*Xb*(1.-2.*Xb)*Wsh 
    partial_excess_H2O+= 2.*Xs*Xs*(1.-Xs)*Wsh - Xs*Xs*(1.-2.*Xs)*Whs 
    return partial_excess_anhydrous_phase, partial_excess_H2O

def activities(X, r, K): # X is mole fraction H2O
    Xb=X/(X + r*(1.-X)) # eq. 5.3b
    XO=1.-Xb-(0.5 - np.sqrt(0.25 - (K-4.)/K*(Xb-Xb*Xb)))/((K-4)/K) # eq. 5.3
    XH2O=XO + 2*Xb - 1.0
    XOH=2.*(Xb-XH2O)
    return np.power(XO,r), XH2O, XOH

def solveT(T, Xs, Tmelt, Smelt, r, K):
    return T-eqm_with_L(Xs, Tmelt, Smelt, r, K(T))

def solve_composition(Xs, T, Tmelt, Smelt, r, K, Wsh, Whs):
    return dGper(T) - excesses_nonideal(Xs, T, r, K(T), Wsh(T), Whs(T))[0]

def solve_composition_br(Xs, T, Tmelt, Smelt, r, K, Wsh, Whs):
    return dGbr(T) - 0.5*(excesses_nonideal(Xs, T, r, K(T), Wsh(T), Whs(T))[0] + excesses_nonideal(Xs, T, r, K(T), Wsh(T), Whs(T))[1])

'''
    P=13.e9
    minimize(eqm_order, 0.001, method='TNC', bounds=((0.0, 1.-np.abs(1.-2.*X)),), args=(X, r, P, T), options={'disp': False}).x[0] 
    print liq.partial_gibbs
    print liq.partial_gibbs[0]
    return Smelt*(T-Tmelt)-liq.partial_gibbs[0]
'''

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
K1 = lambda T: np.exp(-(-70000.-15.*T)/(R*T))
Wsh1 = lambda T: 0

K = lambda T: np.exp(-(-98000.-0.*(T-1483.))/(R*T))
Wsh = lambda T: 48000.

Whs = lambda T: 00000.
Smelt=22. # Cohen and Gong
Tmelt=5373. # K, Cohen and Gong
f=1.0
Msil=40.3044
MH2O=18.01528

dGper=lambda T: Smelt*(T-Tmelt)
dGbr=lambda T: 0.5*(Smelt*(T-Tmelt)) + 0.5*(- 8330. + 20.*(T-1473.15))

Xbr=0.67 # composition of fluid in eqm with Xbr
#Xbr=0.76 
Tbr=1210.+273.15 # K


XH2O = np.exp(excesses_nonideal(Xbr, Tbr, r, K(Tbr), Wsh(Tbr), Whs(Tbr))[1]/(R*Tbr))
dGH2O= R*Tbr*np.log(XH2O)


dGH2O_HP=1000.*((-829.82 - -547.59)- -274.10)
XH2O_HP=np.exp(dGH2O_HP/(R*Tbr))
print XH2O_HP, XH2O
print dGH2O_HP, dGH2O


compositions=np.linspace(0.0001, 0.99, 101)
Gex=np.empty_like(compositions)
Gex_2=np.empty_like(compositions)
for i, X in enumerate(compositions):
    Gex[i]=(1-X)*excesses_nonideal(X, Tbr, r, K(Tbr), Wsh(Tbr), Whs(Tbr))[0] + X*excesses_nonideal(X, Tbr, r, K(Tbr), Wsh(Tbr), Whs(Tbr))[1]
    Gex_2[i]=(1-X)*excesses_nonideal(X, Tmelt, r, K(Tmelt), Wsh(Tmelt), Whs(Tmelt))[0] + X*excesses_nonideal(X, Tmelt, r, K(Tmelt), Wsh(Tmelt), Whs(Tmelt))[1]

plt.plot( compositions, Gex, '-', linewidth=2., label='model at Tbr')
plt.plot( compositions, Gex_2, '-', linewidth=2., label='model at Tmelt')
#plt.plot ( 0.0, Smelt*(Tbr-Tmelt), marker='o', label='model per')
plt.plot ( [1.0], [dGH2O_HP], marker='o', label='HP H2O activity')
plt.plot ( [0.0, 1.0], [excesses_nonideal(Xbr, Tbr, r, K(Tbr), Wsh(Tbr), Whs(Tbr))[0], excesses_nonideal(Xbr, Tbr, r, K(Tbr), Wsh(Tbr), Whs(Tbr))[1]], marker='o', label='model H2O activity')
plt.ylabel("Excess Gibbs (J/mol)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()

def per_br_eqm(T, Tmelt, Smelt, r, K, Wsh, Whs):
    return fsolve(solve_composition, 0.001, args=(T, Tmelt, Smelt, r, K, Wsh, Whs)) - fsolve(solve_composition_br, 0.99, args=(T, Tmelt, Smelt, r, K, Wsh, Whs)) 

T_per_br=fsolve(per_br_eqm, 1210+273.15, args=(Tmelt, Smelt, r, K, Wsh, Whs))


fn0=lambda T: 0.
temperatures=np.linspace(600., 5373., 101)
compositions=np.empty_like(temperatures)
compositions0=np.empty_like(temperatures)
compositionsinf=np.empty_like(temperatures)

temperatures_per=np.linspace(T_per_br, 5373., 101)
compositions_per=np.empty_like(temperatures_per)

temperatures_br=np.linspace(600., T_per_br, 101)
compositions_br=np.empty_like(temperatures_br)

for i, T in enumerate(temperatures):
    compositions0[i]=fsolve(solve_composition, 0.001, args=(T, Tmelt, Smelt, r, K0, fn0, fn0))
    compositionsinf[i]=fsolve(solve_composition, 0.001, args=(T, Tmelt, Smelt, r, Kinf, fn0, fn0))
    compositions[i]=fsolve(solve_composition, 0.001, args=(T, Tmelt, Smelt, r, K, fn0, fn0))

for i, T in enumerate(temperatures_per):
    compositions_per[i]=fsolve(solve_composition, 0.001, args=(T, Tmelt, Smelt, r, K, Wsh, Whs))

for i, T in enumerate(temperatures_br):
    compositions_br[i]=fsolve(solve_composition_br, 0.99, args=(T, Tmelt, Smelt, r, K, Wsh, Whs))

plt.plot( compositions_per, temperatures_per, linewidth=1, label='per')
plt.plot( compositions_br, temperatures_br, linewidth=1, label='br')

plt.plot( compositions, temperatures, linewidth=1, label='K=K(T)')
plt.plot( compositionsinf, temperatures, linewidth=1, label='K=inf')
plt.plot( compositions0, temperatures, linewidth=1, label='K=0')
periclase=[]
brucite=[]
liquid=[]
for line in open('../figures/13GPa_per-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[2] == 'p' or content[2] == 'sp':
            periclase.append([float(content[0])+273.15, float(content[1])/100.])
        if content[2] == 'l':
            liquid.append([float(content[0])+273.15, float(content[1])/100.])
        if content[2] == 'b':
            brucite.append([float(content[0])+273.15, float(content[1])/100.])

periclase=zip(*periclase)
brucite=zip(*brucite)
liquid=zip(*liquid)
plt.plot( periclase[1], periclase[0], marker='.', linestyle='none', label='per+liquid')
plt.plot( brucite[1], brucite[0], marker='.', linestyle='none', label='br+liquid')
plt.plot( liquid[1], liquid[0], marker='.', linestyle='none', label='superliquidus')

plt.ylim(1000.,5500.)
plt.xlim(0.,1.)
plt.ylabel("Temperature (K)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()


data=[[compositions_per, temperatures_per],[compositions_br, temperatures_br],[compositions, temperatures],[compositionsinf, temperatures],[compositions0, temperatures]]

for datapair in data:
    print '>> -W1,black'
    compositions, temperatures=datapair
    for i, X in enumerate(compositions):
        print compositions[i], temperatures[i]