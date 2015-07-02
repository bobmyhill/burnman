#!/usr/python
import os, sys
import numpy as np
sys.path.insert(1,os.path.abspath('..'))

from scipy.optimize import fsolve, minimize, fmin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



# Benchmarks for the solid solution class
import burnman
from burnman.minerals import \
    DKS_2013_liquids_tweaked, \
    SLB_2011
from burnman import tools
from burnman.processchemistry import *
from burnman.chemicalpotentials import *
from burnman import constants
atomic_masses=read_masses()

R=8.31446 # from wiki


X=0.67
r=1.
P=13.e9
T=1300.

def find_eqm_temperature(T, P, solid, liquid, factor):
    solid.set_state(P, T[0])
    liquid.set_state(P, T[0])
    return liquid.gibbs*factor - solid.gibbs

def find_temperature(T, P, Xs, r, K, solid, liquid):
    Xb=Xs/(Xs + r*(1.-Xs)) # eq. 5.3b
    X0=1.-Xb-(0.5 - np.sqrt(0.25 - (K(T)-4.)/K(T)*(Xb-Xb*Xb)))/((K(T)-4)/K(T)) # eq. 5.3
    
    solid.set_state(P, T[0])
    liquid.set_state(P, T[0])
    return (liquid.gibbs - solid.gibbs) + R*T[0]*r*np.log(X0)

'''
def eqm_with_L(X, Tmelt, Smelt, r, K): # X is mole fraction H2O
    Xb=X/(X + r*(1.-X)) # eq. 5.3b
    XO=1.-Xb-(0.5 - np.sqrt(0.25 - (K-4.)/K*(Xb-Xb*Xb)))/((K-4)/K) # eq. 5.3

    factor=1.-(r*R*np.log(XO))/Smelt
    T=Tmelt/factor # eq. 13
    return T
'''

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

def solve_composition(Xs, T, r, K, Wsh, Whs):
    return dGper(T) - excesses_nonideal(Xs, T, r, K(T), Wsh(T), Whs(T))[0]

def solve_composition_br(Xs, T, r, K, Wsh, Whs):
    return dGbr(T) - 0.5*(excesses_nonideal(Xs, T, r, K(T), Wsh(T), Whs(T))[0] + excesses_nonideal(Xs, T, r, K(T), Wsh(T), Whs(T))[1])


# 13 GPa, per
r=1. # Oxygens available for bonding
Kinf = lambda T: 100000000000.
K0 = lambda T: 0.00000000001
K1 = lambda T: np.exp(-(-70000.-15.*T)/(R*T))
Wsh1 = lambda T: 0

K = lambda T: np.exp(-(-75000.-0.*(T-1483.))/(R*T))
Wsh = lambda T: 55000.

Whs = lambda T: 00000.


per=SLB_2011.periclase()
MgO_liq=DKS_2013_liquids_tweaked.MgO_liquid()

T_melt = fsolve(find_eqm_temperature, 3000., args=(13.e9, per, MgO_liq, 1.))[0]
print T_melt


def dGper(temperature):
    per.set_state(13.e9, temperature)
    MgO_liq.set_state(13.e9, temperature)
    return per.gibbs - MgO_liq.gibbs

def dGbr(temperature):
    return 0.5*dGper(temperature) + 0.5*(- 8330. + 20.*(temperature-1473.15))

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
    Tmelt = 4000.
    Gex_2[i]=(1-X)*excesses_nonideal(X, Tmelt, r, K(Tmelt), Wsh(Tmelt), Whs(Tmelt))[0] + X*excesses_nonideal(X, Tmelt, r, K(Tmelt), Wsh(Tmelt), Whs(Tmelt))[1]

plt.plot( compositions, Gex, '-', linewidth=2., label='model at Tbr')
plt.plot( compositions, Gex_2, '-', linewidth=2., label='model at 4000 K')
plt.plot ( 0.0, dGper(Tbr), marker='o', label='model per')
plt.plot ( [1.0], [dGH2O_HP], marker='o', label='HP H2O activity')
plt.plot ( [0.0, 1.0], [excesses_nonideal(Xbr, Tbr, r, K(Tbr), Wsh(Tbr), Whs(Tbr))[0], excesses_nonideal(Xbr, Tbr, r, K(Tbr), Wsh(Tbr), Whs(Tbr))[1]], marker='o', label='model H2O activity')
plt.ylabel("Excess Gibbs (J/mol)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()

def per_br_eqm(data, r, K, Wsh, Whs):
    Xs=data[0]
    T =data[1]
    Gex= excesses_nonideal(Xs, T, r, K(T), Wsh(T), Whs(T))
    values = []
    values.append(dGper(T) - Gex[0])
    values.append(dGbr(T) - 0.5*(Gex[0] + Gex[1]))
    return values


XsT = fsolve(per_br_eqm, [0.9, 1273.], args=(r, K, Wsh, Whs))
print XsT
T_per_br=XsT[1]

fn0=lambda T: 0.
temperatures=np.linspace(600., T_melt, 101)
compositions=np.empty_like(temperatures)
compositions0=np.empty_like(temperatures)
compositionsinf=np.empty_like(temperatures)

temperatures_per=np.linspace(T_per_br, T_melt, 101)
compositions_per=np.empty_like(temperatures_per)

temperatures_br=np.linspace(600., T_per_br, 101)
compositions_br=np.empty_like(temperatures_br)

for i, T in enumerate(temperatures):
    compositions0[i]=fsolve(solve_composition, 0.001, args=(T, r, K0, fn0, fn0))
    compositionsinf[i]=fsolve(solve_composition, 0.001, args=(T, r, Kinf, fn0, fn0))
    #compositions[i]=fsolve(solve_composition, 0.001, args=(T, r, K, fn0, fn0))

for i, T in enumerate(temperatures_per):
    compositions_per[i]=fsolve(solve_composition, 0.99, args=(T, r, K, Wsh, Whs))

for i, T in enumerate(temperatures_br):
    compositions_br[i]=fsolve(solve_composition_br, 0.99, args=(T, r, K, Wsh, Whs))

plt.plot( compositions_per, temperatures_per, linewidth=1, label='per')
plt.plot( compositions_br, temperatures_br, linewidth=1, label='br')

#plt.plot( compositions, temperatures, linewidth=1, label='K=K(T)')
plt.plot( compositionsinf, temperatures, linewidth=1, label='K=inf')
plt.plot( compositions0, temperatures, linewidth=1, label='K=0')

periclase=[]
brucite=[]
liquid=[]
for line in open('data/13GPa_per-H2O.dat'):
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

####################
# a-X relationships
activities_per = np.empty_like(compositions_per)
for i, composition in enumerate(compositions_per):
    temperature = temperatures_per[i]
    activities_per[i] =  np.exp( dGper(temperature) / (constants.gas_constant*temperature))



   
plt.plot(compositions_per, activities_per)
plt.title('Periclase')
plt.xlim(0., 1.)
plt.ylim(0., 1.)
plt.show()
####################

data=[[compositions_per, temperatures_per],[compositions_br, temperatures_br],[compositions, temperatures],[compositionsinf, temperatures],[compositions0, temperatures]]

for datapair in data:
    print '>> -W1,black'
    compositions, temperatures=datapair
    for i, X in enumerate(compositions):
        print compositions[i], temperatures[i]
