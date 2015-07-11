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

def find_temperature(T, P, X_one_cation, r, K, solid, liquid):
    n_silicate_per_water = (1.-X_one_cation)/(X_one_cation*n_cations)
    Xs = 1./(n_silicate_per_water + 1)
    Xb=Xs/(Xs + r*(1.-Xs)) # eq. 5.3b
    X0=1.-Xb-(0.5 - np.sqrt(0.25 - (K(T)-4.)/K(T)*(Xb-Xb*Xb)))/((K(T)-4)/K(T)) # eq. 5.3
    
    solid.set_state(P, T[0])
    liquid.set_state(P, T[0])
    return (liquid.gibbs - solid.gibbs) + R*T[0]*r*np.log(X0)

def find_eqm_temperature(T, P, solid, liquid, factor):
    solid.set_state(P, T[0])
    liquid.set_state(P, T[0])
    return liquid.gibbs*factor - solid.gibbs


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
    XO=1.-Xb-(0.5 - np.sqrt(0.25 - (K-4.)/K*(Xb-Xb*Xb)))/((K-4.)/K) # eq. 5.3
    XH2O=XO + 2*Xb - 1.0
    XOH=2.*(Xb-XH2O)
    return np.power(XO,r), XH2O, XOH


def solve_composition(Xs, T, r, K, Wsh, Whs):
    return dGfo(T) - excesses_nonideal(Xs, T, r, K(T), Wsh(T), Whs(T))[0]


# 13 GPa, fo
r=4./3. # Oxygens available for bonding (one cation basis)
n_cations = 1.
Kinf = lambda T: 100000000000.
K0 = lambda T: 0.00000000001
K1 = lambda T: np.exp(-(-70000.-15.*T)/(R*T))
Wsh1 = lambda T: 0

K = lambda T: 1000000000000. # np.exp(-(-50000-15.*(T - 2000.))/(R*T))
Wsh = lambda T: 00000.

Whs = lambda T: 00000.

'''
Tbr = 1.
X = 5./10.
print ((1-X)*excesses_nonideal(X, Tbr, r, K(Tbr), Wsh(Tbr), Whs(Tbr))[0] + X*excesses_nonideal(X, Tbr, r, K(Tbr), Wsh(Tbr), Whs(Tbr))[1])
exit()
'''

fo=SLB_2011.forsterite()
Mg2SiO4_liq=DKS_2013_liquids_tweaked.Mg2SiO4_liquid()

print fsolve(find_eqm_temperature, 2000., args=(13.e9, fo, Mg2SiO4_liq, 1.))

def dGfo(temperature):
    fo.set_state(13.e9, temperature)
    Mg2SiO4_liq.set_state(13.e9, temperature)
    return (fo.gibbs - Mg2SiO4_liq.gibbs)/3.


compositions=np.linspace(0.0001, 0.99, 101)
Gex=np.empty_like(compositions)
Gex_2=np.empty_like(compositions)
for i, X in enumerate(compositions):
    Tbr = 1000.
    Gex[i]=(1-X)*excesses_nonideal(X, Tbr, r, K(Tbr), Wsh(Tbr), Whs(Tbr))[0] + X*excesses_nonideal(X, Tbr, r, K(Tbr), Wsh(Tbr), Whs(Tbr))[1]
    Tmelt = 2000.
    Gex_2[i]=(1-X)*excesses_nonideal(X, Tmelt, r, K(Tmelt), Wsh(Tmelt), Whs(Tmelt))[0] + X*excesses_nonideal(X, Tmelt, r, K(Tmelt), Wsh(Tmelt), Whs(Tmelt))[1]

plt.plot( compositions, Gex, '-', linewidth=2., label='model at 1000 K')
plt.plot( compositions, Gex_2, '-', linewidth=2., label='model at 2000 K')
plt.ylabel("Excess Gibbs (J/mol)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()



fn0=lambda T: 0.
temperatures=np.linspace(600., 3000., 101)
compositions=np.empty_like(temperatures)
compositions0=np.empty_like(temperatures)
compositionsinf=np.empty_like(temperatures)

temperatures_fo=np.linspace(1600., 3000., 101)
compositions_fo=np.empty_like(temperatures_fo)


for i, T in enumerate(temperatures):
    compositions0[i]=fsolve(solve_composition, 0.001, args=(T, r, K0, fn0, fn0))
    compositionsinf[i]=fsolve(solve_composition, 0.001, args=(T, r, Kinf, fn0, fn0))
    #compositions[i]=fsolve(solve_composition, 0.001, args=(T, r, K, fn0, fn0))

for i, T in enumerate(temperatures_fo):
    compositions_fo[i]=fsolve(solve_composition, 0.001, args=(T, r, K, Wsh, Whs))


plt.plot( compositions_fo, temperatures_fo, linewidth=1, label='fo')

#plt.plot( compositions, temperatures, linewidth=1, label='K=K(T)')
plt.plot( compositionsinf, temperatures, linewidth=1, label='K=inf')
plt.plot( compositions0, temperatures, linewidth=1, label='K=0')


###################
# CALCULATE LIQUIDUS SPLINE
from scipy.interpolate import UnivariateSpline

Xs=[0.0, 0.2, 0.4, 0.55]
Ts=[2300., 1830., 1500., 1280.] # in C (Presnall and Walter, 1993 for dry melting)
spline_PW1993 = UnivariateSpline(Xs, Ts, s=1)

Xs_liquidus = np.linspace(0.0, 0.6, 101)
plt.plot(Xs_liquidus, spline_PW1993(Xs_liquidus)+273.15)
###################

forsterite = []
enstatite=[]
chondrodite=[]
liquid=[]
for line in open('data/13GPa_fo-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[3] == 'f' or content[3] == 'sf' or content[3] == 'f_davide':
            forsterite.append([float(content[0])+273.15, (100. - float(content[1])*7./2.)/100.])
        if content[3] == 'e' or content[3] == 'se' or content[3] == 'e_davide':
            enstatite.append([float(content[0])+273.15, (100. - float(content[1])*7./2.)/100.])
        if content[3] == 'c':
            chondrodite.append([float(content[0])+273.15, (100. - float(content[1])*7./2.)/100.])
        if content[3] == 'l' or content[3] == 'l_davide':
            liquid.append([float(content[0])+273.15,(100. - float(content[1])*7./2.)/100.])

forsterite=zip(*forsterite)
enstatite=zip(*enstatite)
chondrodite=zip(*chondrodite)
liquid=zip(*liquid)
plt.plot( forsterite[1], forsterite[0], marker='.', linestyle='none', label='fo+liquid')
plt.plot( enstatite[1], enstatite[0], marker='.', linestyle='none', label='en+liquid')
plt.plot( chondrodite[1], chondrodite[0], marker='.', linestyle='none', label='chond+liquid')
plt.plot( liquid[1], liquid[0], marker='.', linestyle='none', label='superliquidus')

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
    temperature = spline_PW1993(composition)+273.15
    activities[i] =  np.exp( dGfo(temperature) / (constants.gas_constant*temperature))

   
plt.plot(compositions, activities)
plt.title('Forsterite')
plt.xlim(0., 1.)
plt.ylim(0., 1.)
plt.show()
####################

data=[[compositions_fo, temperatures_fo],[compositions, temperatures],[compositionsinf, temperatures],[compositions0, temperatures]]

for datapair in data:
    print '>> -W1,black'
    compositions, temperatures=datapair
    for i, X in enumerate(compositions):
        print compositions[i], temperatures[i]
