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
from burnman.minerals import DKS_2013_liquids
from burnman import tools
from burnman.processchemistry import *
from burnman.chemicalpotentials import *


def dGbr(temperature):
    return 0.5*delta_gibbs([temperature], 13.e9, anhydrous_phase, liquid, 1., 1.) + 0.5*(- 8330. + 20.*(temperature-1473.15))

def solve_composition_br(Xs, T, r, K, Wsh, Whs):
    return dGbr(T) - 0.5*(excesses_nonideal(Xs, T, r, K(T), Wsh, Whs)[0] + excesses_nonideal(Xs, T, r, K(T), Wsh, Whs)[1])



# 13 GPa, per
r=1. # Oxygens available for bonding (one cation basis)
n_cations = 1.
Kinf = lambda T: 100000000000.
K0 = lambda T: 0.00000000001
K1 = lambda T: 1
G = lambda T: -75000.-0.*(T-1483.)
K = lambda T: np.exp(-(G(T))/(R*T))
Wsh = lambda T: 55000.
Whs = lambda T: 0.


pressure = 13.e9
anhydrous_phase=SLB_2011.periclase()
liquid=DKS_2013_liquids.MgO_liquid()

Tmelt = 4180.
anhydrous_phase.set_state(pressure, Tmelt)
liquid.set_state(pressure, Tmelt)
liquid.params['a'][0][0] += anhydrous_phase.gibbs - liquid.gibbs


#Tmelt = tools.equilibrium_temperature([anhydrous_phase, liquid], [1.0, -1.0], 1.e9, 300.)
#print '1 GPa:', Tmelt

Tmelt = tools.equilibrium_temperature([anhydrous_phase, liquid], [1.0, -1.0], pressure, 3900.)
print '13 GPa:', Tmelt
print liquid.S - anhydrous_phase.S


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
plt.plot( compositions, Gex_2, '-', linewidth=2., label='model at 4000 K')
plt.plot ( 0.0, delta_gibbs([Tbr], 13.e9, anhydrous_phase, liquid, 1., 1.), marker='o', label='model per')
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
    values.append(delta_gibbs([T], pressure, anhydrous_phase, liquid, 1., 1.) - Gex[0])
    values.append(dGbr(T) - 0.5*(Gex[0] + Gex[1]))
    return values


XsT = fsolve(per_br_eqm, [0.9, 1273.], args=(r, K, Wsh, Whs))
print XsT
T_per_br=XsT[1]

fn0=0.
temperatures=np.linspace(600., Tmelt, 101)
compositions1=np.empty_like(temperatures)
compositions0=np.empty_like(temperatures)
compositionsinf=np.empty_like(temperatures)

temperatures_per=np.linspace(T_per_br, Tmelt, 101)
compositions_per=np.empty_like(temperatures_per)

temperatures_br=np.linspace(600., T_per_br, 101)
compositions_br=np.empty_like(temperatures_br)

guess0 = 0.999
guess1 = 0.999
guess_inf = 0.999
for i, T in enumerate(temperatures):
    print(T)
    guess0=fsolve(solve_composition, guess0, args=(T, pressure, r, K0, fn0, fn0, anhydrous_phase, liquid, 1., 1.))
    guess1=fsolve(solve_composition, guess1, args=(T, pressure, r, K, fn0, fn0, anhydrous_phase, liquid, 1., 1.))
    guess_inf=fsolve(solve_composition, guess_inf, args=(T, pressure, r, Kinf, fn0, fn0, anhydrous_phase, liquid, 1., 1.))

    compositions0[i] = guess0
    compositions1[i] = guess1
    compositionsinf[i] = guess_inf
    
guess_per = 0.99
for i, T in enumerate(temperatures_per):
    guess_per=fsolve(solve_composition, guess_per, args=(T, pressure, r, K, Wsh(T), Whs(T), anhydrous_phase, liquid, 1., 1.))
    compositions_per[i] = guess_per
    
guess_br =0.99
for i, T in enumerate(temperatures_br):
    guess_br=fsolve(solve_composition_br, guess_br, args=(T, r, K, Wsh(T), Whs(T))) # only good at 13 GPa
    compositions_br[i] = guess_br
    
print 'PERICLASE EQM'
temperatures_eqm=np.linspace(1273.15,4173.15, 30) 
for i, T in enumerate(temperatures_eqm):
    print T-273.15, fsolve(solve_composition, 0.99, args=(T, pressure, r, K, Wsh(T), Whs(T), anhydrous_phase, liquid, 1., 1.))

print 'BRUCITE EQM'
temperatures_eqm=np.linspace(873.15,1773.15, 19) 
for i, T in enumerate(temperatures_eqm):
    print T-273.15, fsolve(solve_composition_br, 0.99, args=(T, r, K, Wsh(T), Whs(T))) # only good at 13 GPa


for i, T in enumerate(temperatures_br):
    compositions_br[i]=fsolve(solve_composition_br, 0.99, args=(T, r, K, Wsh(T), Whs(T))) # only good at 13 GPa

plt.plot( compositions_per, temperatures_per, linewidth=1, label='per')
plt.plot( compositions_br, temperatures_br, linewidth=1, label='br')

plt.plot( compositions0, temperatures, linewidth=1, label='K=0')
plt.plot( compositions1, temperatures, linewidth=1, label='K=1')
plt.plot( compositionsinf, temperatures, linewidth=1, label='K=inf')

periclase=[]
brucite=[]
superliquidus=[]
for line in open('data/13GPa_per-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[2] == 'p' or content[2] == 'sp':
            periclase.append([float(content[0])+273.15, float(content[1])/100.])
        if content[2] == 'l':
            superliquidus.append([float(content[0])+273.15, float(content[1])/100.])
        if content[2] == 'b':
            brucite.append([float(content[0])+273.15, float(content[1])/100.])

periclase=zip(*periclase)
brucite=zip(*brucite)
superliquidus=zip(*superliquidus)
plt.plot( periclase[1], periclase[0], marker='.', linestyle='none', label='per+liquid')
plt.plot( brucite[1], brucite[0], marker='.', linestyle='none', label='br+liquid')
plt.plot( superliquidus[1], superliquidus[0], marker='.', linestyle='none', label='superliquidus')

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
    activities_per[i] =  np.exp( delta_gibbs([temperature], pressure, anhydrous_phase, liquid, 1., 1.) / (constants.gas_constant*temperature))



   
plt.plot(compositions_per, activities_per)
plt.title('Periclase')
plt.xlim(0., 1.)
plt.ylim(0., 1.)
plt.show()
####################

data=[[compositions_per, temperatures_per],[compositions_br, temperatures_br],[compositions0, temperatures],[compositions1, temperatures],[compositionsinf, temperatures]]

for datapair in data:
    print '>> -W1,black'
    compositions, temperatures=datapair
    for i, X in enumerate(compositions):
        print compositions[i], temperatures[i]
