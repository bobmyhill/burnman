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
K1 = lambda T:1 
G = lambda T: 0. - 75.*(T-1420.)
K = lambda T: np.exp(-(G(T))/(R*T))
Wsh = lambda T: 00000.
Whs = lambda T: 00000.

pressure = 13.e9
anhydrous_phase=SLB_2011.forsterite()
liquid=MgO_SiO2_liquid()
liquid.set_composition([2./3., 1./3.])

Tmelt = fsolve(delta_gibbs, 2000., args=(pressure, anhydrous_phase, liquid, 1./3., 1.))[0]
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



fn0=0.
temperatures=np.linspace(600., Tmelt, 101)
compositions0=np.empty_like(temperatures)
compositions1=np.empty_like(temperatures)
compositionsinf=np.empty_like(temperatures)
compositions_fo=np.empty_like(temperatures)

for i, T in enumerate(temperatures):
    compositions0[i]=fsolve(solve_composition, 0.001, args=(T, pressure, r, K0, fn0, fn0, anhydrous_phase, liquid, 1./3., 1.))
    compositions1[i]=fsolve(solve_composition, 0.001, args=(T, pressure, r, K1, fn0, fn0, anhydrous_phase, liquid, 1./3., 1.))
    compositionsinf[i]=fsolve(solve_composition, 0.001, args=(T, pressure, r, Kinf, fn0, fn0, anhydrous_phase, liquid, 1./3., 1.))
    compositions_fo[i]=fsolve(solve_composition, 0.001, args=(T, pressure, r, K, Wsh(T), Whs(T), anhydrous_phase, liquid, 1./3., 1.))

temperatures_eqm=np.linspace(1273.15,2573.15, 27) 
for i, T in enumerate(temperatures_eqm):
    print T-273.15, fsolve(solve_composition, 0.001, args=(T, pressure, r, K, Wsh(T), Whs(T), anhydrous_phase, liquid, 1./3., 1.))

    
plt.plot( compositions_fo, temperatures, linewidth=1, label='fo')
plt.plot( compositions0, temperatures, linewidth=1, label='K=0')
plt.plot( compositions1, temperatures, linewidth=1, label='K=1')
plt.plot( compositionsinf, temperatures, linewidth=1, label='K=inf')


###################

forsterite = []
enstatite=[]
chondrodite=[]
superliquidus=[]
for line in open('data/13GPa_fo-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[3] == 'f' or content[3] == 'sf' or content[3] == 'f_davide':
            forsterite.append([float(content[0])+273.15, (100. - float(content[1])*3.)/100.])
        if content[3] == 'e' or content[3] == 'se' or content[3] == 'e_davide':
            enstatite.append([float(content[0])+273.15, (100. - float(content[1])*3.)/100.])
        if content[3] == 'c':
            chondrodite.append([float(content[0])+273.15, (100. - float(content[1])*3.)/100.])
        if content[3] == 'l' or content[3] == 'l_davide':
            superliquidus.append([float(content[0])+273.15,(100. - float(content[1])*3.)/100.])

add_T=0.
forsterite=np.array(zip(*forsterite))
enstatite=np.array(zip(*enstatite))
chondrodite=np.array(zip(*chondrodite))
superliquidus=np.array(zip(*superliquidus))
plt.plot( forsterite[1], forsterite[0]+add_T, marker='.', linestyle='none', label='fo+liquid')
plt.plot( enstatite[1], enstatite[0]+add_T, marker='.', linestyle='none', label='en+liquid')
plt.plot( chondrodite[1], chondrodite[0]+add_T+add_T, marker='.', linestyle='none', label='chond+liquid')
plt.plot( superliquidus[1], superliquidus[0]+add_T, marker='.', linestyle='none', label='superliquidus')

plt.ylim(1000.,3000.)
plt.xlim(0.,1.)
plt.ylabel("Temperature (K)")
plt.xlabel("X")
plt.legend(loc='upper right')
plt.show()

####################
# a-X relationships (1 cation basis)
activities = np.empty_like(compositions_fo)
for i, composition in enumerate(compositions_fo):
    temperature = temperatures[i]
    activities[i] =  np.exp( delta_gibbs([temperature], pressure, anhydrous_phase, liquid, 1./3., 1.) / (constants.gas_constant*temperature))

   
plt.plot(compositions_fo, activities)
plt.title('Forsterite')
plt.xlim(0., 1.)
plt.ylim(0., 1.)
plt.show()
####################


model_filename='forsterite_models.xT'
data=[['-W1,grey,-', compositions0, temperatures],
      ['-W1,grey', compositionsinf, temperatures],
      ['-W1,black', compositions_fo, temperatures]] 

f = open(model_filename,'w')
for datapair in data:
    linetype, compositions, temperatures=datapair
    f.write('>> '+str(linetype)+' \n')
    for i, X in enumerate(compositions):
        f.write( str(compositions[i])+' '+str(temperatures[i]-273.15)+'\n' ) # output in C
f.close()

