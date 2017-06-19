import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

# Benchmarks for the solid solution class
import burnman
from burnman import minerals
from burnman.chemicalpotentials import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scipy.optimize as optimize

from mineral_models_new import *
from slb_models_new import *
from equilibrium_functions import *


# Required constraints
# 1. bulk composition (use composition of ol polymorph)
# 2. Mg2Fe2O5 + Fe2SiO4 <-> Fe4O5 + Mg2SiO4

mg2fe2o5 = Mg2Fe2O5()
fm45=MgFeFe2O5()


# HP solid solutions
ol=olivine()
wad=wadsleyite()
rw=ringwoodite()

'''
# SLB solid solutions
ol=mg_fe_olivine()
wad=mg_fe_wadsleyite()
rw=mg_fe_ringwoodite()
'''

data=[]
with open('phasePTX.dat','r') as f:
    for expt in f:
        data.append([var for var in expt.split()])

# Process data
equilibria = []
zeros = []
for expt in data:
    if len(expt) > 1:
        ol_polymorph_name=expt[2]
        if ol_polymorph_name == 'ol' or ol_polymorph_name == 'wad' or ol_polymorph_name == 'rw':
            if ol_polymorph_name == 'ol':
                ol_polymorph=ol
                
            if ol_polymorph_name == 'wad':
                ol_polymorph=wad
                    
            if ol_polymorph_name == 'rw':
                ol_polymorph=rw

            # ol polymorph
            # converting P from kbar to Pa
            # converting T from C to K
            # 2 atoms Mg in Mg2Fe2O5
            # XMg (ol polymorph) = p(fo)
            if ol_polymorph == ol:
                equilibria.append([ol_polymorph, float(expt[3])*1.e8, float(expt[4])+273.15, float(expt[5])/2., float(expt[9])])
                zeros.append(0.)

def equilibrium_composition(arg, ol_polymorph, XMg_Fe4O5, P, T):
    XMg_ol_polymorph=arg[0]

    ol_polymorph.set_composition([XMg_ol_polymorph, 1.0-XMg_ol_polymorph])
    ol_polymorph.set_state(P,T)

    fm45.set_composition([XMg_Fe4O5, 1.0-XMg_Fe4O5])
    fm45.set_state(P,T)
    
    return (ol_polymorph.partial_gibbs[0] + fm45.partial_gibbs[1]) \
        - (ol_polymorph.partial_gibbs[1] + fm45.partial_gibbs[0]) 

f = open('equilibria.dat', 'w')


def fit_data(equilibria, Hex):
    fm45.enthalpy_interaction=[[Hex]]
    burnman.SolidSolution.__init__(fm45)

    XMg2SiO4_diff = []
    for eqm in equilibria:
        ol_polymorph, P, T, XMg2Fe2O5_obs, XMg2SiO4_obs = eqm
        XMg2SiO4_calc=optimize.fsolve(equilibrium_composition, [0.9], args=(ol_polymorph, XMg2Fe2O5_obs, P, T))[0]
        XMg2SiO4_diff.append(XMg2SiO4_obs - XMg2SiO4_calc)

    return XMg2SiO4_diff

guesses = [0.]
popt, pcov = optimize.curve_fit(fit_data, equilibria, zeros, guesses)

for i, p in enumerate(popt):
    print p, '+/-', np.sqrt(pcov[i][i])


print ''
for expt in data:
    if len(expt) > 1:
        ol_polymorph_name=expt[2]

        if ol_polymorph_name == 'ol':
            ol_polymorph=ol

        if ol_polymorph_name == 'wad':
            ol_polymorph=wad

        if ol_polymorph_name == 'rw':
            ol_polymorph=rw

        if ol_polymorph_name == 'ol' or ol_polymorph_name == 'wad' or ol_polymorph_name == 'rw':
            P=float(expt[3])*1.e8 # converting from kbar to Pa
            T=float(expt[4])+273.15 # converting from C to K
            XMg2Fe2O5_obs=float(expt[5])/2. # 2 atoms Mg in Mg2Fe2O5
            XMg2SiO4_obs=float(expt[9])


            XMg2SiO4_calc=optimize.fsolve(equilibrium_composition, [0.9], args=(ol_polymorph, XMg2Fe2O5_obs, P, T))[0]
            if '_' not in expt[0]: 
                print expt[0], ol_polymorph_name, P/1.e9, T-273.15, XMg2SiO4_calc, XMg2SiO4_obs, XMg2SiO4_calc-XMg2SiO4_obs
                f.write(expt[0]+' '+ol_polymorph_name+' '+str(P/1.e9)+' '+str(T-273.15)+' '+str(XMg2SiO4_calc)+' '+str(XMg2SiO4_obs)+' '+str(XMg2SiO4_calc-XMg2SiO4_obs)+'\n')


f.write('\n')
f.close()
print 'equilibria.dat (over)written'
