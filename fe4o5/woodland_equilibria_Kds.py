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

def equilibrium_composition(arg, mineral1, mineral2, XMg_mineral1, P, T):
    XMg_mineral2=arg[0]

    mineral2.set_composition([XMg_mineral2, 1.0-XMg_mineral2])
    mineral2.set_state(P,T)

    mineral1.set_composition([XMg_mineral1, 1.0-XMg_mineral1])
    mineral1.set_state(P,T)
    
    return (mineral2.partial_gibbs[0] + mineral1.partial_gibbs[1]) \
        - (mineral2.partial_gibbs[1] + mineral1.partial_gibbs[0]) 

f = open('equilibria.dat', 'w')


def fit_data(equilibria, H0):
    fm45.endmembers[0][0].params['H_0'] = H0
    fm45.enthalpy_interaction=[[0.e3]]
    burnman.SolidSolution.__init__(fm45)

    XMg2SiO4_diff = []
    for eqm in equilibria:
        ol_polymorph, P, T, XMg2Fe2O5_obs, XMg2SiO4_obs = eqm
        XMg2SiO4_calc=optimize.fsolve(equilibrium_composition, [0.9], args=(fm45, ol_polymorph, XMg2Fe2O5_obs, P, T))[0]
        XMg2SiO4_diff.append(XMg2SiO4_obs - XMg2SiO4_calc)

    return XMg2SiO4_diff

guesses = [fm45.endmembers[0][0].params['H_0']]
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


            XMg2SiO4_calc=optimize.fsolve(equilibrium_composition, [0.9], args=(fm45, ol_polymorph, XMg2Fe2O5_obs, P, T))[0]
            if '_' not in expt[0]: 
                print expt[0], ol_polymorph_name, P/1.e9, T-273.15, XMg2SiO4_calc, XMg2SiO4_obs, XMg2SiO4_calc-XMg2SiO4_obs
                f.write(expt[0]+' '+ol_polymorph_name+' '+str(P/1.e9)+' '+str(T-273.15)+' '+str(XMg2SiO4_calc)+' '+str(XMg2SiO4_obs)+' '+str(XMg2SiO4_calc-XMg2SiO4_obs)+'\n')


f.write('\n')
f.close()
print 'equilibria.dat (over)written'



#Kd = (XFe/XMg)Fe4O5 / (XFe/XMg)ol_polymorph)


Xs = np.linspace(0.01, 0.99, 101)
Kd_ol_fe4o5 = np.empty_like(Xs)
Kd_wad_fe4o5 = np.empty_like(Xs)
Kd_rw_fe4o5 = np.empty_like(Xs)
Kd_wad_ol = np.empty_like(Xs)
Kd_rw_ol = np.empty_like(Xs)

output_arrays = [[fm45, ol, Kd_ol_fe4o5],
                 [fm45, wad, Kd_wad_fe4o5],
                 [fm45, rw, Kd_rw_fe4o5]]

P = 10.e9
T = 1373.15

for (mineral1, mineral2, array) in output_arrays:
    for i, XMg_mineral1 in enumerate(Xs):
        XMg_mineral2=optimize.fsolve(equilibrium_composition, [0.9], args=(mineral1, mineral2, XMg_mineral1, P, T))[0]
        D_mineral1 = (1. - XMg_mineral1)/(XMg_mineral1)
        D_mineral2 = (1. - XMg_mineral2)/(XMg_mineral2)

        array[i] = D_mineral2/D_mineral1


plt.plot(Xs, Kd_ol_fe4o5)
plt.plot(Xs, Kd_wad_fe4o5)
plt.plot(Xs, Kd_rw_fe4o5)
#plt.plot(Xs, Kd_wad_ol)
#plt.plot(Xs, Kd_rw_ol)

filename = 'Kds_10GPa.dat'
f = open(filename, 'w')
for i, XMg_mineral1 in enumerate(Xs):
    f.write(str(XMg_mineral1)+' ')
    for array in output_arrays:
        f.write(str(array[2][i])+' ')
    f.write('\n')

f.write('\n')
f.close()
print filename, '(over)written'


P = 15.e9
T = 1373.15

for (mineral1, mineral2, array) in output_arrays:
    for i, XMg_mineral1 in enumerate(Xs):
        XMg_mineral2=optimize.fsolve(equilibrium_composition, [0.9], args=(mineral1, mineral2, XMg_mineral1, P, T))[0]
        D_mineral1 = (1. - XMg_mineral1)/(XMg_mineral1)
        D_mineral2 = (1. - XMg_mineral2)/(XMg_mineral2)

        array[i] = D_mineral2/D_mineral1

plt.plot(Xs, Kd_ol_fe4o5, '-')
plt.plot(Xs, Kd_wad_fe4o5, '-')
plt.plot(Xs, Kd_rw_fe4o5, '-')
#plt.plot(Xs, Kd_wad_ol, '-')
#plt.plot(Xs, Kd_rw_ol, '-')

filename = 'Kds_15GPa.dat'
f = open(filename, 'w')
for i, XMg_mineral1 in enumerate(Xs):
    f.write(str(XMg_mineral1)+' ')
    for array in output_arrays:
        f.write(str(array[2][i])+' ')
    f.write('\n')

f.write('\n')
f.close()
print filename, '(over)written'

plt.show()
