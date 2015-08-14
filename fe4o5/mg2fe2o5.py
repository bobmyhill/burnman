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
from equilibrium_functions import *
from slb_models_new import *


'''
stv_hp=minerals.HP_2011_ds62.stv()
stv_slb=minerals.SLB_2011.stishovite()


stv_hp.set_state(2.e10, 1673.)
stv_slb.set_state(2.e10, 1673.)
print stv_slb.H, stv_hp.H
print stv_slb.S, stv_hp.S
print stv_slb.V, stv_hp.V
print stv_slb.C_p, stv_hp.C_p
print ''


fo_hp=minerals.HP_2011_ds62.fo()
fo_slb=forsterite()


fa_hp=minerals.HP_2011_ds62.fa()
fa_slb=fayalite()

frw_hp=minerals.HP_2011_ds62.frw()
frw_slb=fe_ringwoodite()

frw_hp.set_state(1.e10, 1273.15)
frw_slb.set_state(1.e10, 1273.15)

print frw_slb.gibbs - frw_hp.gibbs


mrw_slb=mg_ringwoodite()

fo_hp.set_state(1.e5, 298.15)
fo_slb.set_state(1.e5, 298.15)
fa_hp.set_state(1.e5, 298.15)
fa_slb.set_state(1.e5, 298.15)

print fo_slb.gibbs - fo_hp.gibbs
print fa_slb.gibbs - fa_hp.gibbs


fo_hp.set_state(1.e10, 298.15)
fo_slb.set_state(1.e10, 298.15)
fa_hp.set_state(1.e10, 298.15)
fa_slb.set_state(1.e10, 298.15)

print fo_slb.gibbs - fo_hp.gibbs
print fa_slb.gibbs - fa_hp.gibbs

fo_hp.set_state(5.e9, 1273.)
fo_slb.set_state(5.e9, 1273.)
fa_hp.set_state(5.e9, 1273.)
fa_slb.set_state(5.e9, 1273.)

print fo_slb.gibbs - fo_hp.gibbs
print fa_slb.gibbs - fa_hp.gibbs
print ''
print 'SLB fa'
print fa_slb.H, -0.94269e6
print fa_slb.S, 384.75
print fa_slb.V, 4.5825
print fa_slb.C_p, 179.11

print ''
print 'SLB mrw'
mrw_slb.set_state(2.e10, 1673.)
print mrw_slb.H, -0.10195e7
print mrw_slb.S, 334.78
print mrw_slb.V, 3.6929
print mrw_slb.C_p, 177.01
print ''
'''

per=minerals.HP_2011_ds62.per()
hem=minerals.HP_2011_ds62.hem()
mft=minerals.HP_2011_ds62.mft()
mg2fe2o5 = Mg2Fe2O5()

assemblage = [per, hem, mg2fe2o5]
multiplicities = [2., 1., -1.]

temperatures = np.linspace(1473.15, 1873.15, 5)
pressures = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    pressures[i] = optimize.fsolve(eqm_pressure, [10.e9], args=(T, assemblage, multiplicities))[0]
    mft.set_state(pressures[i], T)
    print mft.gibbs + per.gibbs - mg2fe2o5.gibbs, 'should be positive for this line to be stable relative to mft + per' 
plt.plot(pressures/1.e9, temperatures - 273.15)
plt.xlabel('Pressure (GPa)')
plt.xlabel('Temperature (C)')
plt.show()
