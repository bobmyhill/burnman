import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    DKS_2013_solids, \
    DKS_2013_liquids, \
    DKS_2013_liquids_w_HP, \
    HP_2011_ds62, \
    SLB_2011

from burnman import constants
import numpy as np
from scipy.optimize import fsolve, curve_fit
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


SiO2_liq=DKS_2013_liquids.SiO2_liquid()
SiO2_liq_HP=DKS_2013_liquids_w_HP.SiO2_liquid()
stv_DKS = DKS_2013_solids.stishovite()
stv = HP_2011_ds62.stv()
stv_SLB = SLB_2011.stishovite()
coe = HP_2011_ds62.coe()

stv.params['Kprime_0'] = stv.params['Kprime_0'] - 1

def find_temperature(temperature, pressure, phase1, phase2, factor):
    phase1.set_state(pressure, temperature[0])
    phase2.set_state(pressure, temperature[0])
    return phase1.gibbs*factor - phase2.gibbs

P = 13.8e9
T=fsolve(find_temperature, [2000.], args=(P, stv_DKS, SiO2_liq, 1.))[0]

print fsolve(find_temperature, [2000.], args=(P, stv_DKS, SiO2_liq, 1.))[0]
print fsolve(find_temperature, [2000.], args=(P, stv, SiO2_liq_HP, 1.))[0]

'''
P = 100.e9
print fsolve(find_temperature, [2000.], args=(P, stv_DKS, SiO2_liq, 1.))[0]
print fsolve(find_temperature, [2000.], args=(P, stv, SiO2_liq_HP, 1.))[0]
'''

SiO2_liq.set_state(P, T)
SiO2_liq_HP.set_state(P, T)
print SiO2_liq.V, SiO2_liq_HP.V

stv.set_state(P, T)
stv_SLB.set_state(P, T)
stv_DKS.set_state(P, T)

print stv_SLB.H, stv.H, stv_DKS.H
print stv_SLB.S, stv.S, stv_DKS.S
print stv_SLB.V, stv.V, stv_DKS.V
print stv_SLB.K_T, stv.K_T
