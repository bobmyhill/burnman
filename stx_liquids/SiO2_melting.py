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

# Tweaks to stishovite to get volumes at ambient pressure correct
stv.params['a_0'] = 1.4e-5
stv.params['V_0'] = 1.4006e-5
stv.params['H_0'] = stv.params['H_0'] + 2000.
stv.params['S_0'] = stv.params['S_0'] + 2.

P = 1.e5
T = 298.15
stv.set_state(P, T)
stv_SLB.set_state(P, T)


def find_temperature(temperature, pressure, phase1, phase2, factor):
    phase1.set_state(pressure, temperature[0])
    phase2.set_state(pressure, temperature[0])
    return phase1.gibbs*factor - phase2.gibbs



P = 13.7e9
print 'coe<->stv @ 13.7 GPa:', fsolve(find_temperature, [2000.], args=(P, stv, coe, 1.))[0]
T = 2800.+273.15
stv.set_state(P, T)
stv_DKS.set_state(P, T)
coe.set_state(P, T)
SiO2_liq.set_state(P, T)
SiO2_liq_HP.set_state(P, T)
print coe.V, SiO2_liq.V, 'volume difference should be very small'

print 'Smelt:'
print SiO2_liq_HP.S - stv.S, '(HP)'
print SiO2_liq.S - stv_DKS.S, '(DKS)'


T=fsolve(find_temperature, [2000.], args=(P, stv_DKS, SiO2_liq, 1.))[0]

P = 13.7e9
print 'Tmelt @ 13.7 GPa:'
print fsolve(find_temperature, [2000.], args=(P, stv_DKS, SiO2_liq, 1.))[0]
print fsolve(find_temperature, [2000.], args=(P, stv, SiO2_liq_HP, 1.))[0]

P = 100.e9
print 'Tmelt @ 100 GPa:'
print fsolve(find_temperature, [2000.], args=(P, stv_DKS, SiO2_liq, 1.))[0]
print fsolve(find_temperature, [2000.], args=(P, stv, SiO2_liq_HP, 1.))[0]

'''
SiO2_liq.set_state(P, T)
SiO2_liq_HP.set_state(P, T)
print SiO2_liq.V, SiO2_liq_HP.V

stv.set_state(P, T)
stv_SLB.set_state(P, T)
stv_DKS.set_state(P, T)

print stv_SLB.S, stv.S, stv_DKS.S
print stv_SLB.V, stv.V, stv_DKS.V
print stv_SLB.K_T/1.e9, stv.K_T/1.e9
'''
