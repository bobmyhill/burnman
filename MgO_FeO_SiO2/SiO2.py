import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import HP_2011_ds62, Myhill_silicate_liquid
import numpy as np
from scipy import optimize, integrate
import matplotlib.pyplot as plt

def invariant(data):
    P, T = data
    return [stv.calcgibbs(P, T) -  coe.calcgibbs(P, T),
            SiO2_liq.calcgibbs(P, T) -  coe.calcgibbs(P, T)]
        
def eqm_pressure(minerals, multiplicities):
    def eqm(P, T):
        gibbs = 0.
        for i, mineral in enumerate(minerals):
            gibbs = gibbs + mineral.calcgibbs(P[0], T)*multiplicities[i]
        return gibbs
    return eqm

def eqm_temperature(minerals, multiplicities):
    def eqm(T, P):
        gibbs = 0.
        for i, mineral in enumerate(minerals):
            gibbs = gibbs + mineral.calcgibbs(P, T[0])*multiplicities[i]
        return gibbs
    return eqm

q = HP_2011_ds62.q()
coe = HP_2011_ds62.coe()
stv = Myhill_silicate_liquid.stv()
SiO2_liq = Myhill_silicate_liquid.SiO2_liquid()

SiO2_liq.set_state(1.e5, 1996.)
coe.set_state(1.e5, 1996.)

coe.set_state(13.7e9, 3073.15)
print coe.V

stv.set_state(13.7e9, 3073.15)
print stv.S + 40.

print stv.gibbs + 3073.15*(stv.S + 40.)

print SiO2_liq.gibbs, coe.gibbs
print optimize.fsolve(eqm_temperature([coe, SiO2_liq], [1., -1.]), [2000.], args=(1.e5))
print optimize.fsolve(eqm_temperature([coe, SiO2_liq], [1., -1.]), [2000.], args=(2.e9))
Pinv, Tinv = optimize.fsolve(invariant, [13.e9, 2000.])
coe.set_state(Pinv, Tinv)
stv.set_state(Pinv, Tinv)
SiO2_liq.set_state(Pinv, Tinv)
print Pinv/1.e9, Tinv
print 'volumes should be equal at invariant', coe.V, SiO2_liq.V
print 'entropy of melting should be ca. 40 J/K/mol', SiO2_liq.S - stv.S


pressures_q = np.linspace(1.e5, 5.e9, 20)
temperatures_q = np.empty_like(pressures_q)
for i, P in enumerate(pressures_q):
    temperatures_q[i] = optimize.fsolve(eqm_temperature([q, SiO2_liq], [1., -1.]), [2000.], args=(P))

pressures_coe = np.linspace(4.e9, 14.e9, 20)
temperatures_coe = np.empty_like(pressures_coe)
for i, P in enumerate(pressures_coe):
    temperatures_coe[i] = optimize.fsolve(eqm_temperature([coe, SiO2_liq], [1., -1.]), [2000.], args=(P))

pressures_coe_stv = np.linspace(6.e9, 14.e9, 20)
temperatures_coe_stv = np.empty_like(pressures_coe_stv)
for i, P in enumerate(pressures_coe_stv):
    temperatures_coe_stv[i] = optimize.fsolve(eqm_temperature([coe, stv], [1., -1.]), [2000.], args=(P))

pressures_stv = np.linspace(12.e9, 250.e9, 200)
temperatures_stv = np.empty_like(pressures_stv)
for i, P in enumerate(pressures_stv):
    temperatures_stv[i] = optimize.fsolve(eqm_temperature([stv, SiO2_liq], [1., -1.]), [3000.], args=(P))

plt.plot(pressures_coe_stv/1.e9, temperatures_coe_stv-273.15)
plt.plot(pressures_coe/1.e9, temperatures_coe-273.15)
plt.plot(pressures_q/1.e9, temperatures_q-273.15)
plt.plot(pressures_stv/1.e9, temperatures_stv-273.15)
#plt.xlim(0., 20.)
#plt.ylim(1400., 3500.)
plt.ylabel('Temperature (C)')
plt.show()

SiO2_liq.set_state(1.e5, 1673.)
print SiO2_liq.V, 'should be 27.3 cm^3/mol according to Bockris et al., 1956'
SiO2_liq.set_state(1.e5, 3000.)
print SiO2_liq.V, 'should be 27.8 cm^3/mol according to de Koker + stx'
print SiO2_liq.K_T/1.e9, 'should be 6.2 GPa according to de Koker + stx'
print SiO2_liq.S, 'should be 205.5 J/K/mol according to de Koker + stx'

dP = 100.
K0 = SiO2_liq.K_T
SiO2_liq.set_state(1.e5+dP, 3000.)
K1 = SiO2_liq.K_T
print (K1 - K0)/dP, 'should be 14.94 according to de Koker + stx'


SiO2_liq.set_state(14.e9, 3120.)
stv.set_state(14.e9, 3120.)
coe.set_state(14.e9, 3120.)
print coe.S, stv.S, SiO2_liq.S, SiO2_liq.S-stv.S
