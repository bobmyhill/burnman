import numpy as np
from scipy.optimize import fsolve, brentq, root
import matplotlib.pyplot as plt


import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    DKS_2013_liquids, \
    DKS_2013_solids, \
    SLB_2011, \
    HP_2011_ds62
from burnman import constants
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


q = SLB_2011.quartz()
coe = SLB_2011.coesite()
stv = SLB_2011.stishovite()
liq = DKS_2013_liquids.SiO2_liquid()

liq.params['P_0'] = 0.0e9
liq.property_modifiers = [['linear', {'delta_E': 1640860.+22000., 'delta_S': 16.5, 'delta_V': 0.e-7}]]
stv.property_modifiers = [['linear', {'delta_E': 4000., 'delta_S': 0., 'delta_V': 0.}]]


q.set_state(1.e5, 1696.)
liq.set_state(1.e5, 1696.)

print liq.gibbs - q.gibbs
print liq.V - q.V, '2.1'
print liq.S - q.S, '4.5'
print (liq.V - q.V)/(liq.S - q.S)*1.e9, '470'


coe.set_state(13.7e9, 3073.)
stv.set_state(13.7e9, 3073.)
liq.set_state(13.7e9, 3073.)

print ''
print liq.gibbs - stv.gibbs
print liq.V - coe.V, '0.'
print liq.S - stv.S, '35'
print (liq.V - stv.V)/(liq.S - stv.S)*1.e9, '6.'

fig1 = mpimg.imread('figures/hp_sio2_melting.png')
plt.imshow(fig1, extent=[0., 80., 1500., 5000.], aspect='auto')
fig2 = mpimg.imread('figures/sio2_melting.png')
plt.imshow(fig2, extent=[0., 15., 1673., 3273.], aspect='auto')

'''
pressures = np.linspace(1.e5, 5.e9, 21)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = burnman.tools.equilibrium_temperature([q, liq], [1.0, -1.0], P, 2000.)

plt.plot(pressures/1.e9, temperatures)
'''

pressures = np.linspace(5.e9, 13.e9, 21)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = burnman.tools.equilibrium_temperature([coe, liq], [1.0, -1.0], P, 2000.)

plt.plot(pressures/1.e9, temperatures)

pressures = np.linspace(13.e9, 100.e9, 21)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = burnman.tools.equilibrium_temperature([stv, liq], [1.0, -1.0], P, 2000.)

plt.plot(pressures/1.e9, temperatures)

plt.show()

'''
en = SLB_2011.enstatite()
pv = SLB_2011.mg_perovskite()
ppv = SLB_2011.mg_post_perovskite()
liq = DKS_2013_liquids.MgSiO3_liquid()

liq.params['P_0'] = 0.5e9
liq.property_modifiers = [['linear', {'delta_E': 2308841., 'delta_S': -20., 'delta_V': 0.}]]

en.set_state(0.54e9, 1612.5+273.15)
liq.set_state(0.54e9, 1612.5+273.15)

print liq.gibbs - en.gibbs/2.
print liq.V - en.V/2., '5.0'
print liq.S - en.S/2., '45.2'
print (liq.V - en.V/2.)/(liq.S - en.S/2.)*1.e9, '110'

PG = np.loadtxt(fname='data/Presnall_Gasparik_1990_en_melting.dat', unpack=True)
en_mask = [i for (i, v) in enumerate(PG[2]) if v==1]
plt.plot(PG[0][en_mask], PG[1][en_mask] + 273.15, marker='o', linestyle='None')
en_mask = [i for (i, v) in enumerate(PG[2]) if v==2]
plt.plot(PG[0][en_mask], PG[1][en_mask] + 273.15, marker='o', linestyle='None')
en_mask = [i for (i, v) in enumerate(PG[2]) if v==3]
plt.plot(PG[0][en_mask], PG[1][en_mask] + 273.15, marker='o', linestyle='None')


pressures = np.linspace(1.e5, 20.e9, 21)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = burnman.tools.equilibrium_temperature([en, liq], [1.0, -2.0], P, 2000.)

plt.plot(pressures/1.e9, temperatures)

pressures = np.linspace(20.e9, 140.e9, 61)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = burnman.tools.equilibrium_temperature([pv, liq], [1.0, -1.0], P, 2500.)

plt.plot(pressures/1.e9, temperatures)


pressures = np.linspace(100.e9, 140.e9, 21)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = burnman.tools.equilibrium_temperature([ppv, pv], [1.0, -1.0], P, 2500.)

plt.plot(pressures/1.e9, temperatures)

plt.show()
'''


'''
fo = SLB_2011.forsterite()
liq = DKS_2013_liquids.Mg2SiO4_liquid()

liq.params['P_0'] = -3.e9
liq.property_modifiers = [['linear', {'delta_E': 2999250.+24060., 'delta_S': -10., 'delta_V': 0.}]]

fo.set_state(1.e5, 1895. + 273.)
liq.set_state(1.e5, 1895. + 273.)

print liq.gibbs - fo.gibbs
print liq.V - fo.V, '3.4'
print liq.S - fo.S, '65.3 +/- 6.4'
print (liq.V - fo.V)/(liq.S - fo.S)*1.e9, '52'

PW = np.loadtxt(fname='data/Presnall_Walter_1993_fo_melting.dat', unpack=True)
fo_mask = [i for (i, v) in enumerate(PW[3]) if v==1]
plt.plot(PW[0][fo_mask], PW[1][fo_mask] + 273.15, marker='o', linestyle='None')

pressures = np.linspace(1.e5, 20.e9, 101)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = burnman.tools.equilibrium_temperature([fo, liq], [1.0, -1.0], P, 2000.)

plt.plot(pressures/1.e9, temperatures)
plt.show()

'''

