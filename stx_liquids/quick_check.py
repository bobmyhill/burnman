import os, sys
sys.path.insert(1,os.path.abspath('..'))
import numpy as np
import matplotlib.pyplot as plt
import burnman
from burnman.minerals import HP_2011_ds62, SLB_2011, DKS_2013_liquids, DKS_2013_solids, RS_2014_liquids

P0 = 1.e5
P1 = 1.e5 + 1.
T0 = 2000.
T1 = 2000. + 1.
T2 = 2000. + 2.

per = HP_2011_ds62.per()
wus = HP_2011_ds62.fper()
fo = HP_2011_ds62.fo()
fa = HP_2011_ds62.fa()
en = HP_2011_ds62.en()
crst = HP_2011_ds62.crst()

ms = [per, wus, fo, fa, en, crst]
for m in ms:
    m.set_state(1.e5, 2000.)

Mg_X = np.array([0.0, 1./3., 0.5, 1.0])
Mg_mod = np.array([5.0e-6, 3.4e-6, 5.0e-6, 0.e-6])
Mg_Y = np.array([per.V + Mg_mod[0],
                 (fo.V + Mg_mod[1])/3.,
                 (en.V + Mg_mod[2]*2.)/4.,
                 crst.V + Mg_mod[3]])
Mg_mix = Mg_Y - ((1. - Mg_X)*(per.V + Mg_mod[0]) + (Mg_X)*(crst.V + Mg_mod[3]))

Fe_X = np.array([0.0, 1./3., 1.0])
Fe_mod = np.array([2.2e-6, 4.92e-6, 0.e-6])
Fe_Y = np.array([wus.V + Fe_mod[0],
                 (fa.V + Fe_mod[1])/3.,
                 crst.V + Fe_mod[2]])
Fe_mix = Fe_Y - ((1. - Fe_X)*(wus.V + Fe_mod[0]) + (Fe_X)*(crst.V + Fe_mod[2]))

plt.plot(Mg_X, Mg_mix, marker='o', linestyle=':', label='MgO-SiO2')
plt.plot(Fe_X, Fe_mix, marker='o', linestyle=':', label='FeO-SiO2')
plt.legend(loc='lower right')
plt.show()





fa.set_state(1.e5, 1490.)
print fa.V

Fe2SiO4_liq = RS_2014_liquids.Fe2SiO4_liquid()


per = HP_2011_ds62.per()
per = SLB_2011.forsterite()
temperatures = np.linspace(1000., 2000., 3001)
volumes = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    per.set_state(1.e5, T)
    print T
    volumes[i] = per.heat_capacity_p
plt.plot(temperatures, volumes)
plt.show()
    
per.set_state(P0, T0)
Cp0 = per.heat_capacity_p
a0 = per.alpha
V0 = per.V

per.set_state(P1, T0)
Cp1 = per.heat_capacity_p

per.set_state(P0, T1)
a1 = per.alpha
V1 = per.V


per.set_state(P0, T2)
V2 = per.V

d2VdT2 = ((V2-V1) - (V1-V0))
print -T0*d2VdT2

print (Cp1 - Cp0)/(P1 - P0)
print -T0*(a1*V1 - a0*V0)/(T1 - T0)


liq =  DKS_2013_liquids.Mg2SiO4_liquid()
liq.set_state(1.e5, 2174.)
print liq.S

for P in [1.e5, 100.e9, 200.e9]:
    temperatures = np.linspace(2000., 3500. + P/1.e9*10., 101)
    volumes = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        liq.set_state(P, T)
        volumes[i] = liq.heat_capacity_p
    plt.plot(temperatures, volumes)

plt.show()

per = SLB_2011.periclase()
per.params['q_0'] = 0.15
per.params['grueneisen_0'] = 1.6 # 1.4 for Cp, 1.6 for volume

phases = [[DKS_2013_solids.periclase(), DKS_2013_liquids.MgO_liquid(), 1., 1.e5, 3098.],
          [per, DKS_2013_liquids.MgO_liquid(), 1., 1.e5, 3098.],
          [HP_2011_ds62.fo(), DKS_2013_liquids.Mg2SiO4_liquid(), 1., 1.e5, 2174.],
          [HP_2011_ds62.pren(), DKS_2013_liquids.MgSiO3_liquid(), 2., 1.e5, 1850.],
          [HP_2011_ds62.en(), DKS_2013_liquids.MgSiO3_liquid(), 2., 1.e5, 1838.],
          [HP_2011_ds62.en(), HP_2011_ds62.pren(), 1., 1.e5, 1838.],
          [HP_2011_ds62.q(), DKS_2013_liquids.SiO2_liquid(), 1., 1.e5, 1696.],
          [HP_2011_ds62.crst(), DKS_2013_liquids.SiO2_liquid(), 1., 1.e5, 1999.],
          [HP_2011_ds62.fa(), RS_2014_liquids.Fe2SiO4_liquid(), 1., 1.e5, 1490.]]


for (sol, liq, f, P, T) in phases:
    sol.set_state(P, T)
    liq.set_state(P, T)
    print liq.S, sol.S, liq.V - sol.V/f, liq.S - sol.S/f, (liq.V - sol.V/f)/(liq.S - sol.S/f)*1.e9

    
