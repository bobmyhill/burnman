import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import Mineral
from burnman import minerals

from burnman.combinedmineral import CombinedMineral


pren = minerals.HP_2011_ds62.pren()
en = minerals.HP_2011_ds62.en()
hen= minerals.HP_2011_ds62.hen()
maj = CombinedMineral([minerals.HP_2011_ds62.maj()], [0.5], [0., 0., 0.]) # Mg4Si4O12/2.
'''
pren = minerals.HP_2011_ds62.pren()
en = minerals.SLB_2011.enstatite()
hen= minerals.SLB_2011.hp_clinoenstatite()
maj = CombinedMineral([minerals.SLB_2011.mg_majorite()], [0.5], [0., 0., 0.]) # Mg4Si4O12/2.
'''


# pren 1bar
P = 1.e5
T = 1700.
pren.set_state(P, T)
S_fus = 76.7 # 76.7 +/- 6.6 from Richet
dTdP_fus = 120.e-9

S = pren.S + S_fus
V = pren.V + S_fus*dTdP_fus
PTSV = [[P, T, S, V]]


def calc_liquid_SV(m1, m2, PT_m1, PT_m1m2, PT_m2):

    PT_m1 = np.array(PT_m1)
    PT_m1m2 = np.array(PT_m1m2)
    PT_m2 = np.array(PT_m2)
    
    P, T = PT_m1m2
    m1.set_state(P, T)
    m2.set_state(P, T)
    
    dTdP_m1L =  (PT_m1m2 - PT_m1)[1] / (PT_m1m2 - PT_m1)[0]
    dTdP_m2L =  (PT_m1m2 - PT_m2)[1] / (PT_m1m2 - PT_m2)[0]
    
    L_V =  ( dTdP_m2L*dTdP_m1L*(m2.S - m1.S) + dTdP_m2L*m1.V - dTdP_m1L*m2.V ) / (dTdP_m2L - dTdP_m1L)
    L_S = (L_V - m1.V) / dTdP_m1L + m1.S
    print((L_V - m1.V), dTdP_m1L,  dTdP_m2L,  m1.S)
    
    L_S_check = (L_V - m2.V) / dTdP_m2L + m2.S

    print (L_S, L_S_check)
    assert(np.abs(L_S - L_S_check) < 1.e-12)
    return [L_S, L_V]


# en-hen (Presnall and Gasparik)
P = 11.87e9
T = 2500.
S, V = calc_liquid_SV(en, hen, [10.92e9, 2474.], [P, T], [12.70e9, 2541.])
PTSV.append([P, T, S, V])
#S, V = calc_liquid_SV(en, hen, [10.92e9, 2464.], [P, T], [12.70e9, 2531.])
#PTSV.append([P, T, S, V])
S, V = calc_liquid_SV(en, hen, [10.92e9, 2484.], [P, T], [12.70e9, 2531.])
PTSV.append([P, T, S, V])
S, V = calc_liquid_SV(en, hen, [10.92e9, 2464.], [P, T], [12.70e9, 2551.])
PTSV.append([P, T, S, V])
S, V = calc_liquid_SV(en, hen, [10.92e9, 2484.], [P, T], [12.70e9, 2551.])
PTSV.append([P, T, S, V])

'''
# hen-maj (Presnall and Gasparik)
P = 16.45e9
T = 2641.
S, V = calc_liquid_SV(hen, maj, [15.84e9, 2634.], [P, T], [17.28e9, 2671.])
PTSV.append([P, T, S, V])


P = 16.45e9
T = 2641.
S, V = calc_liquid_SV(hen, maj, [15.84e9, 2660.], [P, T], [17.28e9, 2671.])
PTSV.append([P, T, S, V])
'''



P, T, S, V = np.array(PTSV).T

'''
PTSV2 = []
for (P, T, S, V) in PTSV:
    liq.set_state(P, T)
    PTSV2.append([P, T, liq.S, liq.V])
    
PTSV3 = []
for (P, T, S, V) in PTSV:
    liq.set_state(P, 1999.)
    PTSV3.append([P, T, liq.S, liq.V])


P2, T2, S2, V2 = np.array(PTSV2).T
P3, T3, S3, V3 = np.array(PTSV3).T
'''





plt.plot(P/1.e9, T)
plt.show()


plt.scatter(T, S)
#plt.scatter(T2, S2)
#plt.scatter(T3, S3)
plt.show()

plt.scatter(P/1.e9, V)
plt.ylim(4.e-5, 8.e-5)
plt.show()
#plt.scatter(P2/1.e9, V2)
#plt.scatter(P3/1.e9, V3)

'''
pressures = np.linspace(1.e5, 15.e9, 101)
temperatures = pressures*0.
volumes, bulk_moduli = SiO2_liq.evaluate(['V', 'K_T'], pressures, temperatures)
plt.plot(pressures/1.e9, volumes)
plt.show()
'''
