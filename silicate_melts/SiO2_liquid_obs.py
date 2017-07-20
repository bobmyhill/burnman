import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import Mineral
from burnman import minerals

crst = minerals.HP_2011_ds62.crst()
qtz = minerals.HP_2011_ds62.q()
coe = minerals.HP_2011_ds62.coe()
stv = minerals.HP_2011_ds62.stv()
#stv = minerals.SLB_2011.stishovite()
liq = minerals.DKS_2013_liquids.SiO2_liquid()
MgO_liq = minerals.DKS_2013_liquids.MgO_liquid()
Mg5SiO7_liq = minerals.DKS_2013_liquids.Mg5SiO7_liquid()
Mg2SiO4_liq = minerals.DKS_2013_liquids.Mg2SiO4_liquid()
MgSiO3_liq = minerals.DKS_2013_liquids.MgSiO3_liquid()
MgSi2O5_liq = minerals.DKS_2013_liquids.MgSi2O5_liquid()
MgSi3O7_liq = minerals.DKS_2013_liquids.MgSi3O7_liquid()
MgSi5O11_liq = minerals.DKS_2013_liquids.MgSi5O11_liquid()
Mg3Si2O7_liq = minerals.DKS_2013_liquids.Mg3Si2O7_liquid()

SiO2_liq = burnman.Mineral(params={'equation_of_state': 'rkprime',
                                   'V_0': 27.3e-6,
                                   'K_0': 8.e9,
                                   'Kprime_0': 10.0,
                                   'Kprime_inf': 3.0, # matches both the value for the core, and the values for MgO-MgSiO3 in deKoker and Stixrude
                                   'molar_mass': 0.06008,
                                   'G_0': 2.e9,
                                   'Gprime_inf': 1.})

pressures = np.linspace(1.e5, 200.e9, 101)
temperatures = pressures*0.
volumes, bulk_moduli = SiO2_liq.evaluate(['V', 'K_T'], pressures, temperatures)

plt.plot(pressures/1.e9, np.gradient(bulk_moduli, pressures))
plt.show()


# Let's have a look at K'_infty

for pressures in [np.linspace(1.e5, 40.e9, 101)]:
    for T in [2000.]:
        temperatures = pressures*0. + T
        for l in [MgO_liq, Mg5SiO7_liq, Mg2SiO4_liq, Mg3Si2O7_liq, MgSiO3_liq, MgSi2O5_liq, MgSi3O7_liq, MgSi5O11_liq, liq]:
            V, K_T = l.evaluate(['V', 'K_T'], pressures, temperatures)
            dKdP = np.gradient(K_T, pressures, edge_order=2)
            
            #plt.plot(np.log((V[0]/V)[1:-1]), np.log(1./dKdP[1:-1]), label='{0} at {1} K'.format(l.name, T))
            #plt.scatter(np.power(20., (pressures/K_T)[1:-1]), 1./dKdP[1:-1], label='{0} at {1} K'.format(l.name, T))
            plt.plot((pressures/K_T), 1./dKdP, label='{0} at {1} K'.format(l.name, T))
            plt.scatter((pressures/K_T), 1./dKdP)
#plt.xlim(0., 1.)
#plt.plot([0., 3./5.], [0., 3./5.])
plt.plot([0., 0.4], [0., 0.4])
plt.xlim(0.,)
plt.legend(loc='lower right')
plt.show()



# qtz 1bar
P = 1.e5
T = 1700.
qtz.set_state(P, T)
S = qtz.S + 5.53
V = 27.3e-6
PTSV = [[P, T, S, V]]

# crst 1bar
P = 1.e5
T = 1999.
crst.set_state(P, T)
S = crst.S + 4.46
V = 27.3e-6

PTSV.append([P, T, S, V])




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


# crst-q (Ghiorso et al., 2004)
P = 0.426e9
T = 1951.
S, V = calc_liquid_SV(crst, qtz, [0.376e9, 1960.], [P, T], [0.483e9, 1976.])
PTSV.append([P, T, S, V])


# qtz-coe (Zhang et al., 1993)
P = 4.47e9
T = 2716.
S, V = calc_liquid_SV(qtz, coe, [4.19e9, 2710.], [P, T], [4.85e9, 2752.])
PTSV.append([P, T, S, V])

'''
# qtz-coe (Kanzaki et al., 1990)
P = 4.47e9
T = 2697.
S, V = calc_liquid_SV(qtz, coe, [3.95e9, 2663.], [P, T], [5.e9, 2753.])
PTSV.append([P, T, S, V])
'''

# coe-stv
P = 13.7e9
T = 3073.
S, V = calc_liquid_SV(coe, stv, [13.0e9, 3073. + 3.], [P, T], [15.9e9, 3410.]) # from Shen and Lazor
S, V = calc_liquid_SV(coe, stv, [13.0e9, 3073. + 3.], [P, T], [15.9e9, 3270.]) # modified
PTSV.append([P, T, S, V])



PTSV2 = []
for (P, T, S, V) in PTSV:
    liq.set_state(P, T)
    PTSV2.append([P, T, liq.S, liq.V])
    
PTSV3 = []
for (P, T, S, V) in PTSV:
    liq.set_state(P, 1999.)
    PTSV3.append([P, T, liq.S, liq.V])





P, T, S, V = np.array(PTSV).T
P2, T2, S2, V2 = np.array(PTSV2).T
P3, T3, S3, V3 = np.array(PTSV3).T

plt.plot(P/1.e9, T)
plt.show()


plt.plot(T, S)
plt.scatter(T2, S2)
plt.scatter(T3, S3)
plt.show()
plt.plot(P/1.e9, V)
plt.scatter(P2/1.e9, V2)
plt.scatter(P3/1.e9, V3)


pressures = np.linspace(1.e5, 15.e9, 101)
temperatures = pressures*0.
volumes, bulk_moduli = SiO2_liq.evaluate(['V', 'K_T'], pressures, temperatures)
plt.plot(pressures/1.e9, volumes)
plt.show()

