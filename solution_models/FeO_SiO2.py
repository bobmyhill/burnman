import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

from scipy.optimize import fsolve, curve_fit
import burnman
from burnman import minerals

from liq_wuestite_HS2014 import FeO_liquid


wus_liq = FeO_liquid()
fa_liq = minerals.RS_2014_liquids.Fe2SiO4_liquid()
stv_liq = minerals.DKS_2013_liquids_tweaked.SiO2_liquid()

pressures = np.linspace(1.e9, 300.e9)
V_diff = np.empty_like(pressures)

for T in [2000., 3000., 4000., 5000.]:
    for i, P in enumerate(pressures):
        for liq in [wus_liq, fa_liq, stv_liq]:
            liq.set_state(P, T)
            
        V_diff[i] = 2.*wus_liq.V + 1.*stv_liq.V - fa_liq.V

    plt.plot(pressures, V_diff, label=str(T)+' K')

plt.legend(loc='upper right')
plt.show()



pressures = np.linspace(1.e9, 300.e9)
V_diff = np.empty_like(pressures)
V_diff2 = np.empty_like(pressures)

'''
per_liq = minerals.DKS_2013_liquids_tweaked.MgO_liquid()
fo_liq = minerals.DKS_2013_liquids_tweaked.Mg2SiO4_liquid()
pv_liq = minerals.DKS_2013_liquids_tweaked.MgSiO3_liquid()
stv_liq = minerals.DKS_2013_liquids_tweaked.SiO2_liquid()


per_liq = minerals.DKS_2013_liquids.MgO_liquid()
fo_liq = minerals.DKS_2013_liquids.Mg2SiO4_liquid()
pv_liq = minerals.DKS_2013_liquids.MgSiO3_liquid()
stv_liq = minerals.DKS_2013_liquids.SiO2_liquid()


for T in [2000., 3000., 4000., 5000.]:
    for i, P in enumerate(pressures):
        for liq in [per_liq, fo_liq, pv_liq, stv_liq]:
            liq.set_state(P, T)
            
        V_diff[i] = 2.*per_liq.V + 1.*stv_liq.V - fo_liq.V
        #V_diff[i] = per_liq.V + pv_liq.V - fo_liq.V
        V_diff2[i] = per_liq.V + stv_liq.V - pv_liq.V

    plt.plot(pressures, V_diff, label=str(T)+' K')
    #plt.plot(pressures, V_diff2, linestyle='--', label=str(T)+' K')


T = 2973.15
P = 13.e9

coe = minerals.SLB_2011.coesite()
for m in [per_liq, fo_liq, coe]:
    m.set_state(P, T)
            
Vdiff = 2.*per_liq.V + coe.V - fo_liq.V
plt.plot(P, Vdiff, marker='o')
    
plt.legend(loc='upper right')
plt.show()
'''
