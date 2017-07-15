import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

from scipy.optimize import fsolve, curve_fit
import burnman
from burnman import minerals

from liq_wuestite_AA1994 import liq_FeO


wus_liq = liq_FeO()
per_liq = minerals.DKS_2013_liquids.MgO_liquid()

P = 1.e5
for T in [1650., 6000., 8000., 10000.]:
    per_liq.set_state(P, T)
    wus_liq.set_state(P, T)
    print P/1.e9, T, per_liq.V, wus_liq.V
exit()
    
pressures = np.linspace(10.e9, 200.e9, 91)

f_FeO = 0.0

V = np.empty_like(pressures)

fig1 = mpimg.imread('../FeO/figures/Mg75Fe25O_liquid_volume_Holmstrom_Stixrude_2016.png')
plt.imshow(fig1, extent=[10., 200., 6., 18.], aspect='auto')

for T in [6000., 8000., 10000.]:
    for i, P in enumerate(pressures):
        for liq in [wus_liq, per_liq]:
            liq.set_state(P, T)
            
        V[i] = ((1. - f_FeO)*per_liq.V + f_FeO*wus_liq.V)/burnman.constants.Avogadro*1.e30/2.

    plt.plot(pressures/1.e9, V, label=str(T)+' K')

plt.legend(loc='upper right')
plt.xlabel('P (GPa)')
plt.ylabel('V (Angstroms^3 per atom)')
plt.show()

