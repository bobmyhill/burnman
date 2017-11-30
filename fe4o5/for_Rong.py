import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

# Benchmarks for the solid solution class
import burnman
from burnman import minerals
from burnman.chemicalpotentials import *


iron=minerals.HP_2011_ds62.iron()
fper=minerals.HP_2011_ds62.fper()
mt=minerals.HP_2011_ds62.mt()
hem=minerals.HP_2011_ds62.hem()
#wus=ferropericlase()

P = 25.e9
T = 1873.
hem.set_state(P, T)

print(hem.gibbs/1000.)
print(hem.H, hem.S)
print(hem.heat_capacity_p)

#burnman.tools.check_eos_consistency(hem, P, T,
#                                    tol=1.e-6, verbose=True)

print(iron.property_modifiers)


def Cp(temperatures):
    Cps = []
    for T in temperatures:
        hem.set_state(1.e5, T)
        Cps.append(hem.heat_capacity_p)
    return np.array(Cps)

temperatures = np.linspace(298.15, 1873.15, 10001)
Cps = Cp(temperatures)


from scipy.integrate import simps

I1 = simps(Cps, temperatures)
print(I1)

# 250 kbar
# 1600 C
# -172.97 for FeO 
# -486.6 for Fe2O3
