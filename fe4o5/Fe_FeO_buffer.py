import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.chemicalpotentials import *
from mineral_models_new import *
from equilibrium_functions import *

temperatures = np.linspace(1000+273.15, 1600+273.15, 7)
pressures = np.linspace(1.e5, 10.e9 + 1.e5, 21)
iron=fcc_iron()
wus=ferropericlase()
O2=burnman.minerals.HP_2011_fluids.O2()

f = open('fcc_iron_wus_PTlogfO2.dat', 'w')
for T in temperatures:
    O2.set_state(1.e5, T)
    iron_wus_fO2=eqm_curve_wus(iron, wus, pressures, T, O2)
    for i, P in enumerate(pressures):
        f.write(str(P/1.e9)+' '+str(T)+' '+str(iron_wus_fO2[i])+'\n')

f.write('\n')
f.close()
