# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman import tools
from burnman.mineral import Mineral
from burnman.chemicalpotentials import *
from burnman import constants

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from HP_convert import *
from fitting_functions import *
from scipy import optimize


B20=minerals.Fe_Si_O.FeSi_B20()
B2=minerals.Fe_Si_O.FeSi_B2()
FeSi_melt=minerals.Fe_Si_O.FeSi_liquid()
Si_A4=minerals.Fe_Si_O.Si_diamond_A4()
Si_fcc=minerals.Fe_Si_O.Si_fcc_A1()
Si_hcp=minerals.Fe_Si_O.Si_hcp_A3()
Si_bcc=minerals.Fe_Si_O.Si_bcc_A2()
Si_liq=minerals.Fe_Si_O.Si_liquid()
Fe_bcc=minerals.Myhill_calibration_iron.bcc_iron()
Fe_fcc=minerals.Myhill_calibration_iron.fcc_iron()
Fe_hcp=minerals.Myhill_calibration_iron.hcp_iron()
Fe_liq=minerals.Myhill_calibration_iron.liquid_iron_HP()

P = 1.e5
T = 1683.
Fe_liq.set_state(P, T) 
Si_liq.set_state(P, T) 
print (Fe_liq.alpha*Fe_liq.V + Si_liq.alpha*Si_liq.V)/(Fe_liq.V + Si_liq.V)
print (Fe_liq.V + Si_liq.V)/2.



pressures = np.linspace(1.e9, 150.e9, 101)
data = []
T = FeSi_melt.params['T_0']
for P in pressures:
        Fe_liq.set_state(P, T) 
        Si_liq.set_state(P, T)
        FeSi_V = (Fe_liq.V + Si_liq.V)/2.
        data.append([P, T, FeSi_V])

guesses = [FeSi_melt.params['V_0'], 
           FeSi_melt.params['K_0'], 
           FeSi_melt.params['Kprime_0']]

P_FeSi, T_FeSi, V_FeSi = zip(*data)
PT_FeSi = [P_FeSi, T_FeSi]

plt.plot(pressures, V_FeSi)

print optimize.curve_fit(fit_PV_data_full(FeSi_melt), P_FeSi, V_FeSi, guesses)

volumes = np.empty_like(pressures)
for i, P in enumerate(pressures):
    FeSi_melt.set_state(P, T) 
    volumes[i] = FeSi_melt.V

plt.plot(pressures, volumes)
plt.show()
