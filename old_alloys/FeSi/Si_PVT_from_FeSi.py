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
#Si_liq=minerals.Fe_Si_O.Si_liquid()
Fe_bcc=minerals.Myhill_calibration_iron.bcc_iron()
Fe_fcc=minerals.Myhill_calibration_iron.fcc_iron()
Fe_hcp=minerals.Myhill_calibration_iron.hcp_iron()
Fe_liq=minerals.Myhill_calibration_iron.liquid_iron()
Si_liq=minerals.Myhill_calibration_iron.liquid_iron()

Pref = 1.e5
Tref = 1873.15
Si_liq.params['P_0'] = Pref
Si_liq.params['T_0'] = Tref

pressures = np.linspace(1.e9, 150.e9, 11)
temperatures = np.linspace(1000., 3000., 11)
data = []
for P in pressures:
    for T in temperatures:
        Fe_liq.set_state(P, T) 
        FeSi_melt.set_state(P, T)
        Si_V = 2.*FeSi_melt.V - Fe_liq.V
        print Fe_liq.V, FeSi_melt.V, Si_V
        data.append([P, T, Si_V])

guesses = [Si_liq.params['V_0'], 
           Si_liq.params['K_0'], 
           Si_liq.params['Kprime_0'], 
           Si_liq.params['a_0']]

P, T, V_Si = zip(*data)
PT_Si = [P, T]

print optimize.curve_fit(fit_PVT_data_full(Si_liq), PT_Si, V_Si, guesses)


Fe_liq.set_state(Pref, Tref) 
FeSi_melt.set_state(Pref, Tref)

Sconf = -burnman.constants.gas_constant*np.log(0.5)
print Fe_liq.H, (FeSi_melt.H + Tref*Sconf), 2.*(FeSi_melt.H + Tref*Sconf) - Fe_liq.H
print Fe_liq.S, (FeSi_melt.S - Sconf), 2.*(FeSi_melt.S - Sconf) - Fe_liq.S
print Fe_liq.params['Cp'][0], FeSi_melt.params['Cp'][0], 2.*FeSi_melt.params['Cp'][0] - Fe_liq.params['Cp'][0]
