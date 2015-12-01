# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

from scipy.optimize import fsolve, curve_fit
import burnman
from burnman.minerals import Komabayashi_2014

from listify_xy_file import *
from fitting_functions import *
from HP_convert import *

Fe_hcp=Komabayashi_2014.hcp_iron()
Fe_fcc=Komabayashi_2014.fcc_iron()
Fe_liq=Komabayashi_2014.liquid_iron()
Fe_fcc2 = burnman.minerals.Myhill_calibration_iron.fcc_iron()
Fe_hcp2 = burnman.minerals.Myhill_calibration_iron.hcp_iron()
Fe_liq2 = burnman.minerals.Myhill_calibration_iron.liquid_iron()

T_ref = 1809.
P_ref = 50.e9
HP_convert(Fe_fcc2, 300., 2200., T_ref, P_ref)
HP_convert(Fe_hcp2, 300., 2200., T_ref, P_ref)
HP_convert(Fe_liq2, 1809., 2400., T_ref, P_ref)


pressures = np.linspace(1.e9, 50.e9, 6)
for P in pressures:
    T=fsolve(eqm_temperature([Fe_fcc, Fe_liq], [1.0, -1.0]), [2000.], args=(P))[0]
    Fe_fcc2.set_state(P, T)
    Fe_liq2.set_state(P, T)
    print P, T, Fe_fcc.S, Fe_liq.S, Fe_liq.S - Fe_fcc.S, Fe_liq2.S, Fe_fcc2.S

pressures = np.linspace(60.e9, 120.e9, 7)
for P in pressures:
    T=fsolve(eqm_temperature([Fe_hcp, Fe_liq], [1.0, -1.0]), [2000.], args=(P))[0]
    Fe_hcp2.set_state(P, T)
    Fe_liq2.set_state(P, T)
    print P, T, Fe_hcp.S, Fe_liq.S, Fe_liq.S - Fe_hcp.S, Fe_liq2.S, Fe_hcp2.S
