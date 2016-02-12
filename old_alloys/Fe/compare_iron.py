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

T_0 = 4000.
P_0 =  100.e9
Fe_hcp2.params['T_0'] = T_0
Fe_hcp2.params['P_0'] = P_0

Fe_hcp.set_state(P_0, T_0)
V_0 = Fe_hcp.V
K_0 = Fe_hcp.K_T
a_0 = Fe_hcp.alpha

Fe_hcp2.params['S_0'] = Fe_hcp2.params['S_0'] + 71.
guesses = [V_0, K_0, 4., -4./K_0, a_0]


pressures = np.linspace(60.e9, 160.e9, 3)
temperatures = np.linspace(300., 3000., 11)
volumes = np.empty_like(pressures)
PT = []
V = []
for P in pressures:
    for T in temperatures:
        Fe_hcp.set_state(P, T)
        PT.append([P, T])
        V.append(Fe_hcp.V)

PT = zip(*PT)

popt, pcov = curve_fit(fit_EoS_data(Fe_hcp2, ['V_0', 'K_0', 'Kprime_0', 'Kdprime_0', 'a_0']),
                       PT, V, guesses)
print guesses
print popt



Fe_hcp2.set_state(1.e5, 300.)
print Fe_hcp2.K_T/1.e9


# First, let's do some benchmarking
data = listify_xy_file('data/Komabayashi_benchmark.dat')

pressures, temperatures, alpha_hcp, KT_hcp, V_hcp, gamma_hcp, alpha_liq, KT_liq, V_liq, gamma_liq = data

for i, P in enumerate(pressures):
    P = P*1.e9
    T = temperatures[i]
    Fe_hcp.set_state(P, T)

    print alpha_hcp[i], KT_hcp[i], V_hcp[i]
    print Fe_hcp.alpha, Fe_hcp.K_T/1.e9, Fe_hcp.V*1.e6

    
print 'NOW compare HCP model from this study'
for i, P in enumerate(pressures):
    P = P*1.e9
    T = temperatures[i]
    Fe_hcp.set_state(P, T-400.)
    Fe_hcp2.set_state(P, T-400.)

    print P/1.e9, T
    print Fe_hcp.alpha, Fe_hcp.K_T/1.e9, Fe_hcp.V*1.e6, Fe_hcp.S, Fe_hcp.gr 
    print Fe_hcp2.alpha, Fe_hcp2.K_T/1.e9, Fe_hcp2.V*1.e6, Fe_hcp2.S, Fe_hcp2.gr 
    print ''
    
    Fe_hcp.set_state(P, T)
    Fe_hcp2.set_state(P, T)

    print Fe_hcp.alpha, Fe_hcp.K_T/1.e9, Fe_hcp.V*1.e6, Fe_hcp.S, Fe_hcp.gr 
    print Fe_hcp2.alpha, Fe_hcp2.K_T/1.e9, Fe_hcp2.V*1.e6, Fe_hcp2.S, Fe_hcp2.gr 
    print ''





