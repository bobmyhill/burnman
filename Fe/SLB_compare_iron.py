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


T = 300.
pressures = np.linspace(1.e5, 350.e9, 101)
alphas = np.empty_like(pressures)
for i, P in enumerate(pressures):
    Fe_hcp.set_state(P, T)
    alphas[i] = Fe_hcp.alpha

burnman.tools.array_to_file(['P (GPa)', 'alpha (1e-5/K)'],[pressures/1.e9, alphas*1.e5], 'output/Komabayashi_2014_hcp_alphas.dat')
    

Fe_hcp3 = burnman.minerals.Myhill_calibration_iron.hcp_iron_SLB()

Fe_hcp2.params['S_0'] = Fe_hcp2.params['S_0'] + 10.
guesses = [Fe_hcp2.params['V_0'],
           Fe_hcp2.params['K_0'],
           Fe_hcp2.params['Kprime_0']]


pressures = np.linspace(1.e5, 250.e9, 101)
temperatures = np.empty_like(pressures)
volumes = np.empty_like(pressures)
for i, P in enumerate(pressures):
    T = Fe_hcp3.params['T_0']
    temperatures[i] = T
    Fe_hcp.set_state(P, T)
    volumes[i] = Fe_hcp.V

PT = [pressures, temperatures]
V = volumes
    

popt, pcov = curve_fit(fit_EoS_data(Fe_hcp3, ['V_0', 'K_0', 'Kprime_0']), PT, V, guesses)
print guesses
print popt




guesses = [Fe_hcp3.params['grueneisen_0'] ]

temperatures = np.linspace(3000., 5000., 101)
pressures = np.empty_like(temperatures)
volumes = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    P = 100.e9
    pressures[i] = P
    Fe_hcp.set_state(P, T)
    volumes[i] = Fe_hcp.V

    
PT = [pressures, temperatures]
V = volumes

popt, pcov = curve_fit(fit_EoS_data(Fe_hcp3, ['grueneisen_0']), PT, V, guesses)
print guesses
print popt




T_ref = 1809.
P_ref = 50.e9
HP_convert(Fe_fcc2, 300., 2200., T_ref, P_ref)
HP_convert(Fe_hcp2, 300., 2200., T_ref, P_ref)
HP_convert(Fe_liq2, 1809., 2400., T_ref, P_ref)


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
    Fe_hcp3.set_state(P, T-400.)

    print P/1.e9, T
    print Fe_hcp.alpha, Fe_hcp.K_T/1.e9, Fe_hcp.V*1.e6, Fe_hcp.S
    print Fe_hcp3.alpha, Fe_hcp3.K_T/1.e9, Fe_hcp3.V*1.e6, Fe_hcp3.S
    print ''
    
    Fe_hcp.set_state(P, T)
    Fe_hcp3.set_state(P, T)

    print Fe_hcp.alpha, Fe_hcp.K_T/1.e9, Fe_hcp.V*1.e6, Fe_hcp.S, Fe_hcp.gr 
    print Fe_hcp3.alpha, Fe_hcp3.K_T/1.e9, Fe_hcp3.V*1.e6, Fe_hcp3.S, Fe_hcp3.gr 
    print ''





