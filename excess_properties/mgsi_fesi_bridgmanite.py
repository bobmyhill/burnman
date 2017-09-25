# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
Pyrope-Grossular "ideal" solution (where ideality is in Helmholtz free energy)
"""
from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from excess_modelling import *

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import minerals

from scipy.optimize import brentq, curve_fit


plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'

formula = burnman.processchemistry.dictionarize_formula('MgSiO3')
params = {'equation_of_state': 'bm3',
          'V_0': 24.445e-6,
          'K_0': 253.e9,
          'Kprime_0': 3.90,
          'molar_mass': burnman.processchemistry.formula_mass(formula),
          'n': 5.,
          'formula': formula}
MgSiO3 = burnman.Mineral(params=params)

formula = burnman.processchemistry.dictionarize_formula('Mg0.87Fe0.13SiO3')
params = {'equation_of_state': 'bm3',
          'V_0':  163./1.e30*burnman.constants.Avogadro/4.,
          'K_0': 253.e9,
          'Kprime_0': 3.90,
          'molar_mass': burnman.processchemistry.formula_mass(formula),
          'n': 5.,
          'formula': formula}
Wolf_bdg = burnman.Mineral(params=params)

formula = burnman.processchemistry.dictionarize_formula('FeSiO3')
params = {'equation_of_state': 'bm3',
          'V_0':  24.88e-6 ,
          'K_0': 251.e9,
          'Kprime_0': 3.9,
          'molar_mass': burnman.processchemistry.formula_mass(formula),
          'n': 5.,
          'formula': formula}
FeSiO3 = burnman.Mineral(params=params)


P, Perr, V, Verr = np.loadtxt('data/Wolf_FeSiO3_13.dat', unpack=True)
Z = 4.

# Fit pypv data from Walter et al. (2004)
P = P*1.e9
Perr = Perr*1.e9
T = np.array([300.]*len(P))
Terr = np.array([1.]*len(P))
V = V/1.e30*burnman.constants.Avogadro/Z
Verr = Verr/1.e30*burnman.constants.Avogadro/Z

PTV = np.array([P, T, V]).T
nul = 0.*PTV.T[0]
PTV_covariances = np.array([[Perr*Perr, nul, nul],
                            [nul, Terr*Terr, nul],
                            [nul, nul, Verr*Verr]]).T
fitted_eos = burnman.eos_fitting.fit_PTV_data(Wolf_bdg, ['V_0', 'K_0'], PTV, PTV_covariances, verbose=False)
burnman.tools.pretty_print_values(fitted_eos.popt, fitted_eos.pcov, fitted_eos.fit_params)


pressures = np.linspace(1.e5, 120.e9, 101)
plt.errorbar(P/1.e9, V*1.e6, xerr=Perr/1.e9, yerr=Verr*1.e6, linestyle='None')
plt.plot(pressures/1.e9, Wolf_bdg.evaluate(['V'], pressures, [300.]*len(pressures))[0]*1.e6)


pressures_Fiquet, Perr_Fiquet, volumes_Fiquet, Verr_Fiquet = np.loadtxt('data/Fiquet_2000_MgSiO3_pv_PV.dat', unpack=True)
volumes_Fiquet = volumes_Fiquet/1.e30*burnman.constants.Avogadro/Z
Verr_Fiquet = Verr_Fiquet/1.e30*burnman.constants.Avogadro/Z

plt.errorbar(pressures_Fiquet, volumes_Fiquet*1.e6, xerr=Perr_Fiquet, yerr=Verr_Fiquet*1.e6, linestyle='None')
plt.plot(pressures/1.e9, MgSiO3.evaluate(['V'], pressures, [300.]*len(pressures))[0]*1.e6)


volumes_ss = np.empty_like(pressures)
KT_ss = np.empty_like(pressures)
for i, pressure in enumerate(pressures):
    s = solution(pressure, 300.,
                 F_xs = 0., p_xs = 0.,
                 x_a = 0.87, a = MgSiO3, b = FeSiO3, cluster_size = 2.)
    volumes_ss[i] = s.V
    KT_ss[i] = s.K_T

Kprime_ss = np.gradient(KT_ss, pressures, edge_order=2)
print('V_0: {0}'.format(volumes_ss[0])) 
print('K_0: {0}'.format(KT_ss[0])) 
print('K\'_0: {0}'.format(Kprime_ss[0]))   
plt.plot(pressures/1.e9, volumes_ss*1.e6)


plt.show()
