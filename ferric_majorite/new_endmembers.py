# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import Mineral, minerals
from burnman.processchemistry import dictionarize_formula, formula_mass
from burnman.equilibrate import equilibrate
import matplotlib.image as mpimg
from scipy.optimize import fsolve

fcc = minerals.Brosh_2007.fcc_iron()
bcc = minerals.Brosh_2007.bcc_iron()
hcp = minerals.Brosh_2007.hcp_iron()
liq = minerals.Brosh_2007.liquid_iron()

def find_pressure(T, m1, m2, P_guess=5.e9):
    def delta_gibbs(P, T, m1, m2):
        m1.set_state(P[0], T)
        m2.set_state(P[0], T)
        return m1.gibbs - m2.gibbs

    return fsolve(delta_gibbs, [P_guess], args=(T, m1, m2))[0]

def find_temperature(P, m1, m2, T_guess=1000.):
    def delta_gibbs(T, P, m1, m2):
        m1.set_state(P, T[0])
        m2.set_state(P, T[0])
        return m1.gibbs - m2.gibbs

    return fsolve(delta_gibbs, [T_guess], args=(P, m1, m2))[0]


Fe_diag_img = mpimg.imread('figures/fe_perplex.png') # from SE15ver.dat bundled with PerpleX 6.8.3 (September 2018)
Fe_diag_img = mpimg.imread('figures/fe_brosh.png') # alternative, from paper
plt.imshow(Fe_diag_img, extent=[0.0, 350.0, 300, 8000], aspect='auto')


pressures = np.linspace(1.e5, 10.e9, 101)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = find_temperature(P, bcc, fcc, T_guess=500.)

plt.plot(pressures/1.e9, temperatures)

pressures = np.linspace(1.e5, 1.e9, 101)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = find_temperature(P, bcc, fcc, T_guess=1500.)

plt.plot(pressures/1.e9, temperatures)


pressures = np.linspace(1.e5, 1.e9, 101)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = find_temperature(P, bcc, liq, T_guess=1500.)

plt.plot(pressures/1.e9, temperatures)

pressures = np.linspace(1.e5, 65.e9, 101)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = find_temperature(P, fcc, liq, T_guess=2500.)

plt.plot(pressures/1.e9, temperatures)

pressures = np.linspace(55.e9, 350.e9, 101)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = find_temperature(P, hcp, liq, T_guess=3000.)

plt.plot(pressures/1.e9, temperatures)


temperatures = np.linspace(700., 4000., 101)
pressures = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    pressures[i] = find_pressure(T, fcc, hcp)

plt.plot(pressures/1.e9, temperatures)
    
temperatures = np.linspace(300., 1000., 101)
pressures = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    pressures[i] = find_pressure(T, bcc, hcp)

plt.plot(pressures/1.e9, temperatures)
plt.show()



Fe_EoS_img = mpimg.imread('figures/fe_RT_eos_and_hugoniot.png')
plt.imshow(Fe_EoS_img, extent=[0.0, 500.0, 3.5, 7.5], aspect='auto')

pressures = np.linspace(0., 500.e9, 101)

for m in [fcc, bcc, hcp]:
    for T in [300]:
        
        temperatures = T + pressures*0.
        gibbs_m = m.evaluate(['gibbs'], pressures, temperatures)[0]

    
        plt.plot(pressures/1.e9, np.gradient(gibbs_m, pressures, edge_order=2)*1.e6, label=str(T)+' '+m.name)
        plt.scatter([0.], [m.params['V_0']*1.e6])
        
plt.xlim(0., 500.)
plt.legend(loc='best')
plt.show()


pressures = np.linspace(0., 50.e9, 101)

for T in [300., 500., 700.]:
    temperatures = T + pressures*0.
    delta_gibbs = bcc.evaluate(['gibbs'], pressures, temperatures)[0] - hcp.evaluate(['gibbs'], pressures, temperatures)[0]
    plt.plot(pressures/1.e9, delta_gibbs, label=str(T))

    
for T in [1000., 1500., 2000.]:
    temperatures = T + pressures*0.
    delta_gibbs = fcc.evaluate(['gibbs'], pressures, temperatures)[0] - hcp.evaluate(['gibbs'], pressures, temperatures)[0]
    plt.plot(pressures/1.e9, delta_gibbs, label=str(T))


plt.ylim(-100., 100.)
plt.legend(loc='best')
plt.show()

exit()


temperatures = np.linspace(300., 3000., 1001)

for P in [1.e5]:
    pressures = 0.*temperatures + P

    gibbs_fcc = fcc.evaluate(['gibbs'], pressures, temperatures)[0]
    gibbs_bcc = bcc.evaluate(['gibbs'], pressures, temperatures)[0]
    gibbs_hcp = hcp.evaluate(['gibbs'], pressures, temperatures)[0]
    gibbs_liq = liq.evaluate(['gibbs'], pressures, temperatures)[0]
    
    plt.plot(temperatures, gibbs_bcc - gibbs_fcc)
    plt.plot(temperatures, gibbs_hcp - gibbs_fcc)
plt.ylim(0., 100.)
plt.show()
