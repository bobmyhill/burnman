# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.

from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import minerals

from burnman.solidsolution import SolidSolution


class pyrope_grossular(SolidSolution):

    def __init__(self, molar_fractions=None):
        self.name = 'garnet'
        self.type = 'full_subregular'
        self.n_atoms = 20.
        self.T_0 = 300.
        self.P_0 = 1.e5
        self.endmembers = [[minerals.SLB_2011.py(), '[Mg]3Al2Si3O12'],
                           [minerals.SLB_2011.gr(), '[Ca]3Al2Si3O12']]
        self.energy_interaction = [[[0., 0.]]]
        self.volume_interaction = [[[1.2e-6, 1.2e-6]]]
        SolidSolution.__init__(self, molar_fractions)



pygr = pyrope_grossular()
pygr.set_composition([0.5, 0.5])

temperatures = [300., 1000., 2000.]
pressures = np.linspace(1.e5, 100.e9, 31)
Vex = np.empty_like(pressures)
for T in temperatures:
    for i, P in enumerate(pressures):
        pygr.set_state(P, T)
        Vex[i] = pygr.excess_volume
    plt.plot(pressures/1.e9, Vex, label=str(T)+' K')



def findPideal(P, V, T, m1, m2):
    m1.set_state(P[0], T)
    m2.set_state(P[0], T)
    return V - 0.5*(m1.V + m2.V)

def volume_excess(Vnonideal0, Videal0, Kideal0, Kprime, pressure):
    Vexcess0 = Vnonideal0 - Videal0
    Knonideal0 = Kideal0*np.power(Videal0/Vnonideal0, Kprime)
    bideal = Kprime/Kideal0
    bnonideal = Kprime/Knonideal0
    c = 1./Kprime
    return Vnonideal0*np.power((1.+bnonideal*pressure), -c) \
      - Videal0*np.power((1.+bideal*(pressure)), -c)

def intVdP_excess(Vnonideal0, Videal0, Kideal0, Kprime, pressure):
    if np.abs(pressure) < 1.:
        pressure = 1.
    Vexcess0 = Vnonideal0 - Videal0
    Knonideal0 = Kideal0*np.power(Videal0/Vnonideal0, Kprime)
    bideal = Kprime/Kideal0
    bnonideal = Kprime/Knonideal0
    c = 1./Kprime
    return -pressure*(Vnonideal0*np.power((1.+bnonideal*pressure), 1.-c)/(bnonideal*(c - 1.)*pressure) \
                      - Videal0*np.power((1.+bideal*pressure), 1.-c)/(bideal*(c - 1.)*pressure))


def gibbs_excess(P, T, T0, n, Gxs0, Vxs0, Kprime, f_Pth, m1, m2):
    # Ideal and nonideal properties at standard state
    P_0 = 1.e5
    m1.set_state(P_0, T0)
    m2.set_state(P_0, T0)
    V0_ideal = 0.5*(m1.V + m2.V)
    K0_ideal = 2.0*V0_ideal/(m1.V/m1.K_T + m2.V/m2.K_T)
    V0_nonideal = V0_ideal + Vxs0
    
    # Properties at pressure
    m1.set_state(P, T)
    m2.set_state(P, T)
    V_ideal = 0.5*(m1.V + m2.V)
    
    # First, heres Pth (\int aK_T dT | V) for the ideal phase
    # when V=V0 and T=T1 
    P_V0ideal_T = fsolve(findPideal, [P_0 + 5.e6*(T - T0)], args=(V0_ideal, T, m1, m2))[0]
    Pth_V0ideal_T =  P_V0ideal_T - P_0

    # Make the assumption that Pth(V0, T)_nonideal = Pth(V0, T)_ideal*f_Pth 
    Pth_V0nonideal_T = Pth_V0ideal_T*f_Pth
    m1.set_state(P_0 + Pth_V0nonideal_T, T)
    m2.set_state(P_0 + Pth_V0nonideal_T, T)
    V_V0nonideal_ideal = 0.5*(m1.V + m2.V)
    
    # Make the further assumption that the form of the excess volume curve is temperature independent
    V_excess = volume_excess(V0_nonideal, V_V0nonideal_ideal,
                             K0_ideal, Kprime,
                             P - Pth_V0nonideal_T - P_0)
    V_nonideal = V_ideal + V_excess

    # Calculate contributions to the gibbs free energy
    # 1. The isothermal path along T0 from P0 to infinite pressure 
    Gxs_T0 = - intVdP_excess(V0_nonideal, V0_ideal, K0_ideal, Kprime, P_0 - P_0)
    # 2. The isobaric path at infinite pressure from T0 to T has no excess contribution
    # 3. The isothermal path from infinite pressure down to P, T
    Gxs_T = intVdP_excess(V0_nonideal, V_V0nonideal_ideal,
                          K0_ideal, Kprime,
                          P - Pth_V0nonideal_T - P_0)
    
    Gxs = Gxs0 + Gxs_T0 + Gxs_T
    return Gxs, V_excess


py = minerals.SLB_2011.pyrope()
gr = minerals.SLB_2011.grossular()

Vxs0 = 0.3e-6
Kprime = 7.
f_Pth = 1.000
Gxs0 = 0.
T0=300.
n = 20.

'''
pressures = [1.e5, 10.e9, 50.e9, 100.e9]
temperatures = np.linspace(1., 2000., 101)
gibbs_excesses = np.empty_like(temperatures)
S_excesses = np.empty_like(temperatures)
Cp_excesses = np.empty_like(temperatures)
for P in pressures:
    for i, T in enumerate(temperatures):
        G0 = gibbs_excess(P, T-0.5, T0, n, Gxs0, Vxs0, Kprime, f_Pth, py, gr)[0]
        G1 = gibbs_excess(P, T, T0, n, Gxs0, Vxs0, Kprime, f_Pth, py, gr)[0]
        G2 = gibbs_excess(P, T+0.5, T0, n, Gxs0, Vxs0, Kprime, f_Pth, py, gr)[0]
        gibbs_excesses[i] = G1
        S_excesses[i] = G0 - G2
        Cp_excesses[i] = T*(2.*G1 - G0 - G2)/0.25
    plt.plot(temperatures, S_excesses, label=str(P/1.e9)+' GPa')
    
plt.legend(loc='lower right')
plt.show()
'''

temperatures = [300., 1000., 2000.]
pressures = np.linspace(1.e5, 100.e9, 31)
gibbs_excesses = np.empty_like(pressures)
S_excesses = np.empty_like(pressures)
V_excesses = np.empty_like(pressures)
V2_excesses = np.empty_like(pressures)
for T in temperatures:
    for i, P in enumerate(pressures):
        gibbs_excesses[i] = gibbs_excess(P, T, T0, n, Gxs0, Vxs0, Kprime, f_Pth, py, gr)[0]
        S_excesses[i] = gibbs_excess(P, T-0.5, T0, n, Gxs0, Vxs0, Kprime, f_Pth, py, gr)[0] - gibbs_excess(P, T+0.5, T0, n, Gxs0, Vxs0, Kprime, f_Pth, py, gr)[0]
        V_excesses[i] = (gibbs_excess(P+500., T, T0, n, Gxs0, Vxs0, Kprime, f_Pth, py, gr)[0] - gibbs_excess(P-500., T, T0, n, Gxs0, Vxs0, Kprime, f_Pth, py, gr)[0])/1000.
        V2_excesses[i] = gibbs_excess(P+500., T, T0, n, Gxs0, Vxs0, Kprime, f_Pth, py, gr)[1]
    plt.plot(pressures/1.e9, V_excesses, marker='o', linestyle='None', label=str(T)+' K')
    plt.plot(pressures/1.e9, V2_excesses, marker='x', linestyle='None', label=str(T)+' K, direct')
    
plt.legend(loc='lower right')
plt.xlabel('P (GPa)')
plt.show()
