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

def findPideal(P, V, Tstd, m1, m2):
    m1.set_state(P[0], Tstd)
    m2.set_state(P[0], Tstd)
    return V - 0.5*(m1.V + m2.V)

def volume_excess(Vexcess0, Videal0, Kideal0, Kprime, pressure):
    Vnonideal0 = Videal0 + Vexcess0
    Knonideal0 = Kideal0*np.power(Videal0/Vnonideal0, Kprime)
    bideal = Kprime/Kideal0
    bnonideal = Kprime/Knonideal0
    c = 1./Kprime
    return Vnonideal0*np.power((1.+bnonideal*pressure), -c) \
      - Videal0*np.power((1.+bideal*(pressure)), -c)

def intVdP_excess(Vexcess0, Videal0, Kideal0, Kprime, pressure):
    Vnonideal0 = Videal0 + Vexcess0
    Knonideal0 = Kideal0*np.power(Videal0/Vnonideal0, Kprime)
    bideal = Kprime/Kideal0
    bnonideal = Kprime/Knonideal0
    c = 1./Kprime
    return -pressure*(Vnonideal0*np.power((1.+bnonideal*pressure), 1.-c)/(bnonideal*(c - 1.)*pressure) \
                      - Videal0*np.power((1.+bideal*pressure), 1.-c)/(bideal*(c - 1.)*pressure))
    

def gibbs_excess(P, T, Vxs0, Kprime, n_D, Gxs0, m1, m2):

    # Ideal properties at standard state
    n = m1.params['n']
    T0 = m1.params['T_0']

    m1.set_state(1.e5, T0)
    m2.set_state(1.e5, T0)
    V0_ideal = 0.5*(m1.V + m2.V)
    K0_ideal = 2.0*V0_ideal/(m1.V/m1.K_T + m2.V/m2.K_T)
    a0_ideal = 0.5/V0_ideal*(m1.alpha*m1.V + m2.alpha*m2.V)
    Cp0_ideal = 0.5*(m1.C_p + m2.C_p)
    debye0_ideal = np.sqrt(m1.params['Debye_0']*m2.params['Debye_0'])
    Cv0_ideal = Cp0_ideal - V0_ideal*T0*a0_ideal*a0_ideal*K0_ideal
    gr0_ideal = a0_ideal*K0_ideal*V0_ideal/Cv0_ideal
    q0_ideal = 0.5*(m1.params['q_0'] + m2.params['q_0']) 

    # Non ideal properties at standard state
    V0_nonideal = V0_ideal + Vxs0
    V0overV0_ideal = V0_nonideal/V0_ideal
    debye0_nonideal = debye0_ideal*np.power(V0overV0_ideal, n_D)
    
    Cv0_nonideal = burnman.eos.debye.heat_capacity_v(T0, debye0_nonideal, n)
    Cv0overCv0_ideal = Cv0_nonideal/Cv0_ideal
    gr0_nonideal = gr0_ideal*V0overV0_ideal/Cv0overCv0_ideal
    
    # Ideal properties at pressure
    m1.set_state(P, T)
    m2.set_state(P, T)
    V_ideal = 0.5*(m1.V + m2.V)
    f_ideal = 0.5*(pow(V0_ideal/V_ideal, 2./3.) - 1.)
    a2_iikk_ideal = -12.*gr0_ideal + 36.*gr0_ideal*gr0_ideal \
                    - 18.*q0_ideal * gr0_ideal
    fdebye_ideal = np.sqrt(1. + 6.*gr0_ideal*f_ideal + 0.5*a2_iikk_ideal*f_ideal*f_ideal)
    debye_ideal = debye0_ideal * fdebye_ideal
    
    # Non ideal volume at pressure
    P_ideal = fsolve(findPideal, [P], args=(V_ideal, T0, m1, m2))[0]
    Vxs = volume_excess(Vxs0, V0_ideal, K0_ideal, Kprime, P_ideal)

    f_nonideal = 0.5*(pow((V0_ideal+Vxs0)/(V_ideal + Vxs), 2./3.) - 1.)
    a2_iikk_nonideal = a2_iikk_ideal/(debye0_ideal*debye0_ideal)*(debye0_nonideal*debye0_nonideal)
    fdebye_nonideal = np.sqrt(1. + 6.*gr0_nonideal*f_nonideal + 0.5*a2_iikk_nonideal*f_nonideal*f_nonideal)
    debye_nonideal = debye0_nonideal * fdebye_nonideal
    
    # Calculate contributions to the excess Gibbs free energy
    F_quasiharmonic_ideal = burnman.eos.debye.helmholtz_free_energy(T, debye_ideal, n) - \
                            burnman.eos.debye.helmholtz_free_energy(T0, debye_ideal, n)
    F_quasiharmonic_nonideal = burnman.eos.debye.helmholtz_free_energy(T, debye_nonideal, n) - \
                               burnman.eos.debye.helmholtz_free_energy(T0, debye_nonideal, n)
    
    #S_quasiharmonic_ideal = burnman.eos.debye.entropy(T, debye_T_ideal, n)
    #S_quasiharmonic_nonideal = burnman.eos.debye.entropy(T, debye_T_nonideal, n)

    Gxs_T0 = intVdP_excess(Vxs0, V0_ideal, K0_ideal, Kprime, P_ideal) - intVdP_excess(Vxs0, V0_ideal, K0_ideal, Kprime, 1.e5)
    Gxs_quasiharmonic = F_quasiharmonic_nonideal - F_quasiharmonic_ideal
    Gxs = Gxs0 + Gxs_T0 + Gxs_quasiharmonic
    return Gxs


py = minerals.SLB_2011.pyrope()
gr = minerals.SLB_2011.grossular()

#py = minerals.HP_2011_ds62.hlt()
#gr = minerals.HP_2011_ds62.syv()


Vxs0 = 0.3e-6
Kprime = 7.
n_D = 1./6. - Kprime/2.
#n_D = -1.
Gxs0 = 0.


pressures = [1.e5, 10.e9, 50.e9, 100.e9]
temperatures = np.linspace(1., 2000., 31)
gibbs_excesses = np.empty_like(temperatures)
S_excesses = np.empty_like(temperatures)
Cp_excesses = np.empty_like(temperatures)
for P in pressures:
    for i, T in enumerate(temperatures):
        gibbs_excesses[i] = gibbs_excess(P, T, Vxs0, Kprime, n_D, Gxs0, py, gr)
        G0 = gibbs_excess(P, T-0.5, Vxs0, Kprime, n_D, Gxs0, py, gr)
        G1 = gibbs_excess(P, T, Vxs0, Kprime, n_D, Gxs0, py, gr)
        G2 = gibbs_excess(P, T+0.5, Vxs0, Kprime, n_D, Gxs0, py, gr)
        S_excesses[i] = G0 - G2
        Cp_excesses[i] = T*(2.*G1 - G0 - G2)/0.25
    plt.plot(temperatures, S_excesses, label=str(P/1.e9)+' GPa')
    
plt.legend(loc='lower right')
plt.show()


temperatures = [300., 1000., 2000.]
pressures = np.linspace(1.e5, 50.e9, 31)
gibbs_excesses = np.empty_like(pressures)
S_excesses = np.empty_like(pressures)
V_excesses = np.empty_like(pressures)
for T in temperatures:
    for i, P in enumerate(pressures):
        gibbs_excesses[i] = gibbs_excess(P, T, Vxs0, Kprime, n_D, Gxs0, py, gr)
        #S_excesses[i] = gibbs_excess(P, T-0.5, Vxs0, Kprime, n_D, Gxs0, py, gr) - gibbs_excess(P, T+0.5, Vxs0, Kprime, n_D, Gxs0, py, gr)
        V_excesses[i] = (gibbs_excess(P+500., T, Vxs0, Kprime, n_D, Gxs0, py, gr) - gibbs_excess(P-500., T, Vxs0, Kprime, n_D, Gxs0, py, gr))/1000.
    plt.plot(pressures/1.e9, gibbs_excesses, label=str(T)+' K')
    
plt.legend(loc='lower right')
plt.xlabel('P (GPa)')
plt.show()
