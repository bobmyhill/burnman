# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2019 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
example_anisotropic_eos
----------------

This example script demonstrates the anisotropic equation of state
of Myhill (2019).

*Uses:*

* :doc:`mineral_database`


*Demonstrates:*

* creating a mineral with excess contributions
* calculating thermodynamic properties
"""
from __future__ import absolute_import

# Here we import standard python modules that are required for
# usage of BurnMan.  In particular, numpy is used for handling
# numerical arrays and mathematical operations on them, and
# matplotlib is used for generating plots of results of calculations
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))


# Here we import the relevant modules from BurnMan.  The burnman
# module imports several of the most important functionalities of
# the library, including the ability to make composites, and compute
# thermoelastic properties of them.  The minerals module includes
# the mineral physical parameters for the predefined minerals in
# BurnMan
from burnman import eos
from burnman import AnisotropicMineral
from burnman.constants import Avogadro
from burnman.minerals import SLB_2011
from burnman.minerals import HGP_2018_ds633
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz



if __name__ == "__main__":

    fig = plt.figure()
    ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
    bdg_SLB = SLB_2011.mg_bridgmanite()

    P0s = [0., 2.5e9, 10.e9, 40.e9, 160.e9]
    temperatures = np.linspace(10., 2000., 101)
    pressures = np.empty_like(temperatures)
    for P0 in P0s:
        bdg_SLB.set_state(P0, 10.)
        V = bdg_SLB.V
        for i, T in enumerate(temperatures):
            pressures[i] = bdg_SLB.method.pressure(T, V, bdg_SLB.params)
        #plt.plot(pressures/1.e9, temperatures)
        ax[0].plot(temperatures, pressures/1.e9 - pressures[0]/1.e9)



    pressures = [0., 2.5e9, 10.e9, 40.e9, 160.e9]
    temperatures = np.linspace(10., 2000., 101)
    pths = np.empty_like(temperatures)
    for P in pressures:
        Ps = P + 0.*temperatures
        Vs = bdg_SLB.evaluate(['V'], Ps, temperatures)[0]
        for i, V in enumerate(Vs):
            P0 = bdg_SLB.method.pressure(10., V, bdg_SLB.params)
            pths[i] = P - P0
        ax[1].plot(temperatures, Vs*pths/1.e9/bdg_SLB.params['V_0'], label='{0}'.format(P/1.e9))
    ax[1].legend()

    Eth = [eos.einstein.thermal_energy(T, 800., 5.) for T in temperatures]
    gr = 1.5
    V = bdg_SLB.params['V_0']
    ax[1].plot(temperatures, gr/V*(Eth-Eth[0])/1.e9)


    bdg_HP = HGP_2018_ds633.mpv()
    def delta_V(V, P, T):
        Tref = 300.
        gr = 1.5
        theta = 800.
        n = 5.
        Eth = eos.einstein.thermal_energy(T, theta, n)
        Eth0 = eos.einstein.thermal_energy(Tref, theta, n)
        Pth = gr/V*(Eth-Eth0)
        P_Tref = P-Pth
        bdg_HP.set_state(P_Tref, Tref)
        return bdg_HP.V - V

    P = 10.e9
    T = 1000.
    V = brentq(delta_V,
               0.2*bdg_HP.params['V_0'],
               2.0*bdg_HP.params['V_0'],
               args=(P, T))
    print(V)

    plt.show()

    """
    def gr_SLB(x, q0):
        f = 0.5*(np.power(x, -2./3.) - 1.)
        th0 = 800.
        gr0 = 1.3 # 1.63
        g = (-1 + 3*gr0 - 1.5*q0)*f
        nu_o_nu0_sq = (1. + 6.*gr0*f*(1. + g))
        #th = th0*np.sqrt(nu_o_nu0_sq)
        gr = gr0/nu_o_nu0_sq*(2.*f + 1.)*(1. + 2*g)
        return gr

    xs = np.linspace(0.5, 1.5, 101)
    for q0 in [1., 1.25, 1.5, 1.75, 2.]:
        plt.plot(xs, gr_SLB(xs, q0))
    plt.show()
    """


    def gr_SD(x):
        gr0 = 1.6
        grinf = 1.2
        q0 = 2.
        lmda = q0/np.log(gr0/grinf)
        gr = gr0*np.exp(q0/lmda*(np.power(x, lmda) - 1.))
        return gr

    xs = np.linspace(0.0, 1.0, 101)
    plt.plot(xs, gr_SD(xs))
    plt.show()




#Here's the stuff that we rip from Stixrude...


def isothermal_bulk_modulus(self, pressure, temperature, volume, params):
    """
    Returns isothermal bulk modulus :math:`[Pa]`
    """
    T_0 = params['T_0']
    einstein_T = self._einstein_temperature(params['V_0'] / volume, params)
    gr = self.grueneisen_parameter(pressure, temperature, volume, params)

    E_th = einstein.thermal_energy(temperature, einstein_T, params['n'])
    E_th_ref = einstein.thermal_energy(T_0, einstein_T, params['n'])

    C_v = einstein.molar_heat_capacity_v(temperature, einstein_T, params['n'])
    C_v_ref = einstein.molar_heat_capacity_v(T_0, einstein_T, params['n'])

    q = self.volume_dependent_q(params['V_0'] / volume, params)

    dPthdV = ((gr + 1. - q) * gr * (E_th - E_th_ref)
              - pow(gr, 2.) * (C_v * temperature - C_v_ref * T_0))
    dVdP_T0 = isothermal_compressibility / volume

    K = bm.bulk_modulus(volume, params) * (1. + dPthdV * dVdP_T0)

    return K
