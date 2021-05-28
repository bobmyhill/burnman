# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.

"""
example_spintransition
----------------------

This example shows the different minerals that are implemented with a spin
transition.  Minerals with spin transition are implemented by defining two
separate minerals (one for the low and one for the high spin state).  Then a
third dynamic mineral is created that switches between the two previously
defined minerals by comparing the current pressure to the transition pressure.

*Specifically uses:*

* :func:`burnman.mineral_helpers.HelperSpinTransition`
* :func:`burnman.minerals.Murakami_etal_2012.fe_periclase`
* :func:`burnman.minerals.Murakami_etal_2012.fe_periclase_HS`
* :func:`burnman.minerals.Murakami_etal_2012.fe_periclase_LS`


*Demonstrates:*

* implementation of spin transition in (Mg,Fe)O at user defined pressure
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import Mineral
from burnman.processchemistry import dictionarize_formula, formula_mass


from periclase_endmembers import high_spin_wuestite, low_spin_wuestite, periclase

if __name__ == "__main__":

    class SpinEOS(object):
        def __init__(self, params):
            self.params = params
            self.degen = params['orbital degeneracies']*(1. + params['unpaired electrons'])


        def static_energies(self, volume, temperature):
                E_HP = self.params['HP phase'].method.molar_internal_energy(pressure = 0.,
                                                                            temperature = 0.,
                                                                            volume = volume,
                                                                            params = self.params['HP phase'].params)
                E_LP = self.params['LP phase'].method.molar_internal_energy(pressure = 0.,
                                                                            temperature = 0.,
                                                                            volume = volume,
                                                                            params = self.params['LP phase'].params)

                return np.array([E_HP, (E_LP + E_HP)/2. + 60000., E_LP]) # must be in order LS, IS, HS

        def static_pressures(self, volume, temperature):
                P_HP = self.params['HP phase'].method.pressure(temperature = 0.,
                                                               volume = volume,
                                                               params = self.params['HP phase'].params)
                P_LP = self.params['LP phase'].method.pressure(temperature = 0.,
                                                               volume = volume,
                                                               params = self.params['LP phase'].params)

                return np.array([P_HP, (P_LP + P_HP)/2., P_LP]) # must be in order LS, IS, HS

        def thermal_pressures(self, volume, temperature):
            Pth_FeO = (self.params['LP phase'].method.pressure(temperature = temperature,
                                                               volume = volume,
                                                               params = self.params['LP phase'].params)
                       - self.params['LP phase'].method.pressure(temperature = 0.,
                                                                 volume = volume,
                                                                 params = self.params['LP phase'].params))
            Pth_MgO = (self.params['Mg phase'].method.pressure(temperature = temperature,
                                                               volume = volume,
                                                               params = self.params['Mg phase'].params)
                       - self.params['Mg phase'].method.pressure(temperature = 0.,
                                                                 volume = volume,
                                                                 params = self.params['Mg phase'].params))
            return np.array([Pth_FeO, Pth_MgO])


        def Z(self, volume, temperature):
            Es = self.static_energies(volume, temperature)
            beta = 1./(burnman.constants.gas_constant*temperature)
            sumstates = np.sum(self.degen*np.exp(-beta*Es))
            return sumstates

        def p_expectation(self, volume, temperature):
            Es = self.static_energies(volume, temperature)
            beta = 1./(burnman.constants.gas_constant*temperature)
            prps = self.degen*np.exp(-beta*Es)
            return prps / self.Z(volume, temperature)

        def e_expectation(self, volume, temperature):
            Es = self.static_energies(volume, temperature)
            beta = 1./(burnman.constants.gas_constant*temperature)
            sumenergies = np.sum(self.degen * Es * np.exp(-beta*Es))
            return sumenergies / self.Z(volume, temperature)

        def pressure_expectation(self, volume, temperature):
            Es = self.static_energies(volume, temperature)
            Ps = self.static_pressures(volume, temperature)
            beta = 1./(burnman.constants.gas_constant*temperature)
            sumpressures = np.sum(self.degen * Ps * np.exp(-beta*Es))
            return sumpressures / self.Z(volume, temperature)

        def s_expectation(self, volume, temperature):
            pp = self.p_expectation(volume, temperature)
            return -np.sum(burnman.constants.gas_constant*pp*np.log(pp/self.degen))

        def pressure(self, volume, temperature):
            Pth_FeO, Pth_MgO = self.thermal_pressures(volume, temperature)
            P_FeO = self.pressure_expectation(volume, temperature) + Pth_FeO
            P_MgO = self.params['Mg phase'].method.pressure(temperature = 0.,
                                                            volume = volume,
                                                            params = self.params['Mg phase'].params) + Pth_MgO

            return P_FeO*self.params['n'] + P_MgO*(1. - params['n'])


        def volume(self, pressure, temperature):
            func = lambda vol: self.pressure(vol, temperature) - pressure
            try:
                sol = burnman.tools.bracket(func, self.params['LP phase'].params['V_0'],
                                            1.e-2 * self.params['LP phase'].params['V_0'])
            except:
                raise ValueError(
                    'Cannot find a volume, perhaps you are outside of the range of validity for the equation of state?')
            return brentq(func, sol[0], sol[1])



    params = {'LP phase': high_spin_wuestite(),
              'HP phase': low_spin_wuestite(),
              'Mg phase': burnman.minerals.SLB_2011.periclase(),
              'n': 0.35,
              'LP index': 2,
              'HP index': 0,
              'spin state names': ['LS', 'IS', 'HS'],
              'spin state energies': np.array([0., 1., 2.]),  # LS, IS, HS
              'orbital degeneracies': np.array([1., 1., 1.]),
              'unpaired electrons': np.array([0., 2., 4.])}

    fig, ax = plt.subplots(1, 3)

    dat = np.loadtxt('fp35.dat')
    ax[0].scatter(dat[:, 0], dat[:, 2]*6.022/4./10.)
    dat = np.loadtxt('fp35_vol.dat')
    ax[0].plot(dat[:, 0], dat[:, 1]*6.022/4./10., linestyle=':')


    dat = np.loadtxt('Komabayashi_2010.dat')
    ax[0].scatter(dat[:, 2], dat[:, 6]*6.022/4./10.)

    Pmin, Pmax = [1.e9, 150.e9]
    n_P = 101
    pressures = np.linspace(Pmin, Pmax, n_P)
    volumes = np.empty_like(pressures)
    volumes2 = np.empty_like(pressures)
    volumes3 = np.empty_like(pressures)

    energies = np.empty_like(pressures)
    energies2 = np.empty_like(pressures)
    energies3 = np.empty_like(pressures)
    oms = np.empty_like(pressures)

    prps = np.empty((n_P, len(params['spin state energies'])))

    for n in [0.001, 0.2, 0.4, 0.6, 0.8, 0.999]:
        params['n'] = n
        fper = SpinEOS(params)

        for T in [300.]: # , 800., 1300., 1800.]:

            for i, p in enumerate(pressures):
                print(p/1.e9)
                volumes[i] = fper.volume(p, T)
                #fper.params['LP phase'].set_state(p, T)
                #fper.params['HP phase'].set_state(p, T)

                prps[i] = fper.p_expectation(volumes[i], T)

                se = fper.static_energies(volumes[i], T)
                oms[i] = se[2] - se[0]
                #volumes2[i] = fper.params['LP phase'].V
                #volumes3[i] = fper.params['HP phase'].V

            # dat = np.loadtxt('fp35_vll.dat')
            # ax[0].scatter(dat[:,0], dat[:,1]*6.022/4./10.)
            # dat = np.loadtxt('fp35_vlh.dat')
            # ax[0].scatter(dat[:,0], dat[:,1]*6.022/4./10.)

            ax[0].plot(pressures/1.e9, volumes*1.e6)
            #ax[0].plot(pressures/1.e9, volumes2*1.e6, linestyle=':')
            #ax[0].plot(pressures/1.e9, volumes3*1.e6)

            ax[2].plot(volumes*1.e6, oms, label=T)

            ax[1].plot(pressures/1.e9, prps[:, 0], label=f'{fper.params["spin state names"][0]} at {T} K')
            ax[1].plot(pressures/1.e9, prps[:, 1], label=f'{fper.params["spin state names"][1]} at {T} K and n={n}')
            ax[1].plot(pressures/1.e9, prps[:, 2], label=f'{fper.params["spin state names"][2]} at {T} K')
    ax[1].legend()
    ax[2].legend()
    #ax[0].set_xlim(Pmin/1.e9, Pmax/1.e9)
    plt.show()
