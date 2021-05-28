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

if __name__ == "__main__":


    class fper_HS(Mineral):
        """
        MINUTI example (fp35, high spin)
        """
        def __init__(self):
            self.params = {
                'equation_of_state': 'mgd3',
                'V_0': 77.153e-30*burnman.constants.Avogadro/4.,
                'K_0': 160.44e9,
                'Kprime_0': 4.0510,
                'molar_mass': formula_mass({'Mg': 0.65, 'Fe': 0.35, 'O': 1.}),
                'n': 2,
                'Debye_0': 800.,
                'grueneisen_0': 1.45,
                'q_0': 0.8,
                'T_0': 300.}
            Mineral.__init__(self)


    class fper_LS(Mineral):

        """
        MINUTI example (fp35, low spin)
        """

        def __init__(self):
            self.params = {
                'equation_of_state': 'mgd3',
                'V_0': 73.7390e-30*burnman.constants.Avogadro/4.,
                'K_0': 171.43e9,
                'Kprime_0': 4.,
                'molar_mass': formula_mass({'Mg': 0.65, 'Fe': 0.35, 'O': 1.}),
                'n': 2,
                'Debye_0': 800.,
                'grueneisen_0': 1.45,
                'q_0': 0.8,
                'T_0': 300.}
            Mineral.__init__(self)


    class SpinEOS(object):
        def __init__(self, params):
            self.params = params

            self.degen = params['orbital degeneracies']*(1. + params['unpaired electrons'])

            self.e_HP = self.params['spin state energies'][self.params['HP index']]
            self.e_LP = self.params['spin state energies'][self.params['LP index']]

            self.params['LP phase'].set_state(self.params['P_tr'],
                                              self.params['T_0'])
            self.params['HP phase'].set_state(self.params['P_tr'],
                                              self.params['T_0'])

            self.V_tr = (self.params['LP phase'].V
                         + self.params['HP phase'].V)/2.

            self.delFVtr = (self.params['HP phase'].method.molar_internal_energy(pressure = 0.,
                                                                                 temperature = self.params['T_0'],
                                                                                 volume = self.V_tr,
                                                                                 params = self.params['HP phase'].params)
                            - self.params['LP phase'].method.molar_internal_energy(pressure = 0.,
                                                                                   temperature = self.params['T_0'],
                                                                                   volume = self.V_tr,
                                                                                   params = self.params['LP phase'].params))
            self.delFVtr -= (burnman.constants.gas_constant
                             * self.params['T_0'] / self.params['Z']
                             * np.log(self.degen[self.params['HP index']]
                                      / self.degen[self.params['LP index']]))

        def omega(self, volume, temperature):
                delFV = (self.params['HP phase'].method.helmholtz_free_energy(pressure = 0.,
                                                                              temperature = temperature,
                                                                              volume = volume,
                                                                              params = self.params['HP phase'].params)
                         - self.params['LP phase'].method.helmholtz_free_energy(pressure = 0.,
                                                                                temperature = temperature,
                                                                                volume = volume,
                                                                                params = self.params['LP phase'].params))

                return (1./(self.params['n'] * (self.e_HP - self.e_LP))
                        * (delFV - self.delFVtr))

        def Z(self, volume, temperature):
            om = self.omega(volume, temperature)
            beta = 1./(burnman.constants.gas_constant*temperature)
            sumstates = np.sum(self.degen*np.exp(-beta*self.params['spin state energies']*om))
            return sumstates

        def p_expectation(self, volume, temperature):
            om = self.omega(volume, temperature)
            beta = 1./(burnman.constants.gas_constant*temperature)
            prps = self.degen*np.exp(-beta*self.params['spin state energies']*om)
            return prps / self.Z(volume, temperature)

        def e_expectation(self, volume, temperature):
            om = self.omega(volume, temperature)
            beta = 1./(burnman.constants.gas_constant*temperature)
            sumenergies = np.sum(self.degen*self.params['spin state energies']
                                 * np.exp(-beta*self.params['spin state energies']*om))
            return sumenergies / self.Z(volume, temperature)

        def spin_pressure(self, volume, temperature):
            dP = (self.params['HP phase'].method.pressure(0., volume,
                                                          self.params['HP phase'].params)
                  - self.params['LP phase'].method.pressure(0., volume,
                                                            self.params['LP phase'].params))
            e_ext = self.e_expectation(volume, temperature)

            return -(1./(self.e_HP - self.e_LP)*dP*(self.e_LP - e_ext))

        def pressure(self, volume, temperature):
            p1 = self.params['LP phase'].method.pressure(temperature, volume,
                                                         self.params['LP phase'].params)
            p_spin = self.spin_pressure(volume, temperature)
            return p1 + p_spin # equation 25

        def volume(self, pressure, temperature):
            func = lambda vol: self.pressure(vol, temperature) - pressure
            try:
                sol = burnman.tools.bracket(func, self.params['LP phase'].params['V_0'],
                                            1.e-2 * self.params['LP phase'].params['V_0'])
            except:
                raise ValueError(
                    'Cannot find a volume, perhaps you are outside of the range of validity for the equation of state?')
            return brentq(func, sol[0], sol[1])



    params = {'LP phase': fper_HS(),
              'HP phase': fper_LS(),
              'n': 0.35,
              'Z': 4.,
              'LP index': 2,
              'HP index': 0,
              'T_0': 300.,
              'P_tr': 63.588e9,
              'spin state energies': np.array([0., 1., 2.]),  # LS, IS, HS
              'orbital degeneracies': np.array([1., 6., 3.]),
              'unpaired electrons': np.array([0., 2., 4.])}


    fper = SpinEOS(params)

    fig, ax = plt.subplots(1, 3)

    dat = np.loadtxt('fp35.dat')
    ax[0].scatter(dat[:, 0], dat[:, 2]*6.022/4./10.)
    dat = np.loadtxt('fp35_vol.dat')
    ax[0].scatter(dat[:, 0], dat[:, 1]*6.022/4./10.)
    dat = np.loadtxt('fp35_vol_2000K.dat')
    ax[0].scatter(dat[:, 0], dat[:, 1]*6.022/4./10.)

    Pmin, Pmax = [20.e9, 140.e9]
    pressures = np.linspace(Pmin, Pmax, 501)
    volumes = np.empty_like(pressures)
    volumes2 = np.empty_like(pressures)
    volumes3 = np.empty_like(pressures)
    oms = np.empty_like(pressures)
    prps = np.empty((501, 3))

    for T in [300., 2000.]:
        fper.params['LP phase'].set_state(fper.params['P_tr'], T)
        fper.params['HP phase'].set_state(fper.params['P_tr'], T)
        ax[0].scatter([fper.params['P_tr']/1.e9, fper.params['P_tr']/1.e9],
                      [fper.params['LP phase'].V*1.e6, fper.params['HP phase'].V*1.e6])

        for i, p in enumerate(pressures):
            volumes[i] = fper.volume(p, T)
            fper.params['LP phase'].set_state(p, T)
            fper.params['HP phase'].set_state(p, T)

            oms[i] = fper.omega(volumes[i], T)
            prps[i] = fper.p_expectation(volumes[i], T)

            volumes2[i] = fper.params['LP phase'].V
            volumes3[i] = fper.params['HP phase'].V

        # dat = np.loadtxt('fp35_vll.dat')
        # ax[0].scatter(dat[:,0], dat[:,1]*6.022/4./10.)
        # dat = np.loadtxt('fp35_vlh.dat')
        # ax[0].scatter(dat[:,0], dat[:,1]*6.022/4./10.)

        ax[0].plot(pressures/1.e9, volumes*1.e6)
        ax[0].plot(pressures/1.e9, volumes2*1.e6, linestyle=':')
        ax[0].scatter([fper.params['P_tr']/1.e9], [fper.V_tr*1.e6])
        #ax[0].plot(pressures/1.e9, volumes3*1.e6)


        ax[2].plot(volumes*1.e6, oms, label=T)

        #ax[1].plot(pressures/1.e9, prps[:, 0], label='LS')
        ax[1].plot(pressures/1.e9, prps[:, 1], label='IS')
        #ax[1].plot(pressures/1.e9, prps[:, 2], label='HS')
    ax[1].legend()
    ax[2].legend()
    #ax[0].set_xlim(Pmin/1.e9, Pmax/1.e9)
    plt.show()
