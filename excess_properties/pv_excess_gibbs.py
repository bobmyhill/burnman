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

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman.minerals import KMFBZ_2017, SLB_2011, HP_2011_ds62

from scipy.optimize import brentq


MgSiO3_bdg = KMFBZ_2017.mg_si_perovskite()
KMFBZ_FeAlO3_bdg = KMFBZ_2017.fe_al_perovskite()
params = {'equation_of_state': 'rkprime',
          'V_0': MgSiO3_bdg.params['V_0'],
          'K_0': MgSiO3_bdg.params['K_0'],
          'Kprime_0': 4.0,
          'Kprime_inf': 1.6, # best fitting value of ~1.4 is less than the thermodynamic bound of 1.6, and still doesn't (quite) fit the data
          'molar_mass': MgSiO3_bdg.params['molar_mass'],
          'n': MgSiO3_bdg.params['n'],
          'formula': MgSiO3_bdg.params['formula']}
py = burnman.Mineral(params = params)

params = {'equation_of_state': 'rkprime',
          'V_0': 2.79e-5,
          'K_0': 215.e9,
          'Kprime_0': 2.0,
          'Kprime_inf': 1.6, # best fitting value of ~1.4 is less than the thermodynamic bound of 1.6, and still doesn't (quite) fit the data
          'molar_mass': KMFBZ_FeAlO3_bdg.params['molar_mass'],
          'n': KMFBZ_FeAlO3_bdg.params['n'],
          'formula': KMFBZ_FeAlO3_bdg.params['formula']}
gr = burnman.Mineral(params = params)


run_0 = False # 1st order approximation of pyrope-grossular elastic excesses
run_1 = True # excess volumes and entropy as a function of pressure at x_py = 0.5
run_2 = True # excess volumes as a function of composition
run_3 = True # Experimental volumes (py-gr from Newton et al. and alm-gr from Cressey et al.)


# 1) excess volumes and energies as a function of pressure at x_MgSiO3_bdg2 = 0.5
if run_1 == True:
    x_py = 0.5
    
    volumes = np.linspace(0.8*py.params['V_0'], 1.05*py.params['V_0'], 101)
    pressures = np.empty_like(volumes)
    gibbs = np.empty_like(volumes)
    energies = np.empty_like(volumes)
    
    fig = plt.figure()
    ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

    temperature = 300.
    temperatures = [temperature]*len(pressures)
        
    for i, volume in enumerate(volumes):
        p_py = py.method.pressure(temperature, volume, py.params)
        p_gr = gr.method.pressure(temperature, volume, gr.params)
        F_py = py.method.internal_energy(p_py, temperature, volume, py.params)
        F_gr = gr.method.internal_energy(p_gr, temperature, volume, gr.params)
        
        pressures[i] = x_py*p_py + (1. - x_py)*p_gr
        F = (x_py*F_py + (1. - x_py)*F_gr)
        
        gibbs[i] = F + pressures[i]*volume
        energies[i] = F
        
        Gpy, Epy, Vpy = py.evaluate(['gibbs', 'internal_energy', 'V'], pressures, temperatures)
        Ggr, Egr, Vgr = gr.evaluate(['gibbs', 'internal_energy', 'V'], pressures, temperatures)
        
    ax[0].plot(pressures/1.e9, volumes - (x_py*Vpy + (1 - x_py)*Vgr), label='{0} K'.format(temperature))
    ax[1].plot(pressures/1.e9, energies - (x_py*Epy + (1 - x_py)*Egr), label='{0} K'.format(temperature))
    ax[2].plot(pressures/1.e9, gibbs - (x_py*Gpy + (1 - x_py)*Ggr), label='{0} K'.format(temperature))



    ax[0].set_ylabel('$V_{excess}$ (m$^3$/mol)')
    ax[1].set_ylabel('$E_{excess}$ (J/mol)')
    ax[2].set_ylabel('$Gibbs_{excess}$ (J/mol)')
    for i in range(0, 3):
        ax[i].set_xlabel('Pressure (GPa)')
        ax[i].legend(loc='lower left')
    plt.show()


# 2) excess volumes as a function of composition
if run_2 == True:
    compositions = np.linspace(0., 1., 101)
    volumes = np.empty_like(compositions)
    entropies = np.empty_like(compositions)
    gibbs = np.empty_like(compositions)
    energies = np.empty_like(compositions)
    
    
    fig = plt.figure()
    ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
    
    
    def _deltaP(volume, pressure, temperature, x_py):
        p_py = py.method.pressure(temperature, volume, py.params)
        p_gr = gr.method.pressure(temperature, volume, gr.params)
        return pressure - (x_py*p_py + (1. - x_py)*p_gr)
    

    
    pressure = 1.e5
    temperature = 300.
    cluster_size = 3.
    F_disordered_mixing = np.empty_like(compositions)

    
    py.set_state(pressure, temperature)
    gr.set_state(pressure, temperature)
    for i, x_py in enumerate(compositions):
        volume = brentq(_deltaP, 0.9*py.V, 1.1*gr.V, args=(pressure, temperature, x_py))
        p_py = py.method.pressure(temperature, volume, py.params)
        p_gr = gr.method.pressure(temperature, volume, gr.params)
        F_py = py.method.helmholtz_free_energy(p_py, temperature, volume, py.params)
        F_gr = gr.method.helmholtz_free_energy(p_gr, temperature, volume, gr.params)
        F_disordered_mixing[i] = (x_py*F_py + (1. - x_py)*F_gr)*(cluster_size - 1.)/cluster_size

    F_disordered_mixing -= F_disordered_mixing[-1]*compositions + F_disordered_mixing[0]*(1. - compositions)
    
    pressure = 1.e5
    for temperature in [300., 1000., 1500.]:
        
        py.set_state(pressure, temperature)
        gr.set_state(pressure, temperature)

        
        for i, x_py in enumerate(compositions):
            
            volumes[i] = brentq(_deltaP, 0.9*py.V, 1.1*gr.V, args=(pressure, temperature, x_py))
            
            p_py = py.method.pressure(temperature, volumes[i], py.params)
            p_gr = gr.method.pressure(temperature, volumes[i], gr.params)
            S_py = py.method.entropy(p_py, temperature, volumes[i], py.params)
            S_gr = gr.method.entropy(p_gr, temperature, volumes[i], gr.params)
            F_py = py.method.helmholtz_free_energy(p_py, temperature, volumes[i], py.params)
            F_gr = gr.method.helmholtz_free_energy(p_gr, temperature, volumes[i], gr.params)
            
            entropies[i] = x_py*S_py + (1. - x_py)*S_gr
            F = (x_py*F_py + (1. - x_py)*F_gr) - F_disordered_mixing[i]
            gibbs[i] = F + pressure*volumes[i]
            energies[i] = F + temperature*entropies[i]
        
        ax[0].plot(1. - compositions, volumes - (compositions*py.V + (1 - compositions)*gr.V), label='{0} K'.format(temperature))
        ax[1].plot(1. - compositions, entropies - (compositions*py.S + (1 - compositions)*gr.S), label='{0} K'.format(temperature))
        ax[2].plot(1. - compositions, energies - (compositions*py.internal_energy + (1 - compositions)*gr.internal_energy), label='{0} K'.format(temperature))
        ax[3].plot(1. - compositions, gibbs - (compositions*py.gibbs + (1 - compositions)*gr.gibbs), label='{0} K'.format(temperature))



    ax[0].set_ylabel('$V_{excess}$ (m$^3$/mol)')
    ax[1].set_ylabel('$S_{excess}$ (J/K/mol)')
    ax[2].set_ylabel('$E_{excess}$ (J/mol)')
    ax[3].set_ylabel('$Gibbs_{excess}$ (J/mol)')
    for i in range(0, 4):
        ax[i].set_xlim(0., 1.)
        ax[i].set_xlabel('$x$ (grossular)')
        ax[i].legend(loc='lower center')
    plt.show()



