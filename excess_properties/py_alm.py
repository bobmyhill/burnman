# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
Pyrope-Almandine "ideal" solution (where ideality is in Helmholtz free energy)
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
from burnman import minerals

from scipy.optimize import brentq


py = minerals.SLB_2011.pyrope()
alm = minerals.SLB_2011.almandine()

#py = minerals.HP_2011_ds62.py()
#alm = minerals.HP_2011_ds62.alm()

run_0 = True # 1st order approximation of pyrope-almandine elastic excesses
run_1 = False # excess volumes and entropy as a function of pressure at x_py = 0.5
run_2 = True # excess volumes as a function of composition


# 1st order approximation of pyrope-almandine elastic excesses
if run_0 == True:
    temperatures = np.linspace(10., 2000., 501)
    pressures = [1.e5]*len(temperatures)
    plt.plot(temperatures, py.evaluate(['V'], pressures, temperatures)[0], label=py.name)
    plt.plot(temperatures, alm.evaluate(['V'], pressures, temperatures)[0], label=alm.name)
    plt.legend(loc='lower left')
    plt.show()

    pressures = np.linspace(1.e5, 10.e9, 501)
    for T in [300., 1500.]:
        temperatures = [T]*len(pressures)
        
        plt.plot(pressures, py.evaluate(['V'], pressures, temperatures)[0], label=py.name+str(T))
        plt.plot(pressures, alm.evaluate(['V'], pressures, temperatures)[0], label=alm.name+str(T))
    plt.legend(loc='lower left')
    plt.show()

    
    py.set_state(1.e5, 300.)
    alm.set_state(1.e5, 300.)
    
    V_m = 0.5*(py.V + alm.V)
    K_m = 0.5*(py.K_T + alm.K_T)
    p_m = 0.5*V_m*(py.K_T/py.V + alm.K_T/alm.V) - K_m
    V_xs = p_m*V_m/K_m
    S_xs = 0.5*( py.alpha*py.K_T + alm.alpha*alm.K_T) * V_xs + 0.5*( py.alpha*py.K_T*(V_m - py.V) + alm.alpha*alm.K_T*(V_m - alm.V))     
    
    print(V_xs, S_xs)

# 1) excess volumes and entropy as a function of pressure at x_py = 0.5
if run_1 == True:
    x_py = 0.5
    
    volumes = np.linspace(0.8*py.params['V_0'], 1.0*alm.params['V_0'], 101)
    pressures = np.empty_like(volumes)
    entropies = np.empty_like(volumes)
    gibbs = np.empty_like(volumes)
    energies = np.empty_like(volumes)
    
    fig = plt.figure()
    ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
    
    for temperature in [300., 1000., 1500.]:
        temperatures = [temperature]*len(pressures)
        
        for i, volume in enumerate(volumes):
            p_py = py.method.pressure(temperature, volume, py.params)
            p_alm = alm.method.pressure(temperature, volume, alm.params)
            S_py = py.method.entropy(p_py, temperature, volume, py.params)
            S_alm = alm.method.entropy(p_alm, temperature, volume, alm.params)
            F_py = py.method.helmholtz_free_energy(p_py, temperature, volume, py.params)
            F_alm = alm.method.helmholtz_free_energy(p_alm, temperature, volume, alm.params)
            
            pressures[i] = x_py*p_py + (1. - x_py)*p_alm
            entropies[i] = x_py*S_py + (1. - x_py)*S_alm
            F = (x_py*F_py + (1. - x_py)*F_alm)
            
            gibbs[i] = F + pressures[i]*volume
            energies[i] = F + temperature*entropies[i]
            
        Gpy, Epy, Spy, Vpy = py.evaluate(['gibbs', 'internal_energy', 'S', 'V'], pressures, temperatures)
        Galm, Ealm, Salm, Valm = alm.evaluate(['gibbs', 'internal_energy', 'S', 'V'], pressures, temperatures)
        
        ax[0].plot(pressures/1.e9, volumes - (x_py*Vpy + (1 - x_py)*Valm), label='{0} K'.format(temperature))
        ax[1].plot(pressures/1.e9, entropies - (x_py*Spy + (1 - x_py)*Salm), label='{0} K'.format(temperature))
        ax[2].plot(pressures/1.e9, energies - (x_py*Epy + (1 - x_py)*Ealm), label='{0} K'.format(temperature))
        ax[3].plot(pressures/1.e9, gibbs - (x_py*Gpy + (1 - x_py)*Galm), label='{0} K'.format(temperature))



    ax[0].set_ylabel('$V_{excess}$ (m$^3$/mol)')
    ax[1].set_ylabel('$S_{excess}$ (J/K/mol)')
    ax[2].set_ylabel('$E_{excess}$ (J/mol)')
    ax[3].set_ylabel('$Gibbs_{excess}$ (J/mol)')
    for i in range(0, 4):
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
    
    def _deltaV(pressure, volume, temperature, m):
        m.set_state(pressure, temperature)
        return volume - m.V
    
    
    def _deltaP(volume, pressure, temperature, x_py):
        p_py = brentq(_deltaV,
                       py.method.pressure(temperature, volume, py.params) - 1.e9,
                       py.method.pressure(temperature, volume, py.params) + 1.e9,
                       args=(volume, temperature, py))
        p_alm = brentq(_deltaV,
                       alm.method.pressure(temperature, volume, alm.params) - 1.e9,
                       alm.method.pressure(temperature, volume, alm.params) + 1.e9,
                       args=(volume, temperature, alm))
        return pressure - (x_py*p_py + (1. - x_py)*p_alm)
    

    
    pressure = 1.e5
    temperature = 300.
    cluster_size = 3.
    F_disordered_mixing = np.empty_like(compositions)

    
    py.set_state(pressure, temperature)
    alm.set_state(pressure, temperature)
    for i, x_py in enumerate(compositions):
        volume = brentq(_deltaP, 0.9*py.V, 1.1*alm.V, args=(pressure, temperature, x_py))
        p_py = brentq(_deltaV,
                       py.method.pressure(temperature, volume, py.params) - 1.e9,
                       py.method.pressure(temperature, volume, py.params) + 1.e9,
                       args=(volume, temperature, py))
        p_alm = brentq(_deltaV,
                       alm.method.pressure(temperature, volume, alm.params) - 1.e9,
                       alm.method.pressure(temperature, volume, alm.params) + 1.e9,
                       args=(volume, temperature, alm))
        py.set_state(p_py, temperature)
        alm.set_state(p_alm, temperature)
        F_disordered_mixing[i] = ( (x_py*py.helmholtz + (1. - x_py)*alm.helmholtz) *
                                   (cluster_size - 1.)/cluster_size )

    F_disordered_mixing -= F_disordered_mixing[-1]*compositions + F_disordered_mixing[0]*(1. - compositions)
    
    pressure = 1.e5
    for temperature in [300., 1000., 1500.]:
        
        for i, x_py in enumerate(compositions):
            
            py.set_state(pressure, temperature)
            alm.set_state(pressure, temperature)
            volumes[i] = brentq(_deltaP, 0.9*py.V, 1.1*alm.V, args=(pressure, temperature, x_py))
            p_py = brentq(_deltaV,
                          py.method.pressure(temperature, volumes[i], py.params) - 1.e8,
                          py.method.pressure(temperature, volumes[i], py.params) + 1.e8,
                          args=(volumes[i], temperature, py))
            p_alm = brentq(_deltaV,
                           alm.method.pressure(temperature, volumes[i], alm.params) - 1.e6,
                           alm.method.pressure(temperature, volumes[i], alm.params) + 1.e6,
                           args=(volumes[i], temperature, alm))
        
            py.set_state(p_py, temperature)
            alm.set_state(p_alm, temperature)
            
            entropies[i] = x_py*py.S + (1. - x_py)*alm.S
            F = (x_py*py.helmholtz + (1. - x_py)*alm.helmholtz) - F_disordered_mixing[i]
            gibbs[i] = F + pressure*volumes[i]
            energies[i] = F + temperature*entropies[i]

        
        py.set_state(pressure, temperature)
        alm.set_state(pressure, temperature)
        
        ax[0].plot(1. - compositions, volumes - (compositions*py.V + (1 - compositions)*alm.V), label='{0} K'.format(temperature))
        ax[1].plot(1. - compositions, entropies - (compositions*py.S + (1 - compositions)*alm.S), label='{0} K'.format(temperature))
        ax[2].plot(1. - compositions, energies - (compositions*py.internal_energy + (1 - compositions)*alm.internal_energy), label='{0} K'.format(temperature))
        ax[3].plot(1. - compositions, gibbs - (compositions*py.gibbs + (1 - compositions)*alm.gibbs), label='{0} K'.format(temperature))



    ax[0].set_ylabel('$V_{excess}$ (m$^3$/mol)')
    ax[1].set_ylabel('$S_{excess}$ (J/K/mol)')
    ax[2].set_ylabel('$E_{excess}$ (J/mol)')
    ax[3].set_ylabel('$Gibbs_{excess}$ (J/mol)')
    for i in range(0, 4):
        ax[i].set_xlim(0., 1.)
        ax[i].set_xlabel('$x$ (almandine)')
        ax[i].legend(loc='lower center')
    plt.show()



