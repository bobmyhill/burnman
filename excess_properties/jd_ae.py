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

palettes = ['seaborn-darkgrid', 'seaborn-notebook', 'classic', 'seaborn-ticks', 'grayscale',
            'bmh', 'seaborn-talk', 'dark_background', 'ggplot', 'fivethirtyeight',
            '_classic_test', 'seaborn-colorblind', 'seaborn-deep', 'seaborn-whitegrid', 'seaborn',
            'seaborn-poster', 'seaborn-bright', 'seaborn-muted', 'seaborn-paper', 'seaborn-white',
            'seaborn-pastel', 'seaborn-dark', 'seaborn-dark-palette']

plt.style.use(palettes[6])
plt.rcParams['figure.figsize'] = 12, 6 # inches
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'


# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import minerals
from excess_modelling import *

from scipy.optimize import brentq



#jd_params = {'name':'jadeite',
#             'V_0': 60.561e-6,
#             'K_0': 134.0e9,
#             'Kprime_0': 4.5}
jd_params = {'name':'jadeite',
             'V_0': 60.561e-6,
             'K_0': 134.5e9,
             'Kprime_0': 4.5}
ae_params = {'name':'aegirine',
             'V_0': 64.626e-6,
             'K_0': 116.1e9,
             'Kprime_0': 4.5}
ae026_params = {'name':'ae026',
                'V_0': 64.626e-6,
                'K_0': 116.1e9,
                'Kprime_0': 4.49}
ae065_params = {'name':'ae065',
                'V_0': 64.626e-6,
                'K_0': 116.1e9,
                'Kprime_0': 4.48}


for params in [jd_params, ae_params, ae026_params, ae065_params]:
    params['equation_of_state'] = 'vinet'
    params['P_0'] = 1.e5
    params['T_0'] = 300.

jd = burnman.Mineral(params = jd_params)
ae = burnman.Mineral(params = ae_params)
ae026 = burnman.Mineral(params = ae026_params)
ae065 = burnman.Mineral(params = ae065_params)

# First, let's fit the endmember data:
for (m, f) in [(ae, 'data/jd_ae_PV_data/ae100_PV.dat'),
               (jd, 'data/jd_ae_PV_data/ae000_PV.dat'),
               (ae026, 'data/jd_ae_PV_data/ae026_PV.dat'),
               (ae065, 'data/jd_ae_PV_data/ae065_PV.dat')]:
    P, Perr, V, Verr= np.loadtxt(f, unpack=True, comments='%')
    P = P*1.e9
    T = np.array([300.]*len(P))
    Perr = Perr*1.e9
    Terr = np.array([1.]*len(Perr))
    
    params=['V_0',  'K_0', 'Kprime_0']
    params=['V_0',  'K_0']
    PTV = np.array([P, T, V]).T
    
    nul = 0.*PTV.T[0]
    PTV_covariances = np.array([[Perr*Perr, nul, nul],
                                [nul, Terr*Terr, nul],
                                [nul, nul, Verr*Verr]]).T
    
    fitted_eos = burnman.eos_fitting.fit_PTV_data(m, params, PTV, PTV_covariances, verbose=False)

    # Print the optimized parameters
    print('Optimized equation of state for {0}:'.format(m.name))
    burnman.tools.pretty_print_values(fitted_eos.popt, fitted_eos.pcov, fitted_eos.fit_params)

    


run_0 = True # solid solution volumes at HP


# 0) excess volumes and entropy as a function of pressure at x_jd = 0.5
if run_0 == True:

    fig = plt.figure()
    ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]


    temperature = 300.
    pressures = np.linspace(0., 10.e9, 101)
    temperatures = [temperature]*len(pressures)
    
    V_jd = jd.evaluate(['V'], pressures, temperatures)[0]
    V_ae = ae.evaluate(['V'], pressures, temperatures)[0]

    colors = [next(ax[0]._get_lines.prop_cycler)['color'] for i in range(10)]
    
    #ax[0].plot(pressures/1.e9, V_ae*1.e6, label='$jd_{{{0:.0f}}}ae_{{{1:.0f}}}$'.format(0, 100))

    files = ['data/jd_ae_PV_data/ae100_PV.dat', 'data/jd_ae_PV_data/ae065_PV.dat',
             'data/jd_ae_PV_data/ae026_PV.dat', 'data/jd_ae_PV_data/ae000_PV.dat']
    compositions = [0., 0.35, 0.74, 1.]
             
    for (j, p_xs, linestyle) in [(0, 0.e9, '-'),
                      (1, 0., '--'),
                      (2, 0., '--'),
                      (1, -0.300e9, '-'),
                      (2, -0.195e9, '-'),
                      (3, 0.e9, '-')]:

        f = files[j]
        x_jd = compositions[j]
        
        volumes = np.empty_like(pressures)
        K_Ts = np.empty_like(pressures)
        for i, pressure in enumerate(pressures):
            s = solution(pressure, temperature,
                         F_xs = 0., p_xs = p_xs,
                         x_a = x_jd, a = jd, b = ae)
            volumes[i] = s.V
            K_Ts[i] = s.K_T
            
        print(x_jd, volumes[0]*1.e6, K_Ts[0]/1.e9, (K_Ts[1] - K_Ts[0])/(pressures[1] - pressures[0]))

        label = 'jd$_{{{0:.0f}}}$ae$_{{{1:.0f}}}$, $P_{{xs}}$={2} GPa'.format(x_jd*100, (1. - x_jd)*100, p_xs/1.e9)
        ax[0].plot(pressures/1.e9 , volumes*1.e6, label=label, color=colors[j], linestyle=linestyle)
        ax[1].plot(pressures/1.e9, (volumes - (x_jd*V_jd + (1. - x_jd)*V_ae))*1.e6, label=label, color=colors[j], linestyle=linestyle)

        data = np.loadtxt(f, unpack=True, comments="%")

        
        V_jd2 = jd.evaluate(['V'], data[0]*1.e9, [temperature]*len(data[0]))[0]
        V_ae2 = ae.evaluate(['V'], data[0]*1.e9, [temperature]*len(data[0]))[0]
        ax[0].scatter(data[0], data[2]*1.e6, color=colors[j])
        ax[1].scatter(data[0], (data[2] - (x_jd*V_jd2 + (1. - x_jd)*V_ae2))*1.e6, color=colors[j])
        #ax[1].errorbar(data[0], (data[2] - (x_jd*V_jd2 + (1. - x_jd)*V_ae2))*1.e6,
        #               xerr=data[1], yerr=data[3]*1.e6, linestyle='None')
    
    ax[0].set_ylim(55,65)
    ax[1].set_ylim(-0.23,0.03)
    ax[0].set_ylabel('$V$ (cm$^3$/mol)')
    ax[1].set_ylabel('$V_{excess}$ (cm$^3$/mol)')

    
    for i in range(0, 2):
        ax[i].set_xlim(0,10)
        ax[i].set_xlabel('Pressure (GPa)')
        ax[i].legend(loc='best')
        
    fig.tight_layout()
    fig.savefig("jd_ae_volumes.pdf", bbox_inches='tight', dpi=100)
    plt.show()

