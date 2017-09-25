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

py = minerals.SLB_2011.pyrope()
alm = minerals.SLB_2011.almandine()
gr = minerals.SLB_2011.grossular()

g_ideal = burnman.SolidSolution(name = 'garnet',
                                solution_type = 'ideal',
                                endmembers = [[minerals.SLB_2011.pyrope(), '[Mg]3[Al][Al]Si3O12'],
                                              [minerals.SLB_2011.grossular(), '[Ca]3[Al][Al]Si3O12']])

g_nonideal = burnman.SolidSolution(name = 'garnet',
                                   solution_type = 'symmetric',
                                   endmembers = [[minerals.SLB_2011.pyrope(), '[Mg]3[Al][Al]Si3O12'],
                                                 [minerals.SLB_2011.grossular(), '[Ca]3[Al][Al]Si3O12']],
                                   energy_interaction = [[30.e3]],
                                   volume_interaction = [[1.2e-6]])


plt.style.use('seaborn-talk')

run_0 = False # 1st order approximation of pyrope-grossular elastic excesses
run_1 = False # excess volumes and entropy as a function of pressure at x_py = 0.5
run_2 = True # excess volumes as a function of composition
run_3 = False # Experimental entropies (from Dachs and Geiger)
run_4 = False # Experimental volumes (py-gr from Newton et al. and alm-gr from Cressey et al.)


# 1st order approximation of pyrope-grossular elastic excesses
if run_0 == True:
    py.set_state(1.e5, 300.)
    gr.set_state(1.e5, 300.)
    
    V_m = 0.5*(py.V + gr.V)
    K_m = 0.5*(py.K_T + gr.K_T)
    p_m = 0.5*V_m*(py.K_T/py.V + gr.K_T/gr.V) - K_m
    V_xs = p_m*V_m/K_m
    S_xs = 0.5*( py.alpha*py.K_T + gr.alpha*gr.K_T) * V_xs + 0.5*( py.alpha*py.K_T*(V_m - py.V) + gr.alpha*gr.K_T*(V_m - gr.V))     
    
    print(V_xs, S_xs)


def asymmetric_excess(p1, alpha1, w):
    p = np.array([1. - p1, p1])
    alpha = np.array([1., alpha1])
    phi = alpha*p/np.sum(alpha*p)
    return np.sum(alpha*p)*phi[0]*phi[1]*2.*w/(np.sum(alpha))


# 1) excess volumes and entropy as a function of pressure at x_py = 0.5
if run_1 == True:
    x_py = 0.5

    pressures = np.linspace(1.e5, 25.e9, 101)
    
    volumes = np.empty_like(pressures)
    entropies = np.empty_like(pressures)
    enthalpies = np.empty_like(pressures)
    gibbs = np.empty_like(pressures)
    
    plt.rcParams['figure.figsize'] = 12, 10 # inches
    fig = plt.figure()
    ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
    
    for temperature in [300., 1000., 1500.]:
        temperatures = [temperature]*len(pressures)
        
        for i, pressure in enumerate(pressures):
            s = solution(pressure, temperature,
                         F_xs = 0., p_xs = 0.,
                         x_a = x_py, a = py, b = gr, cluster_size=3.)
            volumes[i] = s.V
            entropies[i] = s.S
            gibbs[i] = s.gibbs
            enthalpies[i] = s.H
            
        Gpy, Hpy, Spy, Vpy = py.evaluate(['gibbs', 'H', 'S', 'V'], pressures, temperatures)
        Ggr, Hgr, Sgr, Vgr = gr.evaluate(['gibbs', 'H', 'S', 'V'], pressures, temperatures)
        
        ax[0].plot(pressures/1.e9, (volumes - (x_py*Vpy + (1 - x_py)*Vgr))*1.e6, label='{0} K'.format(temperature))
        ax[1].plot(pressures/1.e9, entropies - (x_py*Spy + (1 - x_py)*Sgr), label='{0} K'.format(temperature))
        ax[2].plot(pressures/1.e9, (enthalpies - (x_py*Hpy + (1 - x_py)*Hgr))*1.e-3, label='{0} K'.format(temperature))
        ax[3].plot(pressures/1.e9, (gibbs - (x_py*Gpy + (1 - x_py)*Ggr))*1.e-3, label='{0} K'.format(temperature))

    ax[0].plot(pressures/1.e9,
               [asymmetric_excess(1. - x_py, 2.7, 1.64) for P in pressures],
               label='Green et al., 2012', linestyle='--')
    
    ax[3].plot(pressures/1.e9,
               [asymmetric_excess(1. - x_py, 2.7, 30.1+0.164*P) for P in pressures/1.e8],
               label='Green et al., 2012', linestyle='--')

    ax[0].set_ylabel('$V_{excess}$ (cm$^3$/mol)')
    ax[1].set_ylabel('$S_{excess}$ (J/K/mol)')
    ax[2].set_ylabel('$\mathcal{H}_{excess}$ (kJ/mol)')
    ax[3].set_ylabel('$\mathcal{G}_{excess}$ (kJ/mol)')
    for i in range(0, 4):
        ax[i].set_xlim(0, 25)
        ax[i].set_xlabel('Pressure (GPa)')
        ax[i].legend(loc='best')
    fig.tight_layout()
    fig.savefig("py_gr_excesses.pdf", bbox_inches='tight', dpi=100)
    plt.show()


# 2) excess volumes as a function of composition
if run_2 == True:
    
    pressure = 1.e5
    temperature = 300.
    cluster_size = 3.
    
    compositions = np.linspace(0., 1., 101)
    volumes = np.empty_like(compositions)
    entropies = np.empty_like(compositions)
    gibbs = np.empty_like(compositions)
    energies = np.empty_like(compositions)
    enthalpies = np.empty_like(compositions)
    bulk_sound_velocities = np.empty_like(compositions)
    bulk_sound_velocities_ideal = np.empty_like(compositions)
    bulk_sound_velocities_nonideal = np.empty_like(compositions)

    
    dFdxVs = np.empty_like(compositions)

    
    plt.rcParams['figure.figsize'] = 12, 8 # inches
    fig = plt.figure()
    ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
    fig2 = plt.figure()
    ax2 = [fig2.add_subplot(1, 1, 1)]
    colors = [next(ax[0]._get_lines.prop_cycler)['color'] for i in range(10)]

    x_pys = np.array([0., 0.2, 0.4, 0.6, 0.8, 1.0])
    Vs = np.array([125.23e-6, 123.19e-6, 120.08e-6, 118.82e-6, 115.82e-6, 112.87e-6])
    KTs = np.array([169.7e9, 158.3e9, 160.7e9, 161.8e9, 159.1e9, 169.2e9])
    KSs = np.empty_like(KTs)
    
    for i, x_py in enumerate(x_pys):
        g_ideal.set_composition([x_py, 1. - x_py])
        g_ideal.set_state(1.e5, 300.)
        KSs[i] = KTs[i] - g_ideal.K_T + g_ideal.K_S
        
    molar_masses = x_pys*py.params['molar_mass'] + (1. - x_pys)*gr.params['molar_mass']
    ax2[0].scatter(1. - x_pys, np.sqrt(KSs*Vs/molar_masses)/1.e3, label='Du et al. (2015)')


    quadratic = lambda x, a, b, c: a*x + b*(1. - x) + c*x*(1. - x)

    x_py_NCK, a_NCK = np.loadtxt('data/py_gr_V_NCK.dat', unpack=True)
    a_err_NCK = np.array([0.001]*len(a_NCK))
    Z = 8.
    volumes_NCK = np.power(a_NCK*1.e-10, 3.)*burnman.constants.Avogadro/Z
    volume_err_NCK = 3.*volumes_NCK*a_err_NCK/a_NCK
    
    popt, pcov = curve_fit(quadratic, xdata = x_py_NCK, ydata = volumes_NCK, sigma=volume_err_NCK)
    
    ax[0].errorbar(1. - x_py_NCK, (volumes_NCK - quadratic(x_py_NCK, popt[0], popt[1], 0.))*1.e6, yerr=volume_err_NCK*1.e6, linestyle='None', color=colors[0])
    ax[0].scatter(1. - x_py_NCK, (volumes_NCK - quadratic(x_py_NCK, popt[0], popt[1], 0.))*1.e6, color=colors[0])
    ax[0].errorbar([-1., -0.9], [0., 0.], fmt='o', markersize=9, label='298 K  (Newton et al., 1977)', color=colors[0])

    
    S_DG2006 = np.loadtxt('data/py_gr_S298.dat', unpack=True)
    popt, pcov = curve_fit(quadratic, xdata = S_DG2006[0], ydata = S_DG2006[1], sigma = S_DG2006[2])
    
    #ax[1].fill_between(np.linspace(0., 1., 101),
    #                 quadratic(np.linspace(0., 1., 101), 0., 0., popt[2] - np.sqrt(pcov[2][2])),
    #                 quadratic(np.linspace(0., 1., 101), 0., 0., popt[2] + np.sqrt(pcov[2][2])),
    #                 facecolor=colors[0], lw=0, alpha=0.3, interpolate=False)
    
    ax[1].errorbar(S_DG2006[0], S_DG2006[1] - quadratic(S_DG2006[0], popt[0], popt[1], 0.), yerr=S_DG2006[2], linestyle='None', color=colors[0])
    ax[1].scatter(S_DG2006[0], S_DG2006[1] - quadratic(S_DG2006[0], popt[0], popt[1], 0.),  color=colors[0])
    #ax[1].plot(np.linspace(0., 1., 101), quadratic(np.linspace(0., 1., 101), 0., 0., popt[2]), linestyle='--', color=colors[0])
    ax[1].errorbar([-1., -0.9], [0., 0.], fmt='o', markersize=9, label='298 K (Dachs and Geiger, 2006)', color=colors[0])

    '''
    cubic = lambda x, a, b, c, d: a*x + b*(1. - x) + c*x*(1. - x) + d*x*x*(1. - x)
    popt, pcov = curve_fit(cubic, xdata = S_DG2006[0], ydata = S_DG2006[1], sigma = S_DG2006[2])
    ax[1].errorbar(S_DG2006[0], S_DG2006[1] - cubic(S_DG2006[0], popt[0], popt[1], 0., 0.), yerr=S_DG2006[2], linestyle='None', fmt='o', markersize=8, label='298 K (Dachs and Geiger, 2006)', color=colors[0])
    ax[1].plot(np.linspace(0., 1., 101), cubic(np.linspace(0., 1., 101), 0., 0., popt[2], popt[3]), label='298 K (Dachs and Geiger, 2006)', color=colors[0])
    '''
    
    ax[1].errorbar(np.array([0., 0.4, 1.0]),
                   np.array([266.27, 268.32, 260.12]) -
                   quadratic(np.array([0., 0.4, 1.0]), popt[0], popt[1], 0.),
                   yerr=[0.266, 0.268, 0.260], linestyle='None', fmt='*', markersize=16, label='298 K (Haselton and Westrum, 1980)', color=colors[0]) 


    # and the enthalpy of solution (in kcal/gfw = kcal/mol)
    x_gr, H_NCK, H_err_NCK = np.loadtxt('data/py_gr_H970K.dat', unpack=True)
    H_NCK *= -4184. # negative because Hsolution_xs = -H_xs
    H_err_NCK *= -4184.
    popt, pcov = curve_fit(quadratic, xdata = x_gr, ydata = H_NCK, sigma = H_err_NCK)
    
    ax[2].errorbar(x_gr, (H_NCK - quadratic(x_gr, popt[0], popt[1], 0.))*1.e-3, yerr=H_err_NCK*1.e-3, linestyle='None', color=colors[0])
    ax[2].scatter(x_gr, (H_NCK - quadratic(x_gr, popt[0], popt[1], 0.))*1.e-3, color=colors[0])
    ax[2].errorbar([-1., -0.9], [0., 0.], fmt='o', markersize=9, label='298 K  (Newton et al., 1977)', color=colors[0])
    
    for j, temperature in enumerate([300., 1000., 1500.]):
        

        for i, x_py in enumerate(compositions):

            W_p = 0.e9
            s = solution(pressure, temperature,
                         F_xs = 0., p_xs = W_p*x_py*(1. - x_py),
                         x_a = x_py, a = py, b = gr, cluster_size = cluster_size)

            volumes[i] = s.V
            entropies[i] = s.S
            gibbs[i] = s.gibbs
            energies[i] = s.internal_energy
            enthalpies[i] = s.H
            bulk_sound_velocities[i] = s.bulk_sound_velocity

            #dFdxVs[i] = s.dFdxV # warning, only good if F_xs and P_xs = 0.

            
            g_ideal.set_composition([x_py, 1. - x_py])
            g_ideal.set_state(pressure, temperature)
            bulk_sound_velocities_ideal[i] = g_ideal.bulk_sound_velocity
            
            g_nonideal.set_composition([x_py, 1. - x_py])
            g_nonideal.set_state(pressure, temperature)
            bulk_sound_velocities_nonideal[i] = g_nonideal.bulk_sound_velocity
            
        '''
        dT = 0.5
        py.set_state(pressure, temperature+dT)
        gr.set_state(pressure, temperature+dT)

        py_dKTdTP = py.K_T
        gr_dKTdTP = gr.K_T
        
        py.set_state(pressure, temperature-dT)
        gr.set_state(pressure, temperature-dT)

        py_dKTdTP -= py.K_T
        gr_dKTdTP -= gr.K_T
        py_dKTdTP = py_dKTdTP/(2.*dT)
        gr_dKTdTP = gr_dKTdTP/(2.*dT)
        '''
        py.set_state(pressure, temperature)
        gr.set_state(pressure, temperature)
        '''
        py_dKTdTV = py_dKTdTP + py.alpha*py.K_T*py.params['Kprime_0']
        gr_dKTdTV = gr_dKTdTP + gr.alpha*gr.K_T*gr.params['Kprime_0']
        '''
        V_approx = np.power( (compositions*py.K_T + (1 - compositions)*gr.K_T) /
                             (compositions*py.K_T*np.power(py.V, py.params['Kprime_0']) +
                              (1 - compositions)*gr.K_T*np.power(gr.V, py.params['Kprime_0'])),
                             -1./py.params['Kprime_0'])


        
        #Sxs_approx = (compositions*(py.alpha*py.K_T +
        #                            0.5*py_dKTdTV*(volumes - py.V))*(volumes - py.V) +
        #              (1. - compositions)*(gr.alpha*gr.K_T +
        #                                   0.5*gr_dKTdTV*(volumes - gr.V))*(volumes - gr.V))

        
        Sxs_approx = (compositions*py.alpha*py.K_T*(volumes - py.V) +
                      (1. - compositions)*gr.alpha*gr.K_T*(volumes - gr.V))

        label='{0:.0f} K'.format(temperature)
        color=colors[j]
        ax[0].plot(1. - compositions, (volumes - (compositions*py.V + (1 - compositions)*gr.V))*1.e6, label=label, color=color)
        #ax[0].plot(1. - compositions, (V_approx - (compositions*py.V + (1 - compositions)*gr.V))*1.e6, label='approx')
        ax[1].plot(1. - compositions, entropies - (compositions*py.S + (1 - compositions)*gr.S), label=label, color=color)
        #ax[1].plot(1. - compositions, Sxs_approx, label='approx')
        ax[2].plot(1. - compositions, (enthalpies - (compositions*py.H + (1 - compositions)*gr.H))*1.e-3, label=label, color=color)
        ax[3].plot(1. - compositions, (gibbs - (compositions*py.gibbs + (1 - compositions)*gr.gibbs))*1.e-3, label=label, color=color)

        
        ax2[0].plot(1. - compositions, bulk_sound_velocities/1.e3, label=label, color=color)
        ax2[0].plot(1. - compositions, bulk_sound_velocities_ideal/1.e3, label=label+' (ideal)', color=color, linestyle='--')
        ax2[0].plot(1. - compositions, bulk_sound_velocities_nonideal/1.e3, label=label+' ($W_V = 1.2$ cm$^3$/mol)', color=color, linestyle='-.')

    ax[0].set_ylabel('$V_{excess}$ (cm$^3$/mol)')
    ax[1].set_ylabel('$S_{excess}$ (J/K/mol)')
    ax[2].set_ylabel('$\mathcal{H}_{excess}$ (kJ/mol)')
    ax[3].set_ylabel('$\mathcal{G}_{excess}$ (kJ/mol)')
    for i in range(0, 4):
        ax[i].set_xlim(0., 1.)
        ax[i].set_xlabel('$p_{{gr}}$')
        ax[i].legend(loc='lower center')
        
    fig.tight_layout()
    fig.savefig("py_gr_1bar_excesses.pdf", bbox_inches='tight', dpi=100)


    ax2[0].set_xlim(0., 1.)
    ax2[0].set_xlabel('$p_{{gr}}$')
    ax2[0].legend(loc='lower center')
    ax2[0].set_ylabel('$V_{{\phi}}$ (km/s)')
    
    fig2.tight_layout()
    fig2.savefig("py_gr_1bar_Vphi.pdf", bbox_inches='tight', dpi=100)

    
    plt.show()


    #plt.plot(1. - compositions, dFdxVs)
    #plt.plot(1. - compositions, np.gradient(gibbs, compositions))
    #plt.show()

if run_3 == True:
    
    pressure = 1.e5
    cluster_size = 3.
    
    temperatures = np.linspace(1., 1000., 101)
    pressures = np.array([pressure]*len(temperatures))
    volumes = np.empty_like(temperatures)
    entropies = np.empty_like(temperatures)
    gibbs = np.empty_like(temperatures)
    energies = np.empty_like(temperatures)

    plt.rcParams['figure.figsize'] = 6, 4 # inches
    fig = plt.figure()
    ax = [fig.add_subplot(1, 1, i) for i in range(1, 2)]
    
    pressure = 1.e5
    for x_py in [0.25, 0.50, 0.75]:
        for i, temperature in enumerate(temperatures):
            
            s = solution(pressure, temperature,
                         F_xs = 0., p_xs = 0.,
                         x_a = x_py, a = py, b = gr, cluster_size = cluster_size)

            volumes[i] = s.V
            entropies[i] = s.S
            gibbs[i] = s.gibbs
            energies[i] = s.internal_energy



        label='py$_{{{0:.0f}}}$gr$_{{{1:.0f}}}$'.format(x_py*100., (1. - x_py)*100.)

        ax[0].plot(temperatures,
                   entropies - (x_py*py.evaluate(['S'], pressures, temperatures)[0] +
                                (1. - x_py)*gr.evaluate(['S'], pressures, temperatures)[0]),
                   label=label)

       
    ax[0].scatter([298.15, 298.15, 298.15], [3.25, 2.85, 1.42])

    T, Spy60gr40, Sgr, Spy = np.loadtxt('data/HW1980_S_pygr.dat', unpack=True)
    ax[0].scatter(T, Spy60gr40 - 0.4*Sgr - 0.6*Spy)
    
    ax[0].set_ylabel('$S_{excess}$ (J/K/mol)')
    for i in range(0, 1):
        ax[i].set_xlabel('Temperature (K)')
        ax[i].legend(loc='best')
        
    fig.tight_layout()
    fig.savefig("py_gr_S_excesses.pdf", bbox_inches='tight', dpi=100)
    plt.show()




# Experimental volumes (py-gr from Newton et al. and alm-gr from Cressey et al.)
if run_4 == True:
    x_py_NCK, a_NCK = np.loadtxt('data/py_gr_V_NCK.dat', unpack=True)
    x_gr_CSW, a_CSW = np.loadtxt('data/alm_gr_V_CSW.dat', unpack=True)
    Z = 8.
    volumes_NCK = np.power(a_NCK*1.e-10, 3.)*burnman.constants.Avogadro/Z
    volumes_CSW = np.power(a_CSW*1.e-10, 3.)*burnman.constants.Avogadro/Z
    
    
    temperature = 300.
    pressures_NCK = np.empty_like(volumes_NCK)
    
    py.params['V_0'] = volumes_NCK[0]
    gr.params['V_0'] = volumes_NCK[-1]
    
    for i in range(len(x_py_NCK)):

        x_py = x_py_NCK[i]
        p_py = py.method.pressure(temperature, volumes_NCK[i], py.params)
        p_gr = gr.method.pressure(temperature, volumes_NCK[i], gr.params)

        pressures_NCK[i] = x_py*p_py + (1. - x_py)*p_gr

               
    pressures_CSW = np.empty_like(volumes_CSW)

    
    alm.params['V_0'] = volumes_CSW[0]
    gr.params['V_0'] = volumes_CSW[-1]
    
    for i in range(len(x_gr_CSW)):

        x_alm = 1. - x_gr_CSW[i]
        p_alm = alm.method.pressure(temperature, volumes_CSW[i], alm.params)
        p_gr = gr.method.pressure(temperature, volumes_CSW[i], gr.params)

        pressures_CSW[i] = x_alm*p_alm + (1. - x_alm)*p_gr

                                         
    plt.scatter(1. - x_py_NCK, pressures_NCK/1.e9, label='pyrope-grossular (Newton et al., 1977)')
    plt.scatter(x_gr_CSW, pressures_CSW/1.e9, label='almandine-grossular (Cressey et al., 1978)')
    
    plt.xlim(0., 1.)
    plt.xlabel('$x$ (grossular)')
    plt.ylabel('$P_{excess}$ (GPa)')
    plt.legend(loc='lower right')
    plt.show()
