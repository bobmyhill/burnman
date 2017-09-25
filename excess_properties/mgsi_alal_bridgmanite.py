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

Z = 4.

MgSi = burnman.minerals.SLB_2011.mg_perovskite()
for T in [1., 300.]:
    MgSi.set_state(1.e5, T)
    print (MgSi.V*1.e6, MgSi.K_T/1.e9)

# Brodholt
formula = burnman.processchemistry.dictionarize_formula('MgSiO3')
MgSi_params = {'name': 'MgSiO$_3$',
               'P_0': 1.e5,
               'T_0': 300.,
               'V_0': 162.42/1.e30*burnman.constants.Avogadro/Z,
               'K_0': 239.e9,
               'Kprime_0': 4.,
               'equation_of_state': 'bm3',
               'formula': formula,
               'molar_mass': burnman.processchemistry.formula_mass(formula)}

formula = burnman.processchemistry.dictionarize_formula('MgAlO2.5')
MgAlv_params = {'name': 'MgAlO$_{{2.5}}$',
                'P_0': 1.e5,
                'T_0': 300.,
                'V_0': 345.16/2./1.e30*burnman.constants.Avogadro/Z,
                'K_0': 155.e9,
                'Kprime_0': 3.9,
                'Kprime_inf': 3.,
                'equation_of_state': 'rkprime',
                'formula': formula,
                'molar_mass': burnman.processchemistry.formula_mass(formula)}
MgAlv = burnman.Mineral(params=MgAlv_params)

# Brodholt
formula = burnman.processchemistry.dictionarize_formula('Mg0.75Si0.75Al1.5O3')
AlAl_params = {'name': 'MgSiO3 - AlAlO$_3$',
               'P_0': 1.e5,
               'T_0': 300.,
               'V_0': 163.31/1.e30*burnman.constants.Avogadro/Z,
               'K_0': 224.e9,
               'Kprime_0': 4.,
               'equation_of_state': 'bm3',
               'formula': formula,
               'molar_mass': burnman.processchemistry.formula_mass(formula)}


formula = burnman.processchemistry.dictionarize_formula('Al2O3')
AlAl_params = {'name': 'AlAlO$_3$',
               'P_0': 1.e5,
               'T_0': 300.,
               'V_0': 25.908e-6,
               'K_0': 223.e9,
               'Kprime_0': 3.9,
               'equation_of_state': 'bm3',
               'formula': formula,
               'molar_mass': burnman.processchemistry.formula_mass(formula)}


AlAl = burnman.Mineral(params=AlAl_params)

#MgSi = burnman.minerals.SLB_2011.mg_perovskite()
#AlAl = burnman.CombinedMineral([burnman.minerals.SLB_2011.mg_perovskite(),
#                                burnman.minerals.SLB_2011.al_perovskite()],
#                               [0.75, 0.25])
#
#AlAl = burnman.minerals.SLB_2011.al_perovskite()

'''
pressures = np.linspace(1.e5, 100.e9, 101)
temperatures = np.array([300.]*len(pressures))
plt.plot(pressures/1.e9, MgSi.evaluate(['V'], pressures, temperatures)[0])
plt.plot(pressures/1.e9, MgAlv.evaluate(['V'], pressures, temperatures)[0])
plt.show()
'''


formula = burnman.processchemistry.dictionarize_formula('Mg0.75Si0.75Al0.5O3')
pypv_params = {'equation_of_state': 'rkprime',
               'V_0': 24.81e-6, # +/- 0.01 e-6
               'K_0': 256.2e9, # +/- 2e9
               'Kprime_0': 3.90,
               'Kprime_inf': 3.0,
               'molar_mass': burnman.processchemistry.formula_mass(formula),
               'n': 3.,
               'formula': formula}
pypv = burnman.Mineral(params=pypv_params)

formula = burnman.processchemistry.dictionarize_formula('MgSiO3')
params = {'equation_of_state': 'rkprime',
          'V_0': 24.445e-6,
          'K_0': 253.e9,
          'Kprime_0': 3.90,
          'Kprime_inf': 3.,
          'molar_mass': burnman.processchemistry.formula_mass(formula),
          'n': 3.,
          'formula': formula}
MgSiO3 = burnman.Mineral(params=params)


pressures = np.linspace(1.e5, 10.e9, 101)
# Solution model
volumes_ss = np.empty_like(pressures)
KT_ss = np.empty_like(pressures)
for i, pressure in enumerate(pressures):
    s = solution(pressure, 300.,
                 F_xs = 0., p_xs = 0.,
                 x_a = 0.75, a = MgSiO3, b = AlAl, cluster_size = 1.)
    volumes_ss[i] = s.V
    KT_ss[i] = s.K_T

Kprime_ss = np.gradient(KT_ss, pressures, edge_order=2)
print('V_0: {0}'.format(volumes_ss[0])) 
print('K_0: {0}'.format(KT_ss[0])) 
print('K\'_0: {0}'.format(Kprime_ss[0]))   
plt.plot(pressures/1.e9, volumes_ss*1.e6)

plt.show()
exit()













X, P, Perr, V, Verr = np.loadtxt('data/Walter_et_al_2004_EoS_al_pv.dat', unpack=True)


# Fit pypv data from Walter et al. (2004)
Ppy = np.array([p for i, p in enumerate(P) if X[i]==0.25])*1.e9
Ppyerr = np.array([p for i, p in enumerate(Perr) if X[i]==0.25])*1.e9
Tpy = np.array([300. for i, p in enumerate(P) if X[i]==0.25])
Tpyerr = np.array([1. for i, p in enumerate(P) if X[i]==0.25])
Vpy = np.array([v for i, v in enumerate(V) if X[i]==0.25])/1.e6
Vpyerr = np.array([v for i, v in enumerate(Verr) if X[i]==0.25])/1.e6

PTV = np.array([Ppy, Tpy, Vpy]).T
nul = 0.*PTV.T[0]
PTV_covariances = np.array([[Ppyerr*Ppyerr, nul, nul],
                            [nul, Tpyerr*Tpyerr, nul],
                            [nul, nul, Vpyerr*Vpyerr]]).T
fitted_eos = burnman.eos_fitting.fit_PTV_data(pypv, ['V_0', 'K_0'], PTV, PTV_covariances, verbose=False)
burnman.tools.pretty_print_values(fitted_eos.popt, fitted_eos.pcov, fitted_eos.fit_params)

for x in [0., 0.05, 0.1, 0.2, 0.25]:
    if x==0.25:
        print([v for i, v in enumerate(V) if X[i]==x])
    plt.errorbar([p for i, p in enumerate(P) if X[i]==x],
                 [v for i, v in enumerate(V) if X[i]==x],
                 xerr=[p for i, p in enumerate(Perr) if X[i]==x],
                 yerr=[v for i, v in enumerate(Verr) if X[i]==x],
                 linestyle='None', label=x)

pressures_Fiquet, Perr_Fiquet, volumes_Fiquet, Verr_Fiquet = np.loadtxt('data/Fiquet_2000_MgSiO3_pv_PV.dat', unpack=True)
volumes_Fiquet = volumes_Fiquet/1.e30*burnman.constants.Avogadro/Z*1.e6
Verr_Fiquet = Verr_Fiquet/1.e30*burnman.constants.Avogadro/Z*1.e6
plt.errorbar(pressures_Fiquet, volumes_Fiquet, xerr = Perr_Fiquet, yerr=Verr_Fiquet, label='Fiquet et al. (2000)', linestyle='None', capsize=3, elinewidth=2)

pressures = np.linspace(1.e5, 100.e9, 101)
temperatures = np.array([300.]*len(pressures))
plt.plot(pressures/1.e9, MgSiO3.evaluate(['V'], pressures, temperatures)[0]*1.e6)
plt.plot(pressures/1.e9, pypv.evaluate(['V'], pressures, temperatures)[0]*1.e6)

plt.legend(loc='best')
plt.show()


plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'

run_2 = True # excess volumes as a function of composition



# 2) excess volumes as a function of composition
if run_2 == True:
    
    pressure = 1.e5
    temperature = 300.
    cluster_size = 1.
    
    compositions = np.linspace(0., 0.99999, 101)
    volumes = np.empty_like(compositions)
    moduli = np.empty_like(compositions)
    enthalpies = np.empty_like(compositions)

    
    plt.rcParams['figure.figsize'] = 12, 6 # inches
    fig = plt.figure()
    ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
    #colors = ['black', 'blue', 'red', 'blue', 'white']

    colors = [next(ax[0]._get_lines.prop_cycler)['color'] for i in range(10)]
    colors[4] = 'white'
    # EXPERIMENTAL DATA
    ax[0].errorbar([0.75],
                   np.array([164.0])/1.e30*burnman.constants.Avogadro/Z*1.e6,
                   yerr=np.array([0.2])/1.e30*burnman.constants.Avogadro/Z*1.e6,
                   marker='v', linestyle='None', color=colors[0])

    
    ax[1].errorbar([0.75], 0., yerr=0.,
                   label='Ito et al. (1998)', marker='v', linestyle='None', color=colors[0])
    
    
    ax[1].errorbar([0.9], [225.], yerr=[1.2],
                  label='Kubo et al. (2000)', marker='^', linestyle='None', color=colors[0])

    
    ax[0].scatter([1.0],
                  [24.445],
                  marker='d', color=colors[0])
    
    ax[1].scatter([1.0],
                  [253.],
                  label='Fiquet et al. (2000)', marker='d', color=colors[0])


    ax[0].scatter(np.array([1.0, 0.95, 0.9, 0.8, 0.75]),
                  np.array([162.47, 162.95, 163.42, 164.37, 164.85])
                  /1.e30*burnman.constants.Avogadro/Z*1.e6,
                  marker='o', color=colors[0])

    ax[1].errorbar([1.0, 0.95, 0.9, 0.8, 0.75],
                   [267.3, 268.9, 262.6, 254.2, 253.4],
                   yerr=[6.9, 7.0, 6.7, 6.3, 6.2],
                   label='Walter et al. (2004)', marker='o',
                   linestyle='None', color=colors[0])   


    '''
    # 37, 52; mixes I, F
    ax[0].errorbar([0.95],
                   np.array([163.18])/1.e30*burnman.constants.Avogadro/Z*1.e6,
                   yerr=np.array([0.09])/1.e30*burnman.constants.Avogadro/Z*1.e6,
                   label='Walter et al. (2006)', marker='o', linestyle='None', color=colors[0])
    ax[1].errorbar([0.95],
                   [261.], yerr=[8], linestyle='None',
                   label='Walter et al. (2006)', marker='o', color=colors[0])
    '''

    # DATASETS
    ax[0].scatter([0.0, 1.0],
                  [burnman.minerals.SLB_2005.al_perovskite().params['V_0']*1.e6,
                   burnman.minerals.SLB_2005.mg_perovskite().params['V_0']*1.e6],
                  marker='s', color=colors[2])
        
    ax[1].scatter([0.0, 1.0],
                  [burnman.minerals.SLB_2005.al_perovskite().params['K_0']/1.e9,
                   burnman.minerals.SLB_2005.mg_perovskite().params['K_0']/1.e9],
                  label='SLB (2005)', marker='s', color=colors[2])

    ax[0].scatter([0.0, 1.0],
                  [burnman.minerals.SLB_2011.al_perovskite().params['V_0']*1.e6,
                   burnman.minerals.SLB_2011.mg_perovskite().params['V_0']*1.e6],
                  marker='h', color=colors[2])
        
    ax[1].scatter([0.0, 1.0],
                  [burnman.minerals.SLB_2011.al_perovskite().params['K_0']/1.e9,
                   burnman.minerals.SLB_2011.mg_perovskite().params['K_0']/1.e9],
                  label='SLB (2011)', marker='h', color=colors[2])

    ax[0].scatter([0.0, 1.0],
                  [burnman.minerals.HHPH_2013.apv().params['V_0']*1.e6,
                   burnman.minerals.HHPH_2013.mpv().params['V_0']*1.e6],
                  marker='d', color=colors[2])
    
    ax[1].scatter([0.0, 1.0],
                  [burnman.minerals.HHPH_2013.apv().params['K_0']/1.e9,
                   burnman.minerals.HHPH_2013.mpv().params['K_0']/1.e9],
                  label='Holland et al. (2013)', marker='d', color=colors[2])

    

    # AB INITIO
    ax[0].scatter([0.0],
                  [4.1610217e-29*4.*burnman.constants.Avogadro/Z*1.e6],
                  label='Thomson et al. (1996)', marker='s', color=colors[4], edgecolors=colors[0])
    
    ax[1].scatter([0.0],
                  [235.],
                  marker='s', color=colors[4], edgecolors=colors[0])

    ax[0].scatter([1.0, 0.75],
                  np.array([162.42, 163.31])
                  /1.e30*burnman.constants.Avogadro/Z*1.e6,
                  label='Brodholt (2000)', marker='^', color=colors[4], edgecolors=colors[0])
    ax[1].scatter([1.0, 0.75], [239., 224.],
                  marker='^', color=colors[4], edgecolors=colors[0])

    ax[0].scatter([0.00],
                  np.array([345.16/2.])
                  /1.e30*burnman.constants.Avogadro/Z*1.e6,
                  label='Brodholt (2000; MgAlO$_{{2.5}}$v$_{{0.5}}$)', marker='v', color=colors[4], edgecolors=colors[3])
    ax[1].scatter([0.00], [155.],
                  marker='v', color=colors[4], edgecolors=colors[3])
    
    ax[0].scatter([1.0, 0.0],
                  np.array([40.78, 43.22])*4./1.e30*burnman.constants.Avogadro/Z*1.e6,
                  label='Caracas and Cohen (2005)', marker='d', color=colors[4], edgecolors=colors[0])
    ax[1].scatter([1.0, 0.0], [232., 202.],
                  marker='d', color=colors[4], edgecolors=colors[0])
    
    ax[0].scatter([1.0, 0.9375, 0.75, 0.5, 0.25, 0.0],
                  np.array([162.095, 162.619, 162.88, 164.418, 165.256, 165.570])
                  /1.e30*burnman.constants.Avogadro/Z*1.e6,
                  label='Panero et al. (2006)', marker='*', color=colors[4], edgecolors=colors[0])
    ax[1].scatter([1.0, 0.9375, 0.75, 0.5, 0.25, 0.0],
                  [262.1, 258.8, 251.7, 237.9, 229.9, 232.9],
                  marker='*', color=colors[4], edgecolors=colors[0])




    


    '''
    # Mg0.9Al0.2Si0.9O3
    # Al/(Al+Mg+Si) = 0.1
    ax[0].scatter([0.9], [163.5/1.e30*burnman.constants.Avogadro/Z*1.e6], label='Kubo et al. (2000)')
    ax[1].scatter([0.9], [225.5], label='Kubo et al. (2000)')

    # Mg0.923Al0.154Si0.923O3
    # Al/(Al+Mg+Si) = 0.077
    ax[0].scatter([0.923], [163.52/1.e30*burnman.constants.Avogadro/Z*1.e6], label='Daniel et al. (2001)')
    ax[1].scatter([0.923], [229.], label='Daniel et al. (2001)')

    # Mg0.95Al0.1Si0.95O3
    # Al/(Al+Mg+Si) = 0.05
    ax[0].scatter([0.95], [163.051/1.e30*burnman.constants.Avogadro/Z*1.e6], label='Zhang and Weidner (1999)')
    ax[0].scatter([0.95], [163.275/1.e30*burnman.constants.Avogadro/Z*1.e6], label='Zhang and Weidner (1999)')
    ax[1].scatter([0.95], [234.], label='Zhang and Weidner (1999)')
    '''
    # 
    for j, (name, m2) in enumerate([('AlAlO$_3$ - MgSiO$_3$', AlAl),
                                    ('MgAlO$_{{2.5}}$v$_{{0.5}}$ - MgSiO$_3$', MgAlv)]):
        

        for i, x_MgSi in enumerate(compositions):

            W_p = 0.e9
            s = solution(
                pressure, temperature,
                         F_xs = 0., p_xs = W_p*x_MgSi*(1. - x_MgSi),
                         x_a = x_MgSi, a = MgSiO3, b = m2, cluster_size = cluster_size)

            volumes[i] = s.V
            moduli[i] = s.K_T
            enthalpies[i] = s.H
            
        MgSiO3.set_state(pressure, temperature)
        m2.set_state(pressure, temperature)

        print('W$_{{H}}$ ({0}) = {1} kJ/mol'.format(name,
                                            4./1000*max(enthalpies -
                                                        (compositions*enthalpies[-1] +
                                                         (1. - compositions)*enthalpies[0]))))
        
        label='{0}'.format(name)
        color=colors[j]
        ax[0].plot(compositions, volumes*1.e6, label=label, color=color)
        ax[1].plot(compositions, moduli/1.e9, color=color)
        #ax[1].plot(compositions, enthalpies - (compositions*enthalpies[-1] + (1. - compositions)*enthalpies[0]), label=label, color=color)

    #ax[0].set_ylim(24,30)
    ax[1].set_ylim(150,)
    
    ax[0].set_ylabel('$V$ (cm$^3$/mol)')
    ax[1].set_ylabel('$K_T$ (GPa)')
    for i in range(0, 2):
        #ax[i].set_xlim(0., 1.)
        ax[i].set_xlabel('$p_{{MgSiO3}}$')
        ax[i].legend(loc='best')
        
    fig.tight_layout()
    fig.savefig("mgsi_alal_bridgmanite_1bar_excesses.pdf", bbox_inches='tight', dpi=100)

    plt.show()
