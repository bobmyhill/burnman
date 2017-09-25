from __future__ import absolute_import
# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.

"""
Pyrope-Grossular "ideal" solution (where ideality is in Helmholtz free energy)
"""

import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from excess_modelling import *

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman.minerals import KMFBZ_2017, SLB_2011, HP_2011_ds62
from bridgmanite_endmembers import *

from scipy.optimize import brentq
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'


# First, let's make the endmembers
KMFBZ_MgSiO3_bdg = KMFBZ_2017.mg_si_perovskite()
KMFBZ_FeAlO3_bdg = KMFBZ_2017.fe_al_perovskite()
KMFBZ_MgSiO3_bdg.name = 'MgSiO$_3$ (K2017)' # Kurnosov et al., 2017
KMFBZ_FeAlO3_bdg.name = 'FeAlO$_3$ (K2017)'


MgSiO3 = mgsio3
MgSiO3.name = 'MgSiO$_3$ (this study)'

FeAlO3_Caracas = fealo3
FeAlO3_Caracas.name = 'FeAlO$_3$ (C2010)'


# Read in data from Murakami et al., 2007
pressures_Murakami, Vp, Vp_err, Vs, Vs_err = np.loadtxt('data/Murakami_2007_MgSiO3_pv_velocities.dat', unpack=True)
pressures_Murakami = pressures_Murakami*1.e9
temperatures_Murakami = np.array([300.]*len(pressures_Murakami))
rho = MgSiO3.evaluate(['density'], pressures_Murakami, temperatures_Murakami)[0]
KS_Murakami = (Vp*Vp - 4./3.*Vs*Vs)*1.e6*rho
KSe_Murakami = np.sqrt(np.power(2.*Vp*Vp_err, 2.) + np.power(4./3.*2.*Vs*Vs_err, 2.)) * (1.e6 * rho)



# Read in data from Kurnosov et al., 2017
# P (GPa) rho C11 C22 C33 C44 C55 C66 C12 C13 C23
data = np.loadtxt('data/Kurnosov_et_al_2017.dat',
                  unpack=True)

pressures_Kurnosov = data[0]*1.e9
temperatures_Kurnosov = np.array([300.]*len(pressures_Kurnosov))
densities_Kurnosov = data[1]*1.e3
C11, C11e, C22, C22e, C33, C33e, C44, C44e, C55, C55e, C66, C66e, C12, C12e, C13, C13e, C23, C23e = data[2:]*1.e9

# Find the best fitting bulk moduli at each pressure
class OrthorhombicModel(object):
    def __init__(self, mineral, delta_params, param_covariances, dof, noise_variance):
        self.m = mineral
        self.data = np.array([[0.]])
        self.delta_params = delta_params
        self.pcov = param_covariances
        self.dof = dof
        self.noise_variance = noise_variance
        
    def set_params(self, param_values):
        index_lists = [[(0, 0)], # C11
                       [(0, 1)], # C12
                       [(0, 2)], # C13
                       [(1, 1)], # C22
                       [(1, 2)], # C23
                       [(2, 2)], # C33
                       [(3, 3)], # C44
                       [(4, 4)], # C55
                       [(5, 5)]] # C66
        self.m.params['rho_0'] = param_values[0]
        self.m.params['stiffness_tensor_0'] = burnman.anisotropy.voigt_array_from_cijs(param_values[1:],
                                                                                       index_lists)
        burnman.Material.__init__(self.m) # need to reset cached values
        
    def get_params(self):
        params = [self.m.params['rho_0']]
        index_lists = [[(0, 0)], # C11
                       [(0, 1)], # C12
                       [(0, 2)], # C13
                       [(1, 1)], # C22
                       [(1, 2)], # C23
                       [(2, 2)], # C33
                       [(3, 3)], # C44
                       [(4, 4)], # C55
                       [(5, 5)]] # C66
        params.extend([self.m.params['stiffness_tensor_0'][idx[0]] for idx in index_lists])
        return params
    
    def function(self, x, flag):
        return None

KS_Kurnosov = np.empty_like(pressures_Kurnosov)
KSe_Kurnosov = np.empty_like(pressures_Kurnosov)
for i in range(len(pressures_Kurnosov)):
    density = densities_Kurnosov[i]
    
    cijs = np.array([C11[i], C12[i], C13[i],
                     C22[i], C23[i], C33[i],
                     C44[i], C55[i], C66[i]])
    params = np.array([density, C11[i], C12[i], C13[i],
                       C22[i], C23[i], C33[i],
                       C44[i], C55[i], C66[i]])
    param_covariances = np.diag(np.power(np.array([0., C11e[i], C12e[i], C13e[i],
                                                   C22e[i], C23e[i], C33e[i],
                                                   C44e[i], C55e[i], C66e[i]]), 2.))

    static_bdg = burnman.anisotropy.OrthorhombicMaterial(rho = density, cijs = cijs)
        
    static_bdg.params['stiffness_tensor_0']
    static_bdg.bulk_modulus_reuss
    
    model = OrthorhombicModel(mineral=static_bdg,
                              delta_params = params*1.e-5,
                              param_covariances = param_covariances,
                              dof = 100.,
                              noise_variance = 1.)
    var = burnman.nonlinear_fitting.confidence_prediction_bands(model = model,
                                                                x_array = np.array([[0.]]),
                                                                confidence_interval = 0.95,
                                                                f = lambda x: static_bdg.bulk_modulus_reuss)[0][0]
    KS_Kurnosov[i] = static_bdg.bulk_modulus_reuss
    KSe_Kurnosov[i] = np.sqrt(var)


KT_est_Kurnosov = KS_Kurnosov - KMFBZ_MgSiO3_bdg.evaluate(['K_S'], pressures_Kurnosov, temperatures_Kurnosov)[0] + KMFBZ_MgSiO3_bdg.evaluate(['K_T'], pressures_Kurnosov, temperatures_Kurnosov)[0]
KT_est_Murakami = KS_Murakami - KMFBZ_MgSiO3_bdg.evaluate(['K_S'], pressures_Murakami, temperatures_Murakami)[0] + KMFBZ_MgSiO3_bdg.evaluate(['K_T'], pressures_Murakami, temperatures_Murakami)[0]

formula_Kurnosov = {'Mg': 0.9, 'Si':0.9, 'Fe':0.1, 'Al':0.1, 'O':3.}
volumes_Kurnosov = burnman.processchemistry.formula_mass(formula_Kurnosov)/densities_Kurnosov


# Read in data from Fiquet et al. (2000)
pressures_Fiquet, Perr_Fiquet, volumes_Fiquet, Verr_Fiquet = np.loadtxt('data/Fiquet_2000_MgSiO3_pv_PV.dat', unpack=True)
pressures_Fiquet = pressures_Fiquet*1.e9 
Perr_Fiquet = Perr_Fiquet*1.e9
Z = 4.
volumes_Fiquet = volumes_Fiquet/1.e30*burnman.constants.Avogadro/Z
Verr_Fiquet = Verr_Fiquet/1.e30*burnman.constants.Avogadro/Z

# Read in data from Daniel et al. (2001)
pressures_Daniel, volumes_Daniel = np.loadtxt('data/Daniel_et_al_2001_Al_pv_77.dat', unpack=True)
pressures_Daniel = pressures_Daniel*1.e9
volumes_Daniel = volumes_Daniel/1.e30*burnman.constants.Avogadro/Z


# Now let's plot up the data
fig_VK = plt.figure(figsize=(12,5))
ax_VK = [fig_VK.add_subplot(1, 2, i) for i in range(1, 3)]
fig_Kp = plt.figure(figsize=(7,5))
ax_Kp = [fig_Kp.add_subplot(1, 1, i) for i in range(1, 2)]


minerals = ['MgSiO3', 'FeAlO3', 'FeAlss', 'AlAlss', 'PREM', 'other']
colors = {m: next(ax_VK[0]._get_lines.prop_cycler)['color'] for m in minerals}



prem = burnman.seismic.PREM()
Pprem, Kprem = prem.evaluate(['pressure', 'K'], np.linspace(6371000. - 5600000.,
                                                            6371000. - 3630000., 21))
ax_Kp[0].plot(Pprem/Kprem, 1./np.gradient(Kprem, Pprem, edge_order=2), label='PREM', color=colors['PREM'])



label='MgSiO$_3$ (F2000)' # Fiquet et al., 2000
ax_VK[0].errorbar(pressures_Fiquet/1.e9, volumes_Fiquet*1.e6, xerr = Perr_Fiquet/1.e9, yerr=Verr_Fiquet*1.e6,
                  label=label, color=colors['MgSiO3'], linestyle='None', capsize=3, elinewidth=2, marker='o')

label='MgSiO$_3$ (M2007)' # Murakami et al., 2007
ax_VK[1].errorbar(pressures_Murakami/1.e9, KT_est_Murakami/1.e9, yerr=KSe_Murakami/1.e9,
                  label=label, color=colors['MgSiO3'], linestyle='None', capsize=3, elinewidth=2, marker='o')

label='(MgSi)$_{0.923}$(AlAl)$_{0.077}$O$_3$ (D2001)' # Daniel et al., 2001
ax_VK[0].scatter(pressures_Daniel/1.e9, volumes_Daniel*1.e6, marker=(4, 1, 0), label=label, color=colors['AlAlss'])


label='(MgSi)$_{0.9}$(FeAl)$_{0.1}$O$_3$ (K2017)' # Kurnosov et al., 2017
ax_VK[0].scatter(pressures_Kurnosov/1.e9, volumes_Kurnosov*1.e6, label=label, color=colors['FeAlss'])
ax_VK[1].errorbar(pressures_Kurnosov/1.e9, KT_est_Kurnosov/1.e9, yerr=KSe_Kurnosov/1.e9,
                  label=label, color=colors['FeAlss'], linestyle='None', capsize=3, elinewidth=2, marker='o')
ax_Kp[0].scatter(pressures_Kurnosov/KT_est_Kurnosov, 1./np.gradient(KT_est_Kurnosov, pressures_Kurnosov, edge_order=2),
                 label=label, color=colors['FeAlss'])



pressures_plot = np.linspace(1.e5, 150.e9, 101)
temperatures_plot = np.array([300.]*len(pressures_plot))
Vs, KTs = MgSiO3.evaluate(['V', 'K_T'], pressures_plot, temperatures_plot)

ax_VK[0].plot(pressures_plot/1.e9, Vs*1.e6, label=MgSiO3.name, color=colors['MgSiO3'])
ax_VK[1].plot(pressures_plot/1.e9, KTs/1.e9, label=MgSiO3.name, color=colors['MgSiO3'])
ax_Kp[0].plot(pressures_plot/KTs, 1./np.gradient(KTs, pressures_plot, edge_order=2), linestyle='--', label=MgSiO3.name, color=colors['MgSiO3'])


pressures_KMFBZ = np.linspace(1.e5, 69.2e9, 101)
temperatures_KMFBZ = np.array([300.]*len(pressures_KMFBZ))
ax_VK[0].plot(pressures_KMFBZ/1.e9,
            (0.1*KMFBZ_FeAlO3_bdg.evaluate(['V'], pressures_KMFBZ, temperatures_KMFBZ)[0] +
             0.9*KMFBZ_MgSiO3_bdg.evaluate(['V'], pressures_KMFBZ, temperatures_KMFBZ)[0])*1.e6, color=colors['FeAlss'], linestyle='-.', label=label)


ax_VK[0].plot(pressures_KMFBZ/1.e9, KMFBZ_FeAlO3_bdg.evaluate(['V'], pressures_KMFBZ, temperatures_KMFBZ)[0]*1.e6, color=colors['FeAlO3'], linestyle='-.', label=KMFBZ_FeAlO3_bdg.name)
ax_VK[1].plot(pressures_KMFBZ/1.e9, KMFBZ_FeAlO3_bdg.evaluate(['K_T'], pressures_KMFBZ, temperatures_KMFBZ)[0]/1.e9, color=colors['FeAlO3'], linestyle='-.', label=KMFBZ_FeAlO3_bdg.name)


ax_VK[0].plot(pressures_plot/1.e9, FeAlO3_Caracas.evaluate(['V'], pressures_plot, temperatures_plot)[0]*1.e6, color=colors['FeAlO3'], label=FeAlO3_Caracas.name)
ax_VK[1].plot(pressures_plot/1.e9, FeAlO3_Caracas.evaluate(['K_T'], pressures_plot, temperatures_plot)[0]/1.e9, color=colors['FeAlO3'], label=FeAlO3_Caracas.name)

ax_Kp[0].plot((pressures_plot/FeAlO3_Caracas.evaluate(['K_T'], pressures_plot, temperatures_plot)[0]),
               1./np.gradient(FeAlO3_Caracas.evaluate(['K_T'], pressures_plot, temperatures_plot)[0],
                              pressures_plot, edge_order=2), linestyle='--',
               color=colors['FeAlO3'], label=FeAlO3_Caracas.name)


volumes_ss = np.empty_like(pressures_plot)
KT_ss = np.empty_like(pressures_plot)
for i, pressure in enumerate(pressures_plot):
    s = solution(pressure, 300.,
                 F_xs = 0., p_xs = 0.,
                 x_a = 0.9, a = MgSiO3, b = FeAlO3_Caracas, cluster_size = 2.)
    volumes_ss[i] = s.V
    KT_ss[i] = s.K_T

Kprime_ss = np.gradient(KT_ss, pressures_plot, edge_order=2)

print('V_0: {0}'.format(volumes_ss[0])) 
print('K_0: {0}'.format(KT_ss[0])) 
print('K\'_0: {0}'.format(Kprime_ss[0])) 


label='(MgSi)$_{0.9}$(FeAl)$_{0.1}$O$_3$ (elastic)'
ax_VK[0].plot(pressures_plot/1.e9, volumes_ss*1.e6, color=colors['FeAlss'], label=label)
ax_VK[1].plot(pressures_plot/1.e9, KT_ss/1.e9, color=colors['FeAlss'], label=label)

ax_Kp[0].plot((pressures_plot/KT_ss), 1./np.gradient(KT_ss, pressures_plot, edge_order=2), color=colors['FeAlss'], label=label)



ax_VK[1].plot(pressures_plot/1.e9, KT_ss/1.e9, color=colors['FeAlss'], label=label)


# Plot some other things on the K' figure
per = burnman.minerals.SLB_2011.periclase()
KTs_per = per.evaluate(['K_T'], pressures_plot, temperatures_plot)[0]
#ax_Kp[0].plot(pressures_plot/KTs_per, 1./np.gradient(KTs_per, pressures_plot, edge_order=2), label='MgO (SLB2011)', color='black')

color='black'
Kps = [2., 3., 4., 5.]
ax_Kp[0].plot([0., 3./5.], [0., 3./5.], linewidth=1, color=color)
for Kp in Kps:
    ax_Kp[0].text(0.005, 1./Kp - 0.03, '$K\'$ = {0:.0f}'.format(Kp))
    ax_Kp[0].plot([0., 1./Kp], [1./Kp, 1./Kp], linestyle='--', linewidth=1, color=color)


ax_Kp[0].text(0.005, 3./5. - 0.03, '$K\'$ = 5/3 (Thomas-Fermi limit)'.format(Kp))






ax_VK[0].set_xlabel('Pressure (GPa)')
ax_VK[0].set_ylabel('Volume (cm$^3$/mol)')

ax_VK[1].set_xlim(0., 100.)
ax_VK[1].set_ylim(180., 620.)

ax_VK[1].set_xlabel('Pressure (GPa)')
ax_VK[1].set_ylabel('$K_T$ (GPa)')


ax_Kp[0].set_xlim(0.,0.3)
ax_Kp[0].set_ylim(0.,0.6)

ax_Kp[0].set_xlabel('$P/K_T$')
ax_Kp[0].set_ylabel('1/$K\'$')


ax_VK[0].legend(loc='best')
ax_VK[1].legend(loc='best')
ax_Kp[0].legend(loc='lower right')


fig_VK.tight_layout()
fig_VK.savefig("bridgmanite_eos.pdf", bbox_inches='tight', dpi=100)

fig_Kp.savefig("bridgmanite_Kprime.pdf", bbox_inches='tight', dpi=100)
plt.show()
