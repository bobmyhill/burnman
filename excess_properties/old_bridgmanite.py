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

from scipy.optimize import brentq
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'

this = True
if this == True:
    def plot_anisotropic_seismic_properties(mineral):
        """
        Makes colour plots of:
        Compressional wave velocity: Vp
        Anisotropy: (Vs1 - Vs2)/(Vs1 + Vs2)
        Vp/Vs1
        linear compressibility: beta
        Youngs Modulus: E
        """
        try:
            plt.style.use('seaborn-talk')
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['axes.edgecolor'] = 'black'
            plt.rcParams['figure.figsize'] = 16, 10 # inches
        except:
            pass
        
        zeniths = np.linspace(np.pi/2., np.pi, 31)
        azimuths = np.linspace(0., 2.*np.pi, 91)
        Rs = np.sin(zeniths)/(1. - np.cos(zeniths))
        r, theta = np.meshgrid(Rs, azimuths)
        
        vps = np.empty_like(r)
        vs1s = np.empty_like(r)
        vs2s = np.empty_like(r)
        betas = np.empty_like(r)
        Es = np.empty_like(r)
        for i, az in enumerate(azimuths):
            for j, phi in enumerate(zeniths):
                d = np.array([np.cos(az)*np.sin(phi), np.sin(az)*np.sin(phi), -np.cos(phi)]) # change_hemispheres
                velocities = mineral.wave_velocities(d)
                betas[i][j] = mineral.linear_compressibility(d)
                Es[i][j] = mineral.youngs_modulus(d)
                vps[i][j] = velocities[0][0]
                vs1s[i][j] = velocities[0][1]
                vs2s[i][j] = velocities[0][2]
                
        fig = plt.figure()
        names = ['Vp (km/s)', 'Vs1 (km/s)', 'Vp/Vs1', 'S-wave anisotropy (%)', 'Linear compressibility (GPa$^{-1}$)', 'Youngs Modulus (GPa)']
        items = [vps/1000., vs1s/1000., vps/vs1s, 200.*(vs1s - vs2s)/(vs1s + vs2s), betas*1.e9, Es/1.e9]
        ax = []
        im = []
        ndivs = 100
        for i, item in enumerate(items):
            ax.append(fig.add_subplot(2, 3, i+1, projection='polar'))
            ax[i].set_yticks([100])
            ax[i].set_title(names[i])

            vmin = np.min(item)
            vmax = np.max(item)
            spacing = np.power(10., np.floor(np.log10(vmax - vmin)))
            nt = int((vmax - vmin - vmax%spacing + vmin%spacing)/spacing)
            if nt == 1:
                spacing = spacing/4.
            elif nt < 4:
                spacing = spacing/2.
            elif nt > 8:
                spacing = spacing*2.
                
            tmin = vmin + (spacing - vmin%spacing)
            tmax = vmax - vmax%spacing
            nt = int((tmax - tmin)/spacing + 1)
            
            ticks = np.linspace(tmin, tmax, nt)
            im.append(ax[i].contourf(theta, r, item, ndivs, cmap=plt.cm.jet_r, vmin=vmin, vmax=vmax))
            lines = ax[i].contour(theta, r, item, ticks, colors=('black',), linewidths=(1,))
            
            cbar = fig.colorbar(im[i], ax=ax[i], ticks=ticks)
            cbar.add_lines(lines)

        plt.tight_layout()
        
# Murakami

MgSiO3_SLB = SLB_2011.mg_perovskite()
params = {'name': 'MgSiO$_3$ (this study)',
          'equation_of_state': 'rkprime',
          'V_0': MgSiO3_SLB.params['V_0'],
          'K_0': 253.e9,
          'Kprime_0': 3.90,
          'Kprime_inf': 2.9,
          'molar_mass': MgSiO3_SLB.params['molar_mass'],
          'n': MgSiO3_SLB.params['n'],
          'formula': MgSiO3_SLB.params['formula']}

MgSiO3 = burnman.Mineral(params=params)

formula = {'Fe': 1., 'Al': 1., 'O': 3.}
params = {'name': 'FeAlO$_3$ (Caracas, 2010)',
          'equation_of_state': 'bm3',
          'V_0': 27.61e-6,
          'K_0': 211.e9,
          'Kprime_0': 3.73,
          'molar_mass': burnman.processchemistry.formula_mass(formula),
          'n': 5.,
          'formula': formula}

FeAlO3_Caracas = burnman.Mineral(params=params)



#fitted_eos = burnman.eos_fitting.fit_PTV_data(MgSiO3, params, PTV, PTV_covariances, verbose=False)

fig = plt.figure()
ax = [fig.add_subplot(2, 1, i) for i in range(1, 3)]

# Murakami
pressures_Murakami, Vp, Vp_err, Vs, Vs_err = np.loadtxt('data/Murakami_2007_MgSiO3_pv_velocities.dat', unpack=True)
pressures_Murakami = pressures_Murakami*1.e9
temperatures_Murakami = np.array([300.]*len(pressures_Murakami))
rho = MgSiO3.evaluate(['density'], pressures_Murakami, temperatures_Murakami)[0]
ax[0].scatter(pressures_Murakami/1.e9, (Vp*Vp - 4./3.*Vs*Vs)*1.e6*rho)
ax[1].scatter(pressures_Murakami/1.e9, Vs*Vs*1.e6*rho)


KS_Murakami = (Vp*Vp - 4./3.*Vs*Vs)*1.e6*rho
KSe_Murakami = np.sqrt(np.power(2.*Vp*Vp_err, 2.) + np.power(4./3.*2.*Vs*Vs_err, 2.)) * (1.e6 * rho)

pressures = np.linspace(1.e5, 135.e9, 101)
temperatures = np.array([300.]*len(pressures))
K_S, K_T, G, rho = MgSiO3.evaluate(['K_S', 'K_T', 'G', 'density'], pressures, temperatures)
ax[0].plot(pressures/1.e9, K_S)
ax[1].plot(pressures/1.e9, G)

plt.show()

plt.plot(pressures/K_T, 1./np.gradient(K_T, pressures, edge_order=2))
plt.show()

'''
# Vanpeteghem et al. data for MgSiO3
P, Perr, a, aerr, b, berr, c, cerr, V, Verr = np.loadtxt('data/Vanpet_et_al_MgSiO3_eos.dat', unpack=True)
P = P*1.e9
T = np.array([300.]*len(P))
Perr = Perr*1.e9
Terr = np.array([1.]*len(Perr))
Z = 4.
V = V/1.e30*burnman.constants.Avogadro/Z
Verr = Verr/1.e30*burnman.constants.Avogadro/Z

PTV = np.array([P, T, V]).T

nul = 0.*PTV.T[0]
PTV_covariances = np.array([[Perr*Perr, nul, nul],
                            [nul, Terr*Terr, nul],
                            [nul, nul, Verr*Verr]]).T
    
MgSiO3_params = {'name':'MgSiO3',
                 'V_0': 2.4419e-05,
                 'K_0': 2.312e+11,
                 'Kprime_0': 4.0,
                 'equation_of_state': 'vinet',
                 'P_0': 1.e5,
                 'T_0': 300.}
MgSiO3 = burnman.Mineral(params=MgSiO3_params)
params=['V_0', 'K_0', 'Kprime_0']


fitted_eos = burnman.eos_fitting.fit_PTV_data(MgSiO3, params, PTV, PTV_covariances, verbose=False)
print(fitted_eos.popt, fitted_eos.pcov)

fig = burnman.nonlinear_fitting.corner_plot(fitted_eos.popt, fitted_eos.pcov)
plt.show()
'''

# Read in data from Kurnosov et al., 2017
# P (GPa) rho C11 C22 C33 C44 C55 C66 C12 C13 C23
data = np.loadtxt('data/Kurnosov_et_al_2017.dat',
                  unpack=True)

pressures = data[0]*1.e9
temperatures = np.array([300.]*len(pressures))
densities = data[1]*1.e3
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

KS = np.empty_like(pressures)
KSe = np.empty_like(pressures)
for i in range(len(pressures)):
    density = densities[i]
    
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

    #plot_anisotropic_seismic_properties(static_bdg)
    #plt.savefig('bdg_anisotropy_{0}.pdf'.format(i))
    #plt.show()
        
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
    KS[i] = static_bdg.bulk_modulus_reuss
    KSe[i] = np.sqrt(var)


# Now we want to simultaneously fit the PTKS and PTrho data

# Here's the endmember MgSiO3 bridgmanite
KMFBZ_MgSiO3_bdg = KMFBZ_2017.mg_si_perovskite()
KMFBZ_FeAlO3_bdg = KMFBZ_2017.fe_al_perovskite()
KMFBZ_MgSiO3_bdg.name = 'MgSiO$_3$ (Kurnosov et al., 2017)'
KMFBZ_FeAlO3_bdg.name = 'FeAlO$_3$ (Kurnosov et al., 2017)'



KT_est = KS - KMFBZ_MgSiO3_bdg.evaluate(['K_S'], pressures, temperatures)[0] + KMFBZ_MgSiO3_bdg.evaluate(['K_T'], pressures, temperatures)[0]
KT_est_Murakami = KS_Murakami - KMFBZ_MgSiO3_bdg.evaluate(['K_S'], pressures_Murakami, temperatures_Murakami)[0] + KMFBZ_MgSiO3_bdg.evaluate(['K_T'], pressures_Murakami, temperatures_Murakami)[0]



# Now let's make a starting model for the solid solution
made_params = copy.deepcopy(KMFBZ_MgSiO3_bdg.params)
made_params['formula'] = burnman.processchemistry.dictionarize_formula('Mg0.9Fe0.1Al0.1Si0.9O3')
made_params['molar_mass'] = burnman.processchemistry.formula_mass(made_params['formula'])
made_params['V_0'] = 2.474e-5
made_params['K_0'] = 247.e9
made_params['Kprime_0'] = 3.7

made_bdg = burnman.Mineral(params = made_params)
volumes = made_params['molar_mass']/densities


fig = plt.figure(figsize=(1,1))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
fig2 = plt.figure(figsize=(12,5))
ax2 = [fig2.add_subplot(1, 2, i) for i in range(1, 3)]
fig3 = plt.figure(figsize=(6,4))
ax3 = [fig3.add_subplot(1, 1, i) for i in range(1, 2)]

colors = [next(ax2[0]._get_lines.prop_cycler)['color'] for i in range(10)]



pressures_MgSiO3 = np.linspace(1.e5, 100.e9, 101)
temperatures_MgSiO3 = np.array([300.]*len(pressures_MgSiO3))
Vs, KTs = MgSiO3.evaluate(['V', 'K_T'], pressures_MgSiO3, temperatures_MgSiO3)
label=MgSiO3.name
color=colors[0]
ax2[0].plot(pressures_MgSiO3/1.e9, Vs*1.e6, label=label, color=color)
ax2[1].plot(pressures_MgSiO3/1.e9, KTs/1.e9, label=label, color=color)
ax[2].plot(Vs*1.e6, KTs/1.e9, label=label, color=color)
ax3[0].plot(pressures_MgSiO3/KTs, 1./np.gradient(KTs, pressures_MgSiO3, edge_order=2), label=label, color=color)


pressures_Fiquet, Perr_Fiquet, volumes_Fiquet, Verr_Fiquet = np.loadtxt('data/Fiquet_2000_MgSiO3_pv_PV.dat', unpack=True)
pressures_Fiquet = pressures_Fiquet*1.e9 
Perr_Fiquet = Perr_Fiquet*1.e9

Z = 4.
volumes_Fiquet = volumes_Fiquet/1.e30*burnman.constants.Avogadro/Z
Verr_Fiquet = Verr_Fiquet/1.e30*burnman.constants.Avogadro/Z


label='MgSiO$_3$ (Fiquet et al., 2000)'
ax2[0].errorbar(pressures_Fiquet/1.e9, volumes_Fiquet*1.e6, xerr = Perr_Fiquet/1.e9, yerr=Verr_Fiquet*1.e6, label=label, color=color, linestyle='None', capsize=3, elinewidth=2)

label='MgSiO$_3$ (Murakami et al., 2007)'
ax2[1].errorbar(pressures_Murakami/1.e9, KT_est_Murakami/1.e9, yerr=KSe_Murakami/1.e9, label=label, color=color, linestyle='None', capsize=3, elinewidth=2)

# Finally, let's make a model with the EoS of Stacey and Davis
params = {'equation_of_state': 'rkprime',
          'V_0': 2.474e-5,
          'K_0': 247.e9,
          'Kprime_0': 3.7,
          'Kprime_inf': 0.8, # best fitting value is less than the thermodynamic bound of 1.6, and still doesn't quite fit the data at high pressure
          'molar_mass': made_bdg.params['molar_mass'],
          'n': made_bdg.params['n'],
          'formula': made_bdg.params['formula']}


params = {'name': '(MgSi)$_{{0.9}}$(FeAl)$_{{0.1}}$O$_3$ (this study)',
          'equation_of_state': 'rkprime',
          'V_0': 2.474e-5,
          'K_0': 247.e9,
          'Kprime_0': 3.9,
          'Kprime_inf': 2.9, 
          'molar_mass': made_bdg.params['molar_mass'],
          'n': made_bdg.params['n'],
          'formula': made_bdg.params['formula']}

FMAS_bdg_ss = burnman.Mineral(params = params)

params = {'name': 'FeAlO$_3$ (this study)',
          'equation_of_state': 'rkprime',
          'V_0': 2.76e-5,
          'K_0': 199.e9,
          'Kprime_0': 3.9,
          'Kprime_inf': 2.9,
          'molar_mass': made_bdg.params['molar_mass'],
          'n': made_bdg.params['n'],
          'formula': made_bdg.params['formula']}

made_FeAlO3 = burnman.Mineral(params = params)


pressures_ss = np.linspace(1.e5, 100.e9, 101)
temperatures_ss = np.array([300.]*len(pressures_ss))
FMAS_bdg_ss_Vs, FMAS_bdg_ss_KTs = FMAS_bdg_ss.evaluate(['V', 'K_T'], pressures_ss, temperatures_ss)
label=FMAS_bdg_ss.name
color=colors[1]
ax2[0].plot(pressures_ss/1.e9, FMAS_bdg_ss_Vs*1.e6, label=label, color=color)
ax2[1].plot(pressures_ss/1.e9, FMAS_bdg_ss_KTs/1.e9, label=label, color=color)
ax[2].plot(FMAS_bdg_ss_Vs*1.e6, FMAS_bdg_ss_KTs/1.e9, label=label, color=color)
#ax3[0].plot(pressures_ss/FMAS_bdg_ss_KTs, 1./np.gradient(FMAS_bdg_ss_KTs, pressures_ss, edge_order=2), label=label, color=color)


per = burnman.minerals.SLB_2011.periclase()
KTs_per = per.evaluate(['K_T'], pressures_ss, temperatures_ss)[0]
ax3[0].plot(pressures_ss/FMAS_bdg_ss_KTs, 1./np.gradient(KTs_per, pressures_ss, edge_order=2), label='MgO (SLB2011)', color=colors[2])


label='(MgSi)$_{0.9}$(FeAl)$_{0.1}$O$_3$ (Kurnosov et al., 2017)'
ax2[0].scatter(pressures/1.e9, volumes*1.e6, label=label, color=color)
ax2[1].errorbar(pressures/1.e9, KT_est/1.e9, yerr=KSe/1.e9, label=label, color=color, linestyle='None', capsize=3, elinewidth=2)
ax[2].errorbar(1.e6*volumes, KT_est/1.e9, yerr=KSe/1.e9, label=label, color=color, linestyle='None', capsize=3, elinewidth=2)
ax3[0].scatter(pressures/KT_est, 1./np.gradient(KT_est, pressures, edge_order=2), label=label, color=color)


pressures_FeAlO3 = np.linspace(1.e5, 69.2e9, 101)
temperatures_FeAlO3 = np.array([300.]*len(pressures_FeAlO3))
ax2[0].plot(pressures_FeAlO3/1.e9,
            (0.1*KMFBZ_FeAlO3_bdg.evaluate(['V'], pressures_FeAlO3, temperatures_FeAlO3)[0] +
             0.9*KMFBZ_MgSiO3_bdg.evaluate(['V'], pressures_FeAlO3, temperatures_FeAlO3)[0])*1.e6, color=color, linestyle='-.', label=label)


color=colors[2]
label='(MgSi)$_{0.923}$(AlAl)$_{0.077}$O$_3$ (Daniel et al., 2001)'
P_Daniel, V_Daniel = np.loadtxt('data/Daniel_et_al_2001_Al_pv_77.dat', unpack=True)

ax2[0].scatter(P_Daniel, V_Daniel/1.e30*burnman.constants.Avogadro/4.*1.e6, marker=(4, 1, 0), label=label, color=color)


prem = burnman.seismic.PREM()
Pprem, Kprem = prem.evaluate(['pressure', 'K'], np.linspace(6371000. - 5600000.,
                                                            6371000. - 3630000., 21))
ax3[0].plot(Pprem/Kprem, 1./np.gradient(Kprem, Pprem, edge_order=2), label='PREM', color='black')

color='black'
Kps = [5./3., 2., 3., 4., 5.]

ax3[0].plot([0., 3./5.], [0., 3./5.])
for Kp in Kps:
    ax3[0].text(0.005, 1./Kp - 0.035, '$K\'$ = {0:.2f}'.format(Kp))
    ax3[0].plot([0., 1./Kp], [1./Kp, 1./Kp], linestyle='--', color=color)
ax3[0].set_xlim(0,)

from scipy.interpolate import interp1d
pressures_ss = np.linspace(1.e5, 150.e9, 101)
temperatures_ss = np.array([300.]*len(pressures_ss))
FMAS_bdg_ss_Vs, FMAS_bdg_ss_KTs = FMAS_bdg_ss.evaluate(['V', 'K_T'], pressures_ss, temperatures_ss)

f_P = interp1d(FMAS_bdg_ss_Vs[::-1], pressures_ss[::-1], kind='cubic')
f_KT = interp1d(FMAS_bdg_ss_Vs[::-1], FMAS_bdg_ss_KTs[::-1], kind='cubic')
f_Kprime = interp1d(FMAS_bdg_ss_Vs[::-1], np.gradient(FMAS_bdg_ss_KTs[::-1], pressures_ss[::-1], edge_order=2), kind='cubic')
MgSiO3.set_state(1.e5, 300.)


label=made_FeAlO3.name
elastic_pressures_FeAlO3 = (pressures_MgSiO3 + 10.*(f_P(Vs) - pressures_MgSiO3))
volumes_FeAlO3 = Vs
ax2[0].plot(elastic_pressures_FeAlO3/1.e9, volumes_FeAlO3*1.e6, color=colors[3], label=label)

volumes_ss = np.empty_like(pressures_ss)
KT_ss = np.empty_like(pressures_ss)
for i, pressure in enumerate(pressures_ss):
    s = solution(pressure, 300.,
                 F_xs = 0., p_xs = 0.,
                 x_a = 0.9, a = MgSiO3, b = FeAlO3_Caracas, cluster_size = 2.)
    volumes_ss[i] = s.V
    KT_ss[i] = s.K_T

ax2[0].plot(pressures_ss/1.e9, volumes_ss*1.e6, color=colors[4], label='elastic')
ax2[1].plot(pressures_ss/1.e9, KT_ss/1.e9, color=colors[4], label='elastic')



fitted_eos = burnman.eos_fitting.fit_PTV_data(made_FeAlO3, ['V_0', 'K_0', 'Kprime_0'],
                                              np.array([elastic_pressures_FeAlO3,
                                                        [300.]*len(pressures_FeAlO3),
                                                        volumes_FeAlO3]).T,
                                              verbose=False)
burnman.tools.pretty_print_values(fitted_eos.popt, fitted_eos.pcov, fitted_eos.fit_params)
#exit()

print(MgSiO3.params)
print(FMAS_bdg_ss.params)
print(made_FeAlO3.params)


ax2[0].plot(pressures_ss/1.e9, made_FeAlO3.evaluate(['V'], pressures_ss, temperatures_ss)[0]*1.e6, color=colors[3], linestyle=':')
ax2[1].plot(pressures_ss/1.e9, made_FeAlO3.evaluate(['K_T'], pressures_ss, temperatures_ss)[0]/1.e9, color=colors[3], linestyle=':')

label=KMFBZ_FeAlO3_bdg.name
ax2[0].plot(pressures_FeAlO3/1.e9, KMFBZ_FeAlO3_bdg.evaluate(['V'], pressures_FeAlO3, temperatures_FeAlO3)[0]*1.e6, color=color, linestyle='-.', label=label)


P_FeAlO3 = (pressures_MgSiO3 + 10.*(f_P(Vs) - pressures_MgSiO3))/1.e9
K_FeAlO3 = (KTs + 10.*(f_KT(Vs) - KTs))/1.e9
ax2[1].plot([P for i, P in enumerate(P_FeAlO3) if P_FeAlO3[i] < 60.],
            [K for i, K in enumerate(K_FeAlO3) if P_FeAlO3[i] < 60.],
            color=colors[3], label=label)
P_FeAlO3 = (pressures_MgSiO3 + 10.*(f_P(Vs)+1.e9 - pressures_MgSiO3))/1.e9
K_FeAlO3 = (KTs + 10.*(f_KT(Vs) - KTs))/1.e9
#ax2[1].plot([P for i, P in enumerate(P_FeAlO3) if P_FeAlO3[i] < 70.],
#            [K for i, K in enumerate(K_FeAlO3) if P_FeAlO3[i] < 70.],
#            color=colors[3], linestyle='--', label=label+' ($P_{{excess}}$ = 1 GPa)')


ax2[0].set_xlabel('Pressure (GPa)')
ax2[0].set_ylabel('Volume (cm$^3$/mol)')

ax2[1].set_xlim(0., 100.)
ax2[1].set_ylim(180., 620.)

ax2[1].set_xlabel('Pressure (GPa)')
ax2[1].set_ylabel('$K_T$ (GPa)')

ax[2].set_xlabel('Volume (cm$^3$/mol)')
ax[2].set_ylabel('$K_T$ (GPa)')

ax3[0].set_xlim(0.,0.6)
ax3[0].set_ylim(0.,0.6)

ax3[0].set_xlabel('$P/K$')
ax3[0].set_ylabel('1/$K\'$')


ax2[0].legend(loc='best')
ax2[1].legend(loc='best')
ax[2].legend(loc='lower right')
ax3[0].legend(loc='lower right')


fig2.tight_layout()
fig2.savefig("bridgmanite_eos.pdf", bbox_inches='tight', dpi=100)

fig3.savefig("bridgmanite_Kprime.pdf", bbox_inches='tight', dpi=100)
plt.show()

exit()


pressures = np.linspace(1.e5, 60.e9, 101)
temperatures = [300.]*len(pressures)

for m in [KMFBZ_2017.mg_si_perovskite(), KMFBZ_2017.fe_si_perovskite(), KMFBZ_2017.fe_al_perovskite(), KMFBZ_2017.al_al_perovskite()]:
    plt.plot(pressures, m.evaluate(['V'], pressures, temperatures)[0], label=m.name)
plt.legend(loc='best')
plt.show()



pressures = np.linspace(-13.e9, 60.e9, 101)
temperatures = [300.]*len(pressures)


V_MgSiO3 = MgSiO3_bdg2.evaluate(['V'], pressures, temperatures)[0]
V_made_bdg = made_bdg.evaluate(['V'], pressures, temperatures)[0]
V_FMAS_bdg_ss = FMAS_bdg_ss.evaluate(['V'], pressures, temperatures)[0]
V_FeAlO3 = KMFBZ_FeAlO3_bdg.evaluate(['V'], pressures, temperatures)[0]

FeAlO3_mech = burnman.CombinedMineral([HP_2011_ds62.hem(), HP_2011_ds62.cor()], [0.5, 0.5])
V_FeAlO3_mech = FeAlO3_mech.evaluate(['V'], pressures, temperatures)[0]



fig = plt.figure()
ax = [fig.add_subplot(2, 1, i) for i in range(1,3)]

ax2[0].plot(pressures, V_MgSiO3, label='MgSiO$_3$')
ax2[0].plot(pressures, V_FeAlO3, label='FeAlO$_3$ (Kurnosov et al., 2017)')
ax2[0].plot(pressures, V_FeAlO3_mech, label='FeAlO$_3$ (Kurnosov et al., 2017)')
ax2[0].plot(pressures, V_MgSiO3 + 10.*(V_FMAS_bdg_ss - V_MgSiO3), label='FeAlO$_3$ (Gibbs)')




def delta_V(pressure, volume, temperature, m):
    m.set_state(pressure, temperature)
    return volume - m.V

mix_pressures = np.empty_like(V_MgSiO3)

for i, V in enumerate(V_MgSiO3):
    mix_pressures[i] = brentq(delta_V,
                              pressures[i], pressures[i] + 10.e9,
                              args=(V, temperatures[i], FMAS_bdg_ss))


ax2[0].plot(mix_pressures, V_MgSiO3, label='mix')
ax2[0].plot(pressures + (mix_pressures - pressures)*10., V_MgSiO3, label='FeAlO$_3$ (Helmholtz)')



params = {'equation_of_state': 'rkprime',
          'V_0': 2.79e-5,
          'K_0': 215.e9,
          'Kprime_0': 2.0,
          'Kprime_inf': 1.6, # best fitting value of ~1.4 is less than the thermodynamic bound of 1.6, and still doesn't (quite) fit the data
          'molar_mass': KMFBZ_FeAlO3_bdg.params['molar_mass'],
          'n': KMFBZ_FeAlO3_bdg.params['n'],
          'formula': KMFBZ_FeAlO3_bdg.params['formula']}
made_FeAlO32 = burnman.Mineral(params = params)
ax2[0].plot(pressures*1.5, made_FeAlO32.evaluate(['V'], pressures*1.5, [300.]*len(pressures*1.5))[0], label='FeAlO3 made')

ax2[0].set_xlim(0,)
ax2[0].legend(loc='upper right')

ax2[1].plot(mix_pressures, FMAS_bdg_ss.evaluate(['gibbs'], mix_pressures, [300.]*len(mix_pressures))[0])


plt.show()
