# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
ferropericlase
--------------

This script creates an AnisotropicSolution object for ferropericlase.

"""

from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from ferropericlase_psi import psi_func, per_anisotropic_parameters
import burnman

from burnman import AnisotropicMineral

per = burnman.minerals.SLB_2011.periclase()

# Overwrite some scalar EoS properties to fit Karki data.
V0 = per.params['V_0']
K0 = per.params['K_0']
Kprime0 = per.params['Kprime_0']
gr0 = per.params['grueneisen_0']
q0 = per.params['q_0']

per.params['V_0'] = 1.00825474*V0
per.params['K_0'] = 0.992695934*K0
per.params['Kprime_0'] = 1.08827120*Kprime0
per.params['grueneisen_0'] = 1.12859229*gr0
per.params['q_0'] = 0.587861963*q0

wus = burnman.minerals.SLB_2011.wuestite()
a = np.cbrt(per.params['V_0'])
cell_parameters = np.array([a, a, a, 90, 90, 90])
m = AnisotropicMineral(per,
                       cell_parameters,
                       per_anisotropic_parameters(),
                       psi_func, orthotropic=True)


per_data = np.loadtxt('data/Karki_2000_periclase_CSijs.dat')
per_data_2 = np.loadtxt('data/isentropic_stiffness_tensor_periclase.dat')
parameters = []


assert burnman.tools.eos.check_anisotropic_eos_consistency(m)

m.set_state(1.e5, 300.)

fig = plt.figure(figsize=(4, 12))
fig2 = plt.figure(figsize=(4, 4))
ax = [fig.add_subplot(3, 1, i) for i in range(1, 4)]
ax.append(fig2.add_subplot(1, 1, 1))


temperatures = [300., 1000., 2000., 3000.]
for T in temperatures:

    # T, P, C11S, C12S, C44S, KT, V, betaTminusbetaS
    # every 2nd data point
    T_data = (np.array([[d[1], d[2], d[3], d[4]]
                       for d in per_data if np.abs(d[0] - T) < 1])/1.e9)[::2]

    Pmin = np.min(T_data[:, 0])*1.e9
    Pmax = np.max(T_data[:, 0])*1.e9
    pressures = np.linspace(Pmin, Pmax, 101)
    G_iso = np.empty_like(pressures)
    G_aniso = np.empty_like(pressures)
    C11 = np.empty_like(pressures)
    C12 = np.empty_like(pressures)
    C44 = np.empty_like(pressures)
    G_slb = np.empty_like(pressures)
    G_V = np.empty_like(pressures)
    G_R = np.empty_like(pressures)

    f = np.empty_like(pressures)
    dXdf = np.empty_like(pressures)

    for i, P in enumerate(pressures):

        per.set_state(P, T)
        m.set_state(P, T)
        C11[i] = m.isentropic_stiffness_tensor[0, 0]
        C12[i] = m.isentropic_stiffness_tensor[0, 1]
        C44[i] = m.isentropic_stiffness_tensor[3, 3]
        G_V[i] = m.isentropic_shear_modulus_voigt
        G_R[i] = m.isentropic_shear_modulus_reuss

        G_slb[i] = per.shear_modulus

    ax[0].plot(pressures/1.e9, C11/1.e9, label=f'{T} K')
    ax[1].plot(pressures/1.e9, C12/1.e9, label=f'{T} K')
    ax[2].plot(pressures/1.e9, C44/1.e9, label=f'{T} K')

    ln = ax[3].plot(pressures/1.e9, G_R/1.e9)
    ax[3].plot(pressures/1.e9, G_V/1.e9, color=ln[0].get_color(),
               label=f'{T} K')

    ax[3].fill_between(pressures/1.e9, G_R/1.e9, G_V/1.e9,
                       alpha=0.3, color=ln[0].get_color())

    ax[3].plot(pressures/1.e9, G_slb/1.e9, label=f'{T} K (SLB2011)',
               linestyle='--', color=ln[0].get_color())

    ax[0].scatter(T_data[:, 0], T_data[:, 1])
    ax[1].scatter(T_data[:, 0], T_data[:, 2])
    ax[2].scatter(T_data[:, 0], T_data[:, 3])

for i in range(4):
    ax[i].set_xlabel('Pressure (GPa)')

ax[0].set_ylabel('$C_{N 11}$ (GPa)')
ax[1].set_ylabel('$C_{N 12}$ (GPa)')
ax[2].set_ylabel('$C_{N 44}$ (GPa)')
ax[3].set_ylabel('$G$ (GPa)')

for i in range(4):
    ax[i].legend()

fig.set_tight_layout(True)
fig.savefig('periclase_stiffness_tensor.pdf')
fig2.set_tight_layout(True)
fig2.savefig('periclase_shear_modulus.pdf')
plt.show()

fig = plt.figure(figsize=(4, 4))
ax = [fig.add_subplot(1, 1, 1)]

temperatures = np.linspace(10., 2000., 101)
P = 1.e5
for i, T in enumerate(temperatures):
    m.set_state(P, T)
    per.set_state(P, T)
    C11[i] = m.isentropic_stiffness_tensor[0, 0]
    C12[i] = m.isentropic_stiffness_tensor[0, 1]
    C44[i] = m.isentropic_stiffness_tensor[3, 3]
    G_V[i] = m.isentropic_shear_modulus_voigt
    G_R[i] = m.isentropic_shear_modulus_reuss

    G_slb[i] = per.shear_modulus
ax[0].plot(temperatures, C11/1.e9, label='$C_{N 11}$')
ax[0].plot(temperatures, C12/1.e9, label='$C_{N 12}$')
ax[0].plot(temperatures, C44/1.e9, label='$C_{44}$')
ln = ax[0].plot(temperatures, G_R/1.e9, color=ln[0].get_color(),
                label='$G$')
ax[0].plot(temperatures, G_V/1.e9, color=ln[0].get_color())
ax[0].fill_between(temperatures, G_R/1.e9, G_V/1.e9,
                   alpha=0.3, color=ln[0].get_color(), zorder=2)
ax[0].plot(temperatures, G_slb/1.e9, label='$G$ (SLB2011)',
           color=ln[0].get_color(), linestyle='--', zorder=1)

# T, PGPa, Perr, rho, rhoerr, C11S, C11Serr, C12S, C12Serr, C44S, C44Serr
LP_data = np.array([[d[0], d[5], d[7], d[9]]
                    for d in per_data_2 if d[1] < 0.1])

ax[0].scatter(LP_data[:, 0], LP_data[:, 1])
ax[0].scatter(LP_data[:, 0], LP_data[:, 2])
ax[0].scatter(LP_data[:, 0], LP_data[:, 3])

ax[0].set_xlabel('Temperature (K)')
ax[0].set_ylabel('Elastic modulus (GPa)')


print('The following parameters were used for the volumetric part of '
      f'the isotropic model: $V_0$: {m.params["V_0"]*1.e6:.5f} cm$^3$/mol, '
      f'$K_0$: {m.params["K_0"]/1.e9:.5f} GPa, '
      f'$K\'_0$: {m.params["Kprime_0"]:.5f}, '
      f'$\Theta_0$: {m.params["Debye_0"]:.5f} K, '
      f'$\gamma_0$: {m.params["grueneisen_0"]:.5f}, '
      f'and $q_0$: {m.params["q_0"]:.5f}.')

ax[0].legend()

fig.set_tight_layout(True)
fig.savefig('periclase_properties_1bar.pdf')
plt.show()
