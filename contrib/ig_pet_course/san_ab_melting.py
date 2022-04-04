# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
an_di_melting
-------------
"""

from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt

import burnman
from burnman import equilibrate
from burnman.minerals import HGP_2018_ds633
from copy import copy

abL = HGP_2018_ds633.abL()
kspL = HGP_2018_ds633.kspL()

abh = HGP_2018_ds633.abh()
san = HGP_2018_ds633.san()

P = 0.22e8  # 0.22 kbar

# Part of Green et al. (2018) melt model
liq = burnman.SolidSolution(name='ideal abL-kspL',
                            solution_type='symmetric',
                            endmembers=[[abL, '[Na]1'],
                                        [kspL, '[K]1']],
                            energy_interaction=[[-6.e3]],
                            volume_interaction=[[3.e-5]])

# Fsp(C1) model
fsp = burnman.SolidSolution(name='C1 ab-san',
                            solution_type='asymmetric',
                            endmembers=[[abh, '[Na]1'],
                                        [san, '[K]1']],
                            energy_interaction=[[25.1e3]],
                            entropy_interaction=[[10.8]],
                            volume_interaction=[[0.338e-5]],
                            alphas=[0.643, 1.])


temperatures = np.linspace(300., 921.0, 201)
fsp1 = copy(fsp)
fsp2 = copy(fsp)

fsp1.set_composition([0.01, 0.99])
fsp2.set_composition([0.99, 0.01])
assemblage = burnman.Composite([fsp1, fsp2])
fsp.set_composition([0.6575, 0.3425])
composition = fsp.formula
equality_constraints = [['P', 1.e5],
                        ['T', temperatures]]
sols, prm = equilibrate(composition, assemblage,
                        equality_constraints, verbose=True)
Ts_solvus = [sol.assemblage.temperature
             for sol in sols if sol.success]
Ts_solvus.extend(Ts_solvus[::-1])
xs_solvus = [sol.assemblage.phases[0].molar_fractions[0]
             for sol in sols if sol.success]
xs_solvus.extend([sol.assemblage.phases[1].molar_fractions[0]
                  for sol in sols if sol.success][::-1])

#plt.plot(xs, Ts)
#plt.show()
#exit()

assemblage = burnman.Composite([liq, fsp])
assemblage.set_state(P, 1473.15)

x_ab_fsps = np.linspace(0.001, 0.999, 101)
x_ab_liqs = np.empty_like(x_ab_fsps)
Ts = np.empty_like(x_ab_fsps)
H_fsps = np.empty_like(x_ab_fsps)
H_liqs = np.empty_like(x_ab_fsps)

for i, x in enumerate(x_ab_fsps):
    liq.set_composition([x, 1.-x])
    fsp.set_composition([x, 1.-x])

    composition = fsp.formula

    equality_constraints = [('phase_fraction', (liq, np.array([0.0]))),
                            ('P', P)]

    sol, prm = equilibrate(composition, assemblage, equality_constraints,
                           verbose=False)
    x_ab_liqs[i] = sol.assemblage.phases[0].molar_fractions[0]
    Ts[i] = sol.assemblage.temperature
    H_liqs[i] = sol.assemblage.phases[0].molar_enthalpy
    H_fsps[i] = sol.assemblage.phases[1].molar_enthalpy

fig_all = plt.figure(figsize=(4, 3))
ax_all = [fig_all.add_subplot(1, 1, 1)]

fig_G = plt.figure(figsize=(4, 3))
ax_G = [fig_G.add_subplot(1, 1, 1)]

fig = plt.figure(figsize=(8, 3))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
fig2 = plt.figure(figsize=(8, 3))
ax2 = [fig2.add_subplot(1, 2, i) for i in range(1, 3)]
ax[0].plot(x_ab_fsps, Ts, color='k')
ax[0].plot(x_ab_liqs, Ts, color='k')
ax_all[0].plot(x_ab_fsps, Ts, color='k')
ax_all[0].plot(x_ab_liqs, Ts, color='k')
ax_all[0].plot(xs_solvus, Ts_solvus)

ax2[0].plot(x_ab_fsps, Ts, color='k')
ax2[0].plot(x_ab_liqs, Ts, color='k')
ax[1].plot(x_ab_fsps, H_fsps/1.e3, color='k')
ax[1].plot(x_ab_liqs, H_liqs/1.e3, color='k')

free_compositional_vectors = [{'K': -1., 'Na': 1.}]

Ts = np.linspace(1320., 1420., 6)

H_fsps = np.empty_like(x_ab_fsps)
H_liqs = np.empty_like(x_ab_fsps)
G_fsps = np.empty_like(x_ab_fsps)
G_liqs = np.empty_like(x_ab_fsps)


for i, T in enumerate(Ts):
    for j, x in enumerate(x_ab_fsps):
        liq.set_composition([x, 1.-x])
        liq.set_state(P, T)
        H_liqs[j] = liq.molar_enthalpy/1.e3
        G_liqs[j] = liq.molar_gibbs/1.e3
        fsp.set_composition([x, 1.-x])
        fsp.set_state(P, T)
        H_fsps[j] = fsp.molar_enthalpy/1.e3
        G_fsps[j] = fsp.molar_gibbs/1.e3

    xHs = []

    liq.set_composition([0.2, 0.8])
    fsp.set_composition([0.1, 0.9])

    composition = fsp.formula

    equality_constraints = [('phase_fraction', (liq, np.array([0.0]))),
                            ('P', P),
                            ('T', T)]

    sol, prm = equilibrate(composition, assemblage, equality_constraints,
                           free_compositional_vectors,
                           verbose=False)
    H_liq1 = sol.assemblage.phases[0].molar_enthalpy
    H_fsp1 = sol.assemblage.phases[1].molar_enthalpy
    x_liq1 = sol.assemblage.phases[0].molar_fractions[0]
    x_fsp1 = sol.assemblage.phases[1].molar_fractions[0]

    if sol.success:
        xHs.extend([[x_ab_fsps[i], H_fsps[i]]
                    for i in range(len(x_ab_fsps))
                    if x_ab_fsps[i] < x_fsp1])
        xHs.extend([[x_fsp1, H_fsp1/1.e3],
                    [x_liq1, H_liq1/1.e3]])
        xHs.extend([[x_ab_fsps[i], H_liqs[i]]
                    for i in range(len(x_ab_fsps))
                    if x_ab_fsps[i] > x_liq1])

    xHs = np.array(xHs)

    liq.set_composition([0.8, 0.2])
    fsp.set_composition([0.9, 0.1])

    composition = fsp.formula

    equality_constraints = [('phase_fraction', (liq, np.array([0.0]))),
                            ('P', P),
                            ('T', T)]

    sol, prm = equilibrate(composition, assemblage, equality_constraints,
                           free_compositional_vectors,
                           verbose=False)
    H_liq2 = sol.assemblage.phases[0].molar_enthalpy
    H_fsp2 = sol.assemblage.phases[1].molar_enthalpy
    x_liq2 = sol.assemblage.phases[0].molar_fractions[0]
    x_fsp2 = sol.assemblage.phases[1].molar_fractions[0]

    if sol.success and x_liq2 < x_fsp2:
        xHs = [xH for xH in xHs if xH[0] < x_liq2]
        xHs.extend([[x_liq2, H_liq2/1.e3],
                    [x_fsp2, H_fsp2/1.e3]])
        xHs.extend([[x_ab_fsps[i], H_fsps[i]]
                    for i in range(len(x_ab_fsps))
                    if x_ab_fsps[i] > x_fsp2])

    xHs = np.array(xHs).T
    ln, = ax[1].plot(xHs[0], xHs[1], label=f'{T} K')

    if i == 0 or i == 3:
        if i == 0:
            ln, = ax2[1].plot(x_ab_fsps, G_fsps, label='solid')
            ln, = ax2[1].plot(x_ab_fsps, G_liqs,
                            color=ln.get_color(),
                            linestyle=':', label=f'melt ({T} K)')
        else:
            ln, = ax2[1].plot(x_ab_fsps, G_fsps)
            ln, = ax2[1].plot(x_ab_fsps, G_liqs, label=f'{T} K',
                            color=ln.get_color(),
                            linestyle=':')

        ax2[0].plot([0., 1.], [T, T], color=ln.get_color())

ax[1].legend(ncol=2)
ax2[1].legend(ncol=2)


ax_all[0].set_xlabel('$x_{{ab}}$')
ax_all[0].set_xlim(0., 1.)
    
for i in range(2):
    ax[i].set_xlabel('$x_{{ab}}$')
    ax[i].set_xlim(0., 1.)
    ax2[i].set_xlabel('$x_{{ab}}$')
    ax2[i].set_xlim(0., 1.)

ax[0].text(0.5, 1380., 'melt')
ax[0].text(0.5, 1280., 'fsp')
ax_all[0].text(0.5, 1380., 'melt')
ax_all[0].text(0.5, 1100., 'feldspar')
ax_all[0].text(0.5, 600., '2 feldspars')

ax[0].set_ylabel('Temperature (K)')
ax_all[0].set_ylabel('Temperature (K)')
ax2[0].text(0.5, 1380., 'melt')
ax2[0].text(0.5, 1280., 'fsp')
ax2[0].set_ylabel('Temperature (K)')

ax[1].set_ylabel('Enthalpy (kJ/mol)')
ax2[1].set_ylabel('Gibbs energy (kJ/mol)')

ax[0].set_ylim(1250., 1450.)
ax2[0].set_ylim(1250., 1450.)

ax[1].set_ylim(-3740., -3500.)
ax2[1].set_ylim(-4550., -4450.)

fig_all.set_tight_layout(True)
fig.set_tight_layout(True)
fig2.set_tight_layout(True)
fig_all.savefig('figures/or_ab_phase_diagram.pdf')
fig.savefig('figures/or_ab_melting_H.pdf')
fig2.savefig('figures/or_ab_melting_G.pdf')



xs = np.linspace(0., 1., 101)
Gs = np.empty_like(xs)
Ts = [600, 800, 1000]

for T in Ts:
    fsp.set_state(1.e5, T)

    for i, x in enumerate(xs):
        fsp.set_composition([x, 1.-x])
        Gs[i] = fsp.excess_gibbs
    ax_G[0].plot(xs, Gs/1000., label=f'{T} K')

ax_G[0].set_xlabel('$x_{{ab}}$')
ax_G[0].set_ylabel('Excess Gibbs energy (kJ/mol)')
ax_G[0].set_xlim(0., 1.)
ax_G[0].legend()
fig_G.set_tight_layout(True)
fig_G.savefig('figures/or_ab_excess_energies.pdf')


plt.show()
