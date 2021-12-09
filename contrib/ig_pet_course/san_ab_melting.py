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

fig = plt.figure(figsize=(8, 3))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
ax[0].plot(x_ab_fsps, Ts)
ax[0].plot(x_ab_liqs, Ts)
ax[1].plot(x_ab_fsps, H_fsps/1.e3, color='k')
ax[1].plot(x_ab_liqs, H_liqs/1.e3, color='k')

free_compositional_vectors = [{'K': -1., 'Na': 1.}]

Ts = np.linspace(1320., 1420., 6)

H_fsps = np.empty_like(x_ab_fsps)
H_liqs = np.empty_like(x_ab_fsps)


for i, T in enumerate(Ts):
    for j, x in enumerate(x_ab_fsps):
        liq.set_composition([x, 1.-x])
        liq.set_state(P, T)
        H_liqs[j] = liq.molar_enthalpy/1.e3
        fsp.set_composition([x, 1.-x])
        fsp.set_state(P, T)
        H_fsps[j] = fsp.molar_enthalpy/1.e3

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

ax[1].legend(ncol=2)

for i in range(2):
    ax[i].set_xlabel('$x_{{ab}}$')
    ax[i].set_xlim(0., 1.)

ax[0].text(0.5, 1380., 'melt')
ax[0].text(0.5, 1280., 'fsp')
ax[0].set_ylabel('Temperature (K)')
ax[1].set_ylabel('Enthalpy (kJ/mol)')

ax[0].set_ylim(1250., 1450.)
ax[1].set_ylim(-3740., -3500.)
fig.set_tight_layout(True)
fig.savefig('figures/or_ab_melting.pdf')
plt.show()
