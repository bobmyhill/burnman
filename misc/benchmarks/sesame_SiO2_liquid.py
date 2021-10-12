# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.
import numpy as np
from scipy.constants import physical_constants
import matplotlib.pyplot as plt

import burnman_path  # adds the local burnman directory to the path
import burnman
assert burnman_path  # silence pyflakes warning

params = {'equation_of_state': 'sesame',
          'directory': 'sesame_301_SiO2',
          'formula': {'Si': 1., 'O': 2.},
          'name': 'SiO2'}
SiO2_liquid = burnman.Mineral(params)

burnman.tools.eos.check_eos_consistency(SiO2_liquid, P=1.e9, T=300.,
                                        including_shear_properties=False,
                                        verbose=True)

eVoverK = physical_constants['kelvin-electron volt relationship'][0]

# Plot recreating Figure 2 from Sjostrom and Crockett (2017)
# https://doi.org/10.1063/1.4971544
fig = plt.figure(figsize=(8, 4))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

pressures = np.logspace(11, 14, 101)
for eV in [100., 50., 20., 10., 5]:
    T = eV/eVoverK
    temperatures = [T] * len(pressures)
    densities = SiO2_liquid.evaluate(['rho'], pressures, temperatures)[0]
    ax[0].plot(densities/1.e3, pressures/1.e9,
               label=f'T = {eV} eV ({int(T)} K)')

ax[0].set_yscale('log')
ax[0].set_xlim(5, 16)
ax[0].set_ylim(1e2, 1e5)
ax[0].legend()
pressures = np.linspace(100.e9, 3000.e9, 101)
for eV in [6.89, 5.17, 3.45, 1.72, 0.86]:
    T = eV/eVoverK
    temperatures = [T] * len(pressures)
    densities = SiO2_liquid.evaluate(['rho'], pressures, temperatures)[0]
    ax[1].plot(densities/1000., pressures/1.e9,
               label=f'T = {eV} eV ({int(T)} K)')

for i in range(2):
    ax[i].set_xlabel('Density (g/cc)')
    ax[i].set_ylabel('Pressure (GPa)')

ax[1].set_xlim(5, 11.5)
ax[1].set_ylim(0, 3000.)
ax[1].legend()

fig.set_tight_layout("True")
plt.show()
