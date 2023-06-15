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
from scipy.optimize import minimize

import burnman
from burnman import equilibrate
from burnman.minerals import SLB_2011
from burnman.tools.equilibration import equilibrate
from scipy.integrate import cumulative_trapezoid

fo = SLB_2011.forsterite()

temperatures_1 = np.linspace(300.0, 1600.0, 501)
pressures_1 = np.empty_like(temperatures_1)
entropies_1 = np.empty_like(temperatures_1)
volumes_1 = np.empty_like(temperatures_1)

fo.set_state(1.0e5, temperatures_1[0])
V0 = fo.V
# Path 1: Isochoric
print("Path 1")
for i, T in enumerate(temperatures_1):
    pressures_1[i] = fo.method.pressure(T, V0, fo.params)
    fo.set_state(pressures_1[i], T)
    entropies_1[i] = fo.S
    volumes_1[i] = fo.V

dS_1 = np.gradient(entropies_1, edge_order=2)
dV_1 = np.gradient(volumes_1, edge_order=2)

# Path 2: Isobaric then isentropic
print("Path 2")
assemblage = burnman.Composite([fo], [1.0], "molar", "forsterite")

pressures_2b = np.linspace(1.0e5, pressures_1[-1], 501)
equality_constraints = [["P", pressures_2b], ["S", entropies_1[-1]]]
sols = equilibrate(assemblage.formula, assemblage, equality_constraints)
temperatures_2b = np.array([sol.assemblage.temperature for sol in sols[0]])
entropies_2b, volumes_2b = fo.evaluate(["S", "V"], pressures_2b, temperatures_2b)

temperatures_2a = np.linspace(temperatures_1[0], temperatures_2b[0], 501)
pressures_2a = temperatures_2a * 0.0 + 1.0e5
entropies_2a, volumes_2a = fo.evaluate(["S", "V"], pressures_2a, temperatures_2a)

temperatures_2 = np.concatenate((temperatures_2a, temperatures_2b))
pressures_2 = np.concatenate((pressures_2a, pressures_2b))
entropies_2 = np.concatenate((entropies_2a, entropies_2b))
dS_2 = np.concatenate(
    (np.gradient(entropies_2a, edge_order=2), np.gradient(entropies_2b, edge_order=2))
)
dV_2 = np.concatenate(
    (np.gradient(volumes_2a, edge_order=2), np.gradient(volumes_2b, edge_order=2))
)


# Path 3: Isentropic then isobaric
print("Path 3")
pressures_3a = np.linspace(1.0e5, pressures_1[-1], 501)[:-1]
equality_constraints = [["P", pressures_3a], ["S", entropies_1[0]]]
sols = equilibrate(assemblage.formula, assemblage, equality_constraints)

temperatures_3a = np.array([sol.assemblage.temperature for sol in sols[0]])
entropies_3a, volumes_3a = fo.evaluate(["S", "V"], pressures_3a, temperatures_3a)


temperatures_3b = np.linspace(temperatures_3a[-1], temperatures_1[-1], 501)[1:]
pressures_3b = temperatures_3b * 0.0 + pressures_1[-1]
entropies_3b, volumes_3b = fo.evaluate(["S", "V"], pressures_3b, temperatures_3b)

temperatures_3 = np.concatenate((temperatures_3a, temperatures_3b))
pressures_3 = np.concatenate((pressures_3a, pressures_3b))
entropies_3 = np.concatenate((entropies_3a, entropies_3b))
dS_3 = np.concatenate(
    (np.gradient(entropies_3a, edge_order=2), np.gradient(entropies_3b, edge_order=2))
)
dV_3 = np.concatenate(
    (np.gradient(volumes_3a, edge_order=2), np.gradient(volumes_3b, edge_order=2))
)


fig = plt.figure(figsize=(9, 3))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

Q_1 = cumulative_trapezoid(temperatures_1 * dS_1, initial=0.0)
Q_2 = cumulative_trapezoid(temperatures_2 * dS_2, initial=0.0)
Q_3 = cumulative_trapezoid(temperatures_3 * dS_3, initial=0.0)


W_1 = cumulative_trapezoid(-pressures_1 * dV_1, initial=0.0)
W_2 = cumulative_trapezoid(-pressures_2 * dV_2, initial=0.0)
W_3 = cumulative_trapezoid(-pressures_3 * dV_3, initial=0.0)


ax[0].plot(temperatures_3, pressures_3 / 1.0e9, label="Path 3", linestyle=":")
ax[0].plot(temperatures_2, pressures_2 / 1.0e9, label="Path 2", linestyle="--")
ax[0].plot(temperatures_1, pressures_1 / 1.0e9, label="Path 1")

ax[1].plot(W_3 / 1000.0, Q_3 / 1000.0, label="Path 3", linestyle=":")
ax[1].plot(W_2 / 1000.0, Q_2 / 1000.0, label="Path 2", linestyle="--")
ax[1].plot(W_1 / 1000.0, Q_1 / 1000.0, label="Path 1")

E_1 = fo.evaluate(["molar_internal_energy"], pressures_1, temperatures_1)[0]
E_2 = fo.evaluate(["molar_internal_energy"], pressures_2, temperatures_2)[0]
E_3 = fo.evaluate(["molar_internal_energy"], pressures_3, temperatures_3)[0]

# ax[2].plot(temperatures_3, (E_3 - E_3[0])/1000., label='Path 3')
# ax[2].plot(temperatures_2, (E_2 - E_2[0])/1000., label='Path 2')
# ax[2].plot(temperatures_1, (E_1 - E_1[0])/1000., label='Path 1')

ax[2].plot(
    temperatures_3, (entropies_3 - entropies_3[0]), label="Path 3", linestyle=":"
)
ax[2].plot(
    temperatures_2, (entropies_2 - entropies_2[0]), label="Path 2", linestyle="--"
)
ax[2].plot(temperatures_1, (entropies_1 - entropies_1[0]), label="Path 1")

# Make a subplot zoom-in of the entropies
# These are in unitless fractions of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.86, 0.3, 0.1, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])

ax2.plot(temperatures_3, (entropies_3 - entropies_3[0]), label="Path 3", linestyle=":")
ax2.plot(temperatures_2, (entropies_2 - entropies_2[0]), label="Path 2", linestyle="--")
ax2.plot(temperatures_1, (entropies_1 - entropies_1[0]), label="Path 1")
ax2.set_xlim(1400.0, 1600.0)
ax2.set_ylim(240.0, 280.0)


ax[0].set_xlabel("Temperature (K)")
ax[0].set_ylabel("Pressure (GPa)")

ax[1].set_xlabel("Work (kJ/mol)")
ax[1].set_ylabel("Heat (kJ/mol)")

ax[2].set_xlabel("Temperature (K)")
ax[2].set_ylabel("Entropies (J/K/mol)")

ax[2].legend()

fig.set_tight_layout(True)
fig.savefig("figures/fo_heat_work.pdf")
plt.show()
