from __future__ import absolute_import
from __future__ import print_function
import os.path
import sys

sys.path.insert(1, os.path.abspath("../.."))

import burnman
from burnman.minerals import DKS_2013_liquids, DKS_2013_solids, SLB_2011
from burnman import constants
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.optimize import fsolve

import matplotlib.pyplot as plt

from FMSH_endmembers_linearised import *


def eqm_T(T, mins, fs=[1.0, 1.0]):
    def delta_G(P):
        mins[0].set_state(P[0], T)
        mins[1].set_state(P[0], T)
        return mins[0].gibbs * fs[0] - mins[1].gibbs * fs[1]

    return fsolve(delta_G, [20.0e9])[0]


def eqm(P, mins, fs=[1.0, 1.0]):
    def delta_G(T):
        mins[0].set_state(P, T[0])
        mins[1].set_state(P, T[0])
        return mins[0].gibbs * fs[0] - mins[1].gibbs * fs[1]

    return fsolve(delta_G, [2000.0])[0]


def inv(mins, fs=[1.0, 1.0, 1]):
    def delta_G(args):
        P, T = args
        for m in mins:
            m.set_state(P, T)
        return [
            mins[0].gibbs * fs[0] - mins[1].gibbs * fs[1],
            mins[0].gibbs * fs[0] - mins[2].gibbs * fs[2],
        ]

    return fsolve(delta_G, [22.4e9, 2500.0])


fo_wad_melt = inv([fo, wad, Mg2SiO4L])
wad_ring_lm = inv([wad, ring, lm], [1.0, 1.0, 1.0])
wad_lm_melt = inv([wad, lm, Mg2SiO4L], [1.0, 1.0, 1.0])


fig = plt.figure(figsize=(12, 4))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

temperatures = np.linspace(1400, fo_wad_melt[1], 21)
pressures = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    pressures[i] = eqm_T(T, [fo, wad], [1.0, 1.0])

ax[0].plot(pressures / 1.0e9, temperatures, color="black")

temperatures = np.linspace(wad_ring_lm[1], wad_lm_melt[1], 21)
pressures = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    pressures[i] = eqm_T(T, [wad, lm], [1.0, 1.0])

ax[0].plot(pressures / 1.0e9, temperatures, color="black")

temperatures = np.linspace(1400.0, wad_ring_lm[1], 21)
pressures = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    pressures[i] = eqm_T(T, [wad, ring], [1.0, 1.0])

ax[0].plot(pressures / 1.0e9, temperatures, color="black")


temperatures = np.linspace(1400.0, wad_ring_lm[1], 21)
pressures = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    pressures[i] = eqm_T(T, [ring, lm], [1.0, 1.0])

ax[0].plot(pressures / 1.0e9, temperatures, color="black")


pressures = np.linspace(0.0e9, fo_wad_melt[0], 21)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = eqm(P, [fo, Mg2SiO4L])

ax[0].plot(pressures / 1.0e9, temperatures, color="black")


pressures = np.linspace(10.0e9, wad_lm_melt[0] + 2.0e9, 21)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = eqm(P, [ring, Mg2SiO4L])

ax[0].plot(pressures / 1.0e9, temperatures, color="black", linestyle=":")

pressures = np.linspace(fo_wad_melt[0], wad_lm_melt[0], 21)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = eqm(P, [wad, Mg2SiO4L])

ax[0].plot(pressures / 1.0e9, temperatures, color="black")


pressures = np.linspace(wad_lm_melt[0], 26.0e9, 21)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = eqm(P, [lm, Mg2SiO4L], [1.0, 1.0])

ax[0].plot(pressures / 1.0e9, temperatures, color="black")

ax[0].text(7.0, 1700.0 + 273.15, "fo")
ax[0].text(18.0, 1900.0 + 273.15, "wad")
ax[0].text(21.0, 1500.0 + 273.15, "ring")
ax[0].text(23.0, 2000.0 + 273.15, "bdg+per")
ax[0].text(7.0, 2500.0 + 273.15, "melt")

ax[0].set_ylim(1673, 2973)
ax[0].set_xlim(0, 26)

ax[0].set_xlabel("Pressure (GPa)")
ax[0].set_ylabel("Temperature (K)")

mins = [fo, wad, ring, lm, Mg2SiO4L]
pressures = np.linspace(1.0e5, 30.0e9, 101)
T = 2073
temperatures = pressures * 0.0 + T
for m in mins:
    rhos = m.evaluate(["density"], pressures, temperatures)[0]
    ax[1].plot(pressures / 1.0e9, rhos, label=f"{m.name}, {T} K".replace("_", " "))


"""
V_S = 1./lm.evaluate(['rho'], pressures, temperatures)[0]
V_L = 1./Mg2SiO4L.evaluate(['rho'], pressures, temperatures)[0]

from scipy.optimize import curve_fit

def dV(x, a, b, c):
    return a/np.log(b*x+c)


def dV(x, a, b, c):
    return a/((b*x + c)*np.log(b*x+c))

sol = curve_fit(dV, pressures, V_L - V_S, [-6.8e-06, -8.e-12,  8.e-01])[0]
print(sol)
"""

ax[1].set_xlabel("Pressure (GPa)")
ax[1].set_ylabel("Density (kg/m$^3$)")
ax[1].legend()

fig.tight_layout()
fig.savefig("output_figures/Mg2SiO4_melting_linearised.pdf")
plt.show()
