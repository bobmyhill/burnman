import os.path
import sys

sys.path.insert(1, os.path.abspath("../.."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from burnman import Mineral, Solution
from burnman.classes.solutionmodel import IdealSolution
from burnman.minerals.Pitzer_Sterner_1994 import H2O_Pitzer_Sterner
from model_parameters import Mg2SiO4_params, Fe2SiO4_params


fo_liq = Mineral(Mg2SiO4_params)
fa_liq = Mineral(Fe2SiO4_params)

melt_modifiers = {
    "delta_E": 175553.0,
    "delta_S": 100.3,
    "delta_V": -3.277e-6,
    "a": 2.60339693e-06,
    "b": 2.64753089e-11,
    "c": 1.18703511e00,
}
fo_liq.property_modifiers = [["linlog", melt_modifiers]]
fa_liq.property_modifiers = [["linlog", melt_modifiers]]


class melt_all_liquid(Solution):
    def __init__(self, molar_fractions=None):
        self.name = "melt"
        self.solution_model = IdealSolution(
            endmembers=[
                [fo_liq, "[Mg]MgSiO4"],
                [fa_liq, "[Fe]FeSiO4"],
                [H2O_Pitzer_Sterner(), "[Hh]O"],
            ]
        )

        Solution.__init__(self, molar_fractions=molar_fractions)


# Mg/(Mg+Fe) = 0.8
# 2 wt% H20

melt = melt_all_liquid([0.8, 0.2, 0.0])
molar_mass_sol = melt.molar_mass
melt = melt_all_liquid([0.0, 0.0, 1.0])
molar_mass_H2O = melt.molar_mass

fig = plt.figure(figsize=(14, 6))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
Fig2_img = mpimg.imread("data/Sakamaki_et_al_2006_Fig_2.png")
ax[0].imshow(Fig2_img, extent=[0.0, 20.0, 0.0, 50.0], aspect="auto")

T = 2473.15
pressures = np.linspace(1.5e9, 22.0e9, 101)
temperatures = pressures * 0.0 + T

melt = melt_all_liquid([0.0, 0.0, 1.0])
volumes = melt.evaluate(["V"], pressures, temperatures)[0]

ax[0].plot(
    pressures / 1.0e9,
    volumes * 1.0e6,
    label=f"Pure H$_2$O ({T} K; Sterner and Pitzer, 1994)",
)
ax[0].plot(
    (pressures - 1.5e9) / 1.0e9,
    volumes * 1.0e6 - 2.5,
    label=f"Pure H$_2$O ({T} K; shifted)",
)


Fig3b_img = mpimg.imread("data/Sakamaki_et_al_2006_hydrous_umafic_melt_1873K.png")
ax[1].imshow(Fig3b_img, extent=[0.0, 20.0, 2500.0, 4000.0], aspect="auto")

for mass_H2O in [0.00, 0.02, 0.04, 0.06, 0.08]:
    mass_sol = 1.0 - mass_H2O

    mole_sol = mass_sol / molar_mass_sol
    mole_H2O = mass_H2O / molar_mass_H2O

    f_sol = mole_sol / (mole_sol + mole_H2O)
    f_H2O = mole_H2O / (mole_sol + mole_H2O)

    composition = [0.8 * f_sol, 0.2 * f_sol, f_H2O]
    print(f"{mass_H2O}, {composition}")
    melt = melt_all_liquid(composition)

    pressures = np.linspace(6.0e9, 20.0e9, 101)
    temperatures = pressures * 0.0 + 1873.0

    densities = melt.evaluate(["rho"], pressures, temperatures)[0]

    ax[1].plot(
        pressures / 1.0e9,
        densities,
        label=f"This study, {mass_H2O*100} wt% H$_2$O (P+S 1994)",
    )
ax[0].legend()
ax[1].legend()

ax[0].set_xlabel("Pressure (GPa)")
ax[0].set_ylabel("Volumes (cm$^3$/mol)")
ax[1].set_xlabel("Pressure (GPa)")
ax[1].set_ylabel("Densities (kg/m$^3$)")
fig.savefig("output_figures/Sakamaki_comparison.pdf")
plt.show()
