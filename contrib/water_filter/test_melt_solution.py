import os.path
import sys

sys.path.insert(1, os.path.abspath("../.."))

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from burnman import Mineral, Solution
from burnman.classes.solutionmodel import IdealSolution
from burnman.minerals.Pitzer_Sterner_1994 import H2O_Pitzer_Sterner
from model_parameters import Mg2SiO4_params, Fe2SiO4_params


class melt_half_solid(Solution):
    def __init__(self, molar_fractions=None):
        self.name = "melt"
        self.solution_model = IdealSolution(
            endmembers=[
                [Mineral(Mg2SiO4_params), "[Mg]MgSiO4"],
                [Mineral(Fe2SiO4_params), "[Fe]FeSiO4"],
                [H2O_Pitzer_Sterner(), "[Hh]O"],
            ],
        )

        Solution.__init__(self, molar_fractions=molar_fractions)


composition = [0.38627502932533253, 0.13620393497687355, 0.47752103569779397]
melt = melt_half_solid(composition)

melt.set_state(13.0e9, 2500.0)

print(melt.rho)


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


melt = melt_all_liquid(composition)

melt.set_state(13.0e9, 2200.0)

print(melt.rho)
print(melt.molar_mass)
print(melt.solution_model)
print(melt.endmembers[0][0]._property_modifiers["dGdP"])
