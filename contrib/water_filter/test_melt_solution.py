from burnman import Mineral, Solution
from burnman.classes.solutionmodel import IdealSolution
from burnman.minerals.Pitzer_Sterner_1994 import H2O_Pitzer_Sterner
from model_parameters import Mg2SiO4_params, Fe2SiO4_params
import numpy as np
from model_parameters import R, ol, wad, ring, lm, melt
from model_parameters import olwad, wadring, ringlm
from model_parameters import liq_sp
from scipy.special import expi


def _li(x):
    return expi(np.log(x))


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

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def V_quick(pressure):
    """
    This is the same as the linlog function
    f = params["b"] * pressure + params["c"]
    dGdP = params["delta_V"] + params["a"] / np.log(f)

    G = (
        params["delta_E"]
        - (temperature) * params["delta_S"]
        + (pressure) * params["delta_V"]
        + params["a"] / params["b"] * (_li(f) - _li(params["c"]))
    )
    """
    f = melt_modifiers["b"] * pressure + melt_modifiers["c"]
    return melt_modifiers["delta_V"] + melt_modifiers["a"] / np.log(f)


def V_quicker(pressure, a, b, c):
    """
    A simpler version of the linlog function
    """
    return a / (pressure + b) + c


def G_quicker(pressure, a, b, c, G0):
    """
    A simpler version of the linlog function
    """
    return a * np.log(pressure / b + 1.0) + c * pressure + G0


pressures = np.linspace(1.0e9, 13.0e9, 101)
"""
popt, pcov = curve_fit(
    V_quicker,
    pressures,
    V_quick(pressures),
    [1.00250593e05, 7.13877676e09, -2.14329378e-06],
)
print(popt)

pressures = np.linspace(1.0e5, 13.0e9, 101)
plt.plot(pressures, V_quick(pressures))
plt.plot(pressures, V_quicker(pressures, *popt), linestyle=":")
plt.show()

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
"""

from model_parameters import melt


def li_f(P):
    """
    Computes the melting temperature of a solid at a given pressure
    This solid may be metastable.
    """
    f = melt["b"] * P + melt["c"]
    return _li(f)


def dli_fdP(P):
    f = melt["b"] * P + melt["c"]
    return melt["c"]/np.log(f)

def li_f_cheap(pressure, a, b, c, G0):
    """
    Computes the melting temperature of a solid at a given pressure
    This solid may be metastable.
    """
    a, b, c = [4.57e10, 7.13e9, -0.517]
    return a * np.log(pressure / b + 1.0) + c * pressure + G0


popt, pcov = curve_fit(
    V_quicker,
    pressures,
    li_f(pressures),
    [4.57e10, 7.13e9, -0.517],
)
print(popt)
pressures = np.linspace(1.0e5, 25.0e9, 101)
plt.plot(pressures, dli_fdP(pressures))
plt.plot(pressures, V_quicker(pressures, *popt), linestyle=":")
plt.show()