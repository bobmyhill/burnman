from sympy.solvers import solve
from sympy import Symbol, simplify, log
import numpy as np

import matplotlib.pyplot as plt
from MSH_endmembers import *
from burnman.minerals import HGP_2018_ds633

hen = HGP_2018_ds633.hen()
R = 8.31446


def gibbs(mineral, P, T):
    mineral.set_state(P, T)
    return mineral.gibbs


solve_melting_equations = False
if solve_melting_equations:
    X_H2O = Symbol("X_H2O")
    X_MgSiO3 = Symbol("X_MgSiO3")
    X_Mg2SiO4 = Symbol("X_Mg2SiO4")

    x_fo0 = Symbol("x_fo0")
    x_fo1 = Symbol("x_fo1")
    x_en = Symbol("x_en")
    x_L = Symbol("x_L")
    p_H2OL = Symbol("p_H2OL")

    # Approximate alpha(MgSiO3_en) = 1 (i.e. pure enstatite)
    # y = gamma_H2OL * np.exp(-(G_H2MgSiO4fo(P, T) - G_MgSiO3en(P, T) - G_H2OL(P, T)) / (R*T))
    y = Symbol("y")

    # Approximate gamma(Mg2SiO4_fo) = 1
    # z = gamma_Mg2SiO4L * np.exp(-(G_Mg2SiO4fo(P, T) - G_Mg2SiO4L(P, T)) / (R*T))
    z = Symbol("z")

    f = Symbol("f")
    p_H2MgSiO4fo0 = Symbol("p_H2MgSiO4fo0")
    p_H2MgSiO4fo1 = Symbol("p_H2MgSiO4fo1")

    # Proportions sum to 1
    p_Mg2SiO4L = 1 - p_H2OL
    p_Mg2SiO4fo0 = 1 - p_H2MgSiO4fo0
    p_Mg2SiO4fo1 = 1 - p_H2MgSiO4fo1

    # Constraints
    bulk_constraint_1 = (
        (x_fo0 * p_H2MgSiO4fo0) + (x_fo1 * p_H2MgSiO4fo1) + (x_L * p_H2OL) - X_H2O
    )
    bulk_constraint_2 = (
        x_en + (x_fo0 * p_H2MgSiO4fo0) + (x_fo1 * p_H2MgSiO4fo1) - X_MgSiO3
    )
    bulk_constraint_3 = (
        (x_fo0 * p_Mg2SiO4fo0) + (x_fo1 * p_Mg2SiO4fo1) + (x_L * p_Mg2SiO4L) - X_Mg2SiO4
    )

    transition_constraint = x_fo1 - f * (x_fo0 + x_fo1)

    # The solve needs to be a square matrix, even if there is a non-trivial nullspace
    equations = (
        bulk_constraint_1,
        bulk_constraint_2,
        bulk_constraint_3,
        transition_constraint,
    )

    eqs = solve(equations, (x_fo0, x_fo1, x_en, x_L), dict=True)[0]
    print(eqs)
    exit()

    # Constraints
    bulk_constraint_1 = (x_fo * p_H2MgSiO4fo) + (x_L * p_H2OL) - X_H2O
    bulk_constraint_2 = x_en + (x_fo * p_H2MgSiO4fo) - X_MgSiO3
    bulk_constraint_3 = (x_fo * p_Mg2SiO4fo) + (x_L * p_Mg2SiO4L) - X_Mg2SiO4

    equilibrium_constraint_1 = p_H2OL * y - p_H2MgSiO4fo
    equilibrium_constraint_2 = p_Mg2SiO4L * z - p_Mg2SiO4fo

    # The solve needs to be a square matrix, even if there is a non-trivial nullspace
    equations = (
        bulk_constraint_1,
        bulk_constraint_2,
        bulk_constraint_3,
        equilibrium_constraint_1,
        equilibrium_constraint_2,
    )
    eqs = solve(equations, (x_fo, x_en, x_L, p_H2OL, p_H2MgSiO4fo), dict=True)[0]
    print(eqs)


def compositions(
    P, T, X_Mg2SiO4, X_H2O, X_MgSiO3, fo_polymorph, H2MgSiO4_polymorph, MgSiO3_polymorph
):
    gamma_Mg2SiO4L = 1.0
    gamma_H2OL = 1.0

    # Approximate alpha(MgSiO3_en) = 1 (i.e. pure enstatite)
    y = gamma_H2OL * np.exp(
        -(
            gibbs(H2MgSiO4_polymorph, P, T)
            - gibbs(MgSiO3_polymorph, P, T)
            - gibbs(H2OL, P, T)
        )
        / (R * T)
    )

    # Approximate gamma(Mg2SiO4_fo) = 1
    z = gamma_Mg2SiO4L * np.exp(
        -(gibbs(fo_polymorph, P, T) - gibbs(Mg2SiO4L, P, T)) / (R * T)
    )

    # Copied directly from the sympy output
    x_fo = -(X_H2O * y - X_H2O + X_Mg2SiO4 * z - X_Mg2SiO4) / ((y - 1) * (z - 1))
    x_en = -(
        X_H2O * y**2
        - X_H2O * y
        + X_Mg2SiO4 * y * z
        - X_Mg2SiO4 * y
        - X_MgSiO3 * y**2
        + X_MgSiO3 * y * z
        + X_MgSiO3 * y
        - X_MgSiO3 * z
    ) / ((y - 1) * (y - z))
    x_L = (X_H2O * y * z - X_H2O * z + X_Mg2SiO4 * y * z - X_Mg2SiO4 * y) / (
        (y - 1) * (z - 1)
    )
    p_H2OL = -(z - 1) / (y - z)
    p_H2MgSiO4fo = -y * (z - 1) / (y - z)
    return (x_fo, x_en, x_L, p_H2OL, p_H2MgSiO4fo)


temperatures = np.linspace(1000.0, 2570.0, 101)
a_H2Os = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    x_fo, x_en, x_L, p_H2OL, p_H2MgSiO4fo = compositions(
        P=13.0e9,
        T=T,
        X_Mg2SiO4=1.0,
        X_H2O=0.2,
        X_MgSiO3=1.0,
        fo_polymorph=fo,
        H2MgSiO4_polymorph=H2MgSiO4fo,
        MgSiO3_polymorph=hen,
    )
