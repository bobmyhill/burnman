from sympy.solvers import solve
from sympy import Symbol, simplify, log
import numpy as np

from MSH_endmembers import *
R = 8.31446

def gibbs(mineral, P, T):
    mineral.set_state(P, T)
    return mineral.gibbs

solve_melting_equations = False
if solve_melting_equations:
    X_H2O = Symbol('X_H2O')
    X_MgSiO3 = Symbol('X_MgSiO3')
    X_Mg2SiO4 = Symbol('X_Mg2SiO4')

    x_fo = Symbol('x_fo')
    x_en = Symbol('x_en')
    x_L = Symbol('x_L')
    p_H2OL = Symbol('p_H2OL')

    # Approximate alpha(MgSiO3_en) = 1 (i.e. pure enstatite)
    # y = gamma_H2OL * np.exp(-(G_H2MgSiO4fo(P, T) - G_MgSiO3en(P, T) - G_H2OL(P, T)) / (R*T))
    y = Symbol('y')

    # Approximate gamma(Mg2SiO4_fo) = 1
    # z = gamma_Mg2SiO4L * np.exp(-(G_Mg2SiO4fo(P, T) - G_Mg2SiO4L(P, T)) / (R*T))
    z = Symbol('z')

    p_H2MgSiO4fo = Symbol('p_H2MgSiO4fo')

    # Proportions sum to 1
    p_Mg2SiO4L = 1 - p_H2OL
    p_Mg2SiO4fo = 1 - p_H2MgSiO4fo

    # Constraints
    bulk_constraint_1 = (x_fo * p_H2MgSiO4fo) + (x_L * p_H2OL) - X_H2O
    bulk_constraint_2 = x_en + (x_fo * p_H2MgSiO4fo) - X_MgSiO3
    bulk_constraint_3 = (x_fo * p_Mg2SiO4fo) + (x_L * p_Mg2SiO4L) - X_Mg2SiO4

    equilibrium_constraint_1 = p_H2OL * y - p_H2MgSiO4fo
    equilibrium_constraint_2 = p_Mg2SiO4L * z - p_Mg2SiO4fo

    # The solve needs to be a square matrix, even if there is a non-trivial nullspace
    equations = (bulk_constraint_1,
                 bulk_constraint_2,
                 bulk_constraint_3,
                 equilibrium_constraint_1,
                 equilibrium_constraint_2)
    eqs = solve(equations, (x_fo, x_en, x_L, p_H2OL, p_H2MgSiO4fo), dict=True)[0]
    print(eqs)


def compositions(P, T, X_Mg2SiO4, X_H2O, X_MgSiO3, gamma_H2OL, gamma_Mg2SiO4L,
                 fo_polymorph, H2MgSiO4_polymorph, MgSiO3_polymorph):

    # Approximate alpha(MgSiO3_en) = 1 (i.e. pure enstatite)
    y = gamma_H2OL * np.exp(-(gibbs(H2MgSiO4_polymorph, P, T) - gibbs(MgSiO3_polymorph, P, T) - gibbs(H2OL, P, T)) / (R*T))

    # Approximate gamma(Mg2SiO4_fo) = 1
    z = gamma_Mg2SiO4L * np.exp(-(gibbs(fo_polymorph, P, T) - gibbs(Mg2SiO4L, P, T)) / (R*T))

    # Copied directly from the sympy output
    x_fo = -(X_H2O*y - X_H2O + X_Mg2SiO4*z - X_Mg2SiO4)/((y - 1)*(z - 1))
    x_en = -(X_H2O*y**2 - X_H2O*y + X_Mg2SiO4*y*z - X_Mg2SiO4*y - X_MgSiO3*y**2 + X_MgSiO3*y*z + X_MgSiO3*y - X_MgSiO3*z)/((y - 1)*(y - z))
    x_L = (X_H2O*y*z - X_H2O*z + X_Mg2SiO4*y*z - X_Mg2SiO4*y)/((y - 1)*(z - 1))
    p_H2OL = -(z - 1)/(y - z)
    p_H2MgSiO4fo = -y*(z - 1)/(y - z)
    return (x_fo, x_en, x_L, p_H2OL, p_H2MgSiO4fo)


print(compositions(P=13.e9,
                   T=2500.,
                   X_Mg2SiO4=1.,
                   X_H2O=1.,
                   X_MgSiO3=1.,
                   gamma_H2OL=1.,
                   gamma_Mg2SiO4L=1.,
                   fo_polymorph=fo,
                   H2MgSiO4_polymorph=H2MgSiO4fo,
                   MgSiO3_polymorph=hen))
