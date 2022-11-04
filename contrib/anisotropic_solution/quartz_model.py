from __future__ import absolute_import


import numpy as np

from burnman import Mineral, Solution
from burnman import AnisotropicMineral, AnisotropicSolution
from burnman.classes.solutionmodel import PolynomialSolution

from burnman.utils.chemistry import dictionarize_formula, formula_mass
from burnman.utils.anisotropy import contract_compliances

from utils.interaction_terms import transform_terms
from anisotropic_endmember_params import alpha1_params, alpha2_params, psi_func_mbr


class alpha_quartz(Mineral):
    def __init__(self):
        formula = "SiO2"
        formula = dictionarize_formula(formula)
        self.params = {
            "name": "alpha quartz",
            "formula": formula,
            "equation_of_state": "slb3",
            "F_0": -2313.86317911,
            "V_0": 2.2761615699999998e-05,
            "K_0": 74803223700.0,
            "Kprime_0": 7.01334832,
            "Debye_0": 1069.02139,
            "grueneisen_0": -0.0615759576,
            "q_0": 10.940017,
            "G_0": 44856170000.0,
            "Gprime_0": 0.95315,
            "eta_s_0": 2.36469,
            "n": sum(formula.values()),
            "molar_mass": formula_mass(formula),
        }

        self.property_modifiers = [
            ["einstein", {"Theta_0": 510.158911, "Cv_inf": 39.9366924855}],
            ["einstein", {"Theta_0": 801.41137578, "Cv_inf": -52.9869089}],
            ["einstein", {"Theta_0": 2109.4459429999997, "Cv_inf": 13.0502164145}],
        ]

        Mineral.__init__(self)


class beta_quartz(Mineral):
    def __init__(self):
        formula = "SiO2"
        formula = dictionarize_formula(formula)
        self.params = {
            "name": "beta quartz",
            "formula": formula,
            "equation_of_state": "slb3",
            "F_0": 0.0,
            "V_0": 2.37899663e-05,
            "K_0": 72234965900.0,
            "Kprime_0": 7.01334832,
            "Debye_0": 1076.2553699999999,
            "grueneisen_0": -0.237705826,
            "q_0": 10.940017,
            "G_0": 44856170000.0,
            "Gprime_0": 0.95315,
            "eta_s_0": 2.36469,
            "n": sum(formula.values()),
            "molar_mass": formula_mass(formula),
        }

        self.property_modifiers = [
            ["einstein_delta", {"Theta_0": 243.517385, "S_inf": 2.00297001}],
            ["einstein_delta", {"Theta_0": 873.841765, "S_inf": 2.19210207}],
            ["einstein", {"Theta_0": 510.158911, "Cv_inf": 39.9366924855}],
            ["einstein", {"Theta_0": 801.41137578, "Cv_inf": -52.9869089}],
            ["einstein", {"Theta_0": 2109.4459429999997, "Cv_inf": 13.0502164145}],
        ]

        Mineral.__init__(self)


alpha = alpha_quartz()
beta = beta_quartz()

# Let
# Q0 = (p0 + p1) - p2
# Q1 = (p0 + p2) - p1
# Q2 = (p1 + p2) - p0
p_10 = 1.07586176
c_10 = 606.4469673112811
p_01 = 1.0 - p_10
c_01 = -293.3945695485221
n_01 = 6

"""
delta_gibbs = alpha.gibbs - beta.gibbs

Q2 = Q * Q
Q2s = Q2 - 1.0
Q6s = Q2 * Q2 * Q2 - 1.0
Q10s = Q2 * Q2 * Q2 * Q2 * Q2- 1.0

Gxs = (
    (p_10 * Q2s[0] + p_01 * Q2s[1]) * delta_gibbs
    + p_10 * c_10 * Q6s[0]
    + p_01 * c_01 * Q10s[1]
)
"""

# Mbr terms
mbr_terms = transform_terms(2, [0, 1, 2], np.array([1.0 * p_10, -1.0 * p_10]))
mbr_terms.extend(transform_terms(2, [0, 2, 1], np.array([1.0 * p_01, -1.0 * p_01])))

# Constant terms
ESV_terms = transform_terms(6, [0, 1, 2], np.array([p_10 * c_10, 0.0, 0.0]))
ESV_terms.extend(transform_terms(6, [0, 2, 1], np.array([p_01 * c_01, 0.0, 0.0])))

a = np.sqrt(3) / 2
f = np.cbrt(alpha.params["V_0"] / a / (4.9137 * 4.9137 * 5.4047))
cell_parameters_alpha = np.array(
    [4.9137 * f, 4.9137 * f, 5.4047 * f, 90.0, 90.0, 120.0]
)

alpha1 = AnisotropicMineral(
    alpha,
    cell_parameters=cell_parameters_alpha,
    anisotropic_parameters=alpha1_params,
    psi_function=psi_func_mbr,
    orthotropic=True,
)

alpha2 = AnisotropicMineral(
    alpha,
    cell_parameters=cell_parameters_alpha,
    anisotropic_parameters=alpha2_params,
    psi_function=psi_func_mbr,
    orthotropic=True,
)


model = PolynomialSolution(
    endmembers=[[alpha1, ""], [alpha1, ""], [alpha2, ""]],
    ESV_interactions=ESV_terms,
    interaction_endmembers=[alpha, beta],
    endmember_coefficients_and_interactions=mbr_terms,
)

qtz = Solution(name="qtz", solution_model=model)


def excess_gibbs(Q, pressure, temperature):
    molar_fractions = np.zeros(3)

    Q2 = 1.0 - (Q[0] + Q[1])
    molar_fractions[0] = (Q[0] + Q[1]) / 2.0
    molar_fractions[1] = (Q[0] + Q2) / 2.0
    molar_fractions[2] = (Q[1] + Q2) / 2.0

    qtz.set_state(pressure, temperature)
    qtz.set_composition(molar_fractions)
    return qtz.gibbs - qtz.endmembers[0][0].gibbs


def equilibrium_Q(pressure, temperature):

    alpha.set_state(pressure, temperature)
    beta.set_state(pressure, temperature)

    delta_gibbs = alpha.gibbs - beta.gibbs

    Q = np.zeros(2)
    if delta_gibbs < 0.0:
        # Q_0 > 0, Q_1 = 0
        Q[0] = np.power(np.abs(2.0 * delta_gibbs / (6.0 * c_10)), 1.0 / (4.0))
    else:
        Q[1] = np.power(np.abs(2.0 * delta_gibbs / (n_01 * c_01)), 1.0 / (n_01 - 2.0))
    return Q


def equilibrate_m(m):
    def equilibrate(pressure, temperature):
        Q = np.zeros(3)
        Q[:2] = equilibrium_Q(pressure, temperature)
        Q[2] = 1.0 - Q[0] - Q[1]

        p = np.zeros(3)
        p[0] = (Q[0] + Q[1]) / 2.0
        p[1] = (Q[0] + Q[2]) / 2.0
        p[2] = (Q[1] + Q[2]) / 2.0

        m.set_composition(p)
        m.set_state(pressure, temperature)

    return equilibrate


qtz.equilibrate = equilibrate_m(qtz)


def psi_xs_func(lnV, Pthi, X, params):

    sum_X = np.sum(X)

    ac = params["a"] + params["c"] * (lnV - params["lnV_0"])
    Psi = np.einsum("ijpq, p, q->ij", ac, X, X) / sum_X
    dPsidf = np.einsum("ijpq, p, q->ij", params["c"], X, X) / sum_X

    dPsidPth = np.zeros((6, 6, len(X)))

    b = np.einsum("ijpq, pk, q->ijk", ac, np.eye(len(X)), X) / sum_X
    bT = np.einsum("ijpq, qk, p->ijk", ac, np.eye(len(X)), X) / sum_X
    dPsidX = -np.einsum("ij, k->ijk", Psi, np.ones_like(X) / sum_X) + b + bT
    return (Psi, dPsidf, dPsidPth, dPsidX)


e_T = 0.0011
e_d = -0.0001
eps_Q0 = (e_T - e_d) * np.diag([-1.0, -1.0, 2.0])
eps_Q1 = e_d * np.diag([-1.0, -1.0, 2.0])
aniso_mbr_terms = transform_terms(2, [0, 1, 2], eps_Q0)
aniso_mbr_terms.extend(transform_terms(2, [0, 2, 1], eps_Q1))
a = np.zeros((6, 6, 3, 3))
for term in aniso_mbr_terms:
    a[:, :, term[3], term[4]] = a[:, :, term[3], term[4]] + contract_compliances(
        np.einsum("ij, kl->ijkl", np.array(term[:3]), np.eye(3) / 3.0)
    )

params = {
    "a": a,
    "c": np.zeros((6, 6, 3, 3)),
    "lnV_0": np.log(1.0),
}

qtz_a = AnisotropicSolution(
    name="qtz",
    scalar_solution=qtz,
    anisotropic_parameters=params,
    psi_excess_function=psi_xs_func,
    dXdQ=np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]]).T,
    orthotropic=True,
    relaxed=True,
)


qtz_a.equilibrate = equilibrate_m(qtz_a)


def make_anisotropic_model(args):

    a_over_c_0 = args[0]
    p11, p12, p14, p33, p44 = args[1:6]
    e_T, e_d = args[6:8]

    q11, q12, q14, q33, q44 = args[8:13]

    p13 = (1.0 - (2.0 * p12 + 2.0 * p11 + p33)) / 4.0
    p66 = 2.0 * (p11 - p12)
    q13 = (0.0 - (2.0 * q12 + 2.0 * q11 + q33)) / 4.0
    q66 = 2.0 * (q11 - q12)

    a_0 = a_over_c_0
    c_0 = 1.0
    f = np.cbrt(alpha.params["V_0"] / (np.sqrt(3.0) / 2.0) / (a_0 * a_0 * c_0))
    cell_parameters_alpha = np.array([a_0 * f, a_0 * f, c_0 * f, 90.0, 90.0, 120.0])

    alpha1_params = {
        "a": np.array(
            [
                [p11, p12, p13, p14, 0.0, 0.0],
                [p12, p11, p13, -p14, 0.0, 0.0],
                [p13, p13, p33, 0.0, 0.0, 0.0],
                [p14, -p14, 0.0, p44, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, p44, 2.0 * p14],
                [0.0, 0.0, 0.0, 0.0, 2.0 * p14, p66],
            ]
        ),
        "b_1": np.array(
            [
                [q11, q12, q13, q14, 0.0, 0.0],
                [q12, q11, q13, -q14, 0.0, 0.0],
                [q13, q13, q33, 0.0, 0.0, 0.0],
                [q14, -q14, 0.0, q44, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, q44, 2.0 * q14],
                [0.0, 0.0, 0.0, 0.0, 2.0 * q14, q66],
            ]
        ),
        "c_1": np.ones((6, 6)),
        "b_2": np.zeros((6, 6)),
        "c_2": np.ones((6, 6)),
        "d": np.zeros((6, 6)),
    }

    alpha2_params = {
        "a": np.array(
            [
                [p11, p12, p13, -p14, 0.0, 0.0],
                [p12, p11, p13, p14, 0.0, 0.0],
                [p13, p13, p33, 0.0, 0.0, 0.0],
                [-p14, p14, 0.0, p44, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, p44, -2.0 * p14],
                [0.0, 0.0, 0.0, 0.0, -2.0 * p14, p66],
            ]
        ),
        "b_1": np.array(
            [
                [q11, q12, q13, -q14, 0.0, 0.0],
                [q12, q11, q13, q14, 0.0, 0.0],
                [q13, q13, q33, 0.0, 0.0, 0.0],
                [-q14, q14, 0.0, q44, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, q44, -2.0 * q14],
                [0.0, 0.0, 0.0, 0.0, -2.0 * q14, q66],
            ]
        ),
        "c_1": np.ones((6, 6)),
        "b_2": np.zeros((6, 6)),
        "c_2": np.ones((6, 6)),
        "d": np.zeros((6, 6)),
    }

    alpha1 = AnisotropicMineral(
        alpha,
        cell_parameters=cell_parameters_alpha,
        anisotropic_parameters=alpha1_params,
        psi_function=psi_func_mbr,
        orthotropic=True,
    )

    alpha2 = AnisotropicMineral(
        alpha,
        cell_parameters=cell_parameters_alpha,
        anisotropic_parameters=alpha2_params,
        psi_function=psi_func_mbr,
        orthotropic=True,
    )

    model = PolynomialSolution(
        endmembers=[[alpha1, ""], [alpha1, ""], [alpha2, ""]],
        ESV_interactions=ESV_terms,
        interaction_endmembers=[alpha, beta],
        endmember_coefficients_and_interactions=mbr_terms,
    )

    qtz = Solution(name="qtz", solution_model=model)
    qtz.equilibrate = equilibrate_m(qtz)

    eps_Q0 = (e_T - e_d) * np.diag([-1.0, -1.0, 2.0])
    eps_Q1 = e_d * np.diag([-1.0, -1.0, 2.0])
    aniso_mbr_terms = transform_terms(2, [0, 1, 2], eps_Q0)
    aniso_mbr_terms.extend(transform_terms(2, [0, 2, 1], eps_Q1))
    a = np.zeros((6, 6, 3, 3))
    for term in aniso_mbr_terms:
        a[:, :, term[3], term[4]] = a[:, :, term[3], term[4]] + contract_compliances(
            np.einsum("ij, kl->ijkl", np.array(term[:3]), np.eye(3) / 3.0)
        )

    params = {
        "a": a,
        "c": np.zeros((6, 6, 3, 3)),
        "lnV_0": np.log(1.0),
    }

    qtz_a = AnisotropicSolution(
        name="qtz",
        scalar_solution=qtz,
        anisotropic_parameters=params,
        psi_excess_function=psi_xs_func,
        dXdQ=np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]]).T,
        orthotropic=True,
        relaxed=True,
    )

    qtz_a.equilibrate = equilibrate_m(qtz_a)

    return qtz_a


def make_anisotropic_model_2(args):

    a_over_c_0 = args[0]
    p11, p12, p14, p33, p44 = args[1:6]
    e_T, e_d = args[6:8]

    q11, q12, q14, q33, q44 = args[8:13]
    r11, r12, r14, r33, r44 = args[13:18]
    s11, s12, s33, s44 = args[18:22]

    s14 = 0.0

    p13 = (1.0 - (2.0 * p12 + 2.0 * p11 + p33)) / 4.0
    p66 = 2.0 * (p11 - p12)
    q13 = (0.0 - (2.0 * q12 + 2.0 * q11 + q33)) / 4.0
    q66 = 2.0 * (q11 - q12)
    r13 = (0.0 - (2.0 * r12 + 2.0 * r11 + r33)) / 4.0
    r66 = 2.0 * (r11 - r12)
    s13 = (0.0 - (2.0 * s12 + 2.0 * s11 + s33)) / 4.0
    s66 = 2.0 * (s11 - s12)

    a_0 = a_over_c_0
    c_0 = 1.0
    f = np.cbrt(alpha.params["V_0"] / (np.sqrt(3.0) / 2.0) / (a_0 * a_0 * c_0))
    cell_parameters_alpha = np.array([a_0 * f, a_0 * f, c_0 * f, 90.0, 90.0, 120.0])

    alpha1_params = {
        "a": np.array(
            [
                [p11, p12, p13, p14, 0.0, 0.0],
                [p12, p11, p13, -p14, 0.0, 0.0],
                [p13, p13, p33, 0.0, 0.0, 0.0],
                [p14, -p14, 0.0, p44, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, p44, 2.0 * p14],
                [0.0, 0.0, 0.0, 0.0, 2.0 * p14, p66],
            ]
        ),
        "b_1": np.array(
            [
                [q11, q12, q13, q14, 0.0, 0.0],
                [q12, q11, q13, -q14, 0.0, 0.0],
                [q13, q13, q33, 0.0, 0.0, 0.0],
                [q14, -q14, 0.0, q44, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, q44, 2.0 * q14],
                [0.0, 0.0, 0.0, 0.0, 2.0 * q14, q66],
            ]
        ),
        "c_1": np.ones((6, 6)),
        "b_2": np.zeros((6, 6)),
        "c_2": np.ones((6, 6)),
        "d": np.zeros((6, 6)),
    }

    alpha2_params = {
        "a": np.array(
            [
                [p11, p12, p13, -p14, 0.0, 0.0],
                [p12, p11, p13, p14, 0.0, 0.0],
                [p13, p13, p33, 0.0, 0.0, 0.0],
                [-p14, p14, 0.0, p44, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, p44, -2.0 * p14],
                [0.0, 0.0, 0.0, 0.0, -2.0 * p14, p66],
            ]
        ),
        "b_1": np.array(
            [
                [q11, q12, q13, -q14, 0.0, 0.0],
                [q12, q11, q13, q14, 0.0, 0.0],
                [q13, q13, q33, 0.0, 0.0, 0.0],
                [-q14, q14, 0.0, q44, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, q44, -2.0 * q14],
                [0.0, 0.0, 0.0, 0.0, -2.0 * q14, q66],
            ]
        ),
        "c_1": np.ones((6, 6)),
        "b_2": np.zeros((6, 6)),
        "c_2": np.ones((6, 6)),
        "d": np.zeros((6, 6)),
    }

    alpha1 = AnisotropicMineral(
        alpha,
        cell_parameters=cell_parameters_alpha,
        anisotropic_parameters=alpha1_params,
        psi_function=psi_func_mbr,
        orthotropic=True,
    )

    alpha2 = AnisotropicMineral(
        alpha,
        cell_parameters=cell_parameters_alpha,
        anisotropic_parameters=alpha2_params,
        psi_function=psi_func_mbr,
        orthotropic=True,
    )

    model = PolynomialSolution(
        endmembers=[[alpha1, ""], [alpha1, ""], [alpha2, ""]],
        ESV_interactions=ESV_terms,
        interaction_endmembers=[alpha, beta],
        endmember_coefficients_and_interactions=mbr_terms,
    )

    qtz = Solution(name="qtz", solution_model=model)
    qtz.equilibrate = equilibrate_m(qtz)

    eps_Q0 = (e_T - e_d) * np.diag([-1.0, -1.0, 2.0])
    eps_Q1 = e_d * np.diag([-1.0, -1.0, 2.0])
    aniso_mbr_terms = transform_terms(2, [0, 1, 2], eps_Q0)
    aniso_mbr_terms.extend(transform_terms(2, [0, 2, 1], eps_Q1))
    a = np.zeros((6, 6, 3, 3))
    for term in aniso_mbr_terms:
        a[:, :, term[3], term[4]] = a[:, :, term[3], term[4]] + contract_compliances(
            np.einsum("ij, kl->ijkl", np.array(term[:3]), np.eye(3) / 3.0)
        )

    Psi_Q0 = np.array(
        [
            [r11, r12, r13, r14, 0.0, 0.0],
            [r12, r11, r13, -r14, 0.0, 0.0],
            [r13, r13, r33, 0.0, 0.0, 0.0],
            [r14, -r14, 0.0, r44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, r44, 2.0 * r14],
            [0.0, 0.0, 0.0, 0.0, 2.0 * r14, r66],
        ]
    )

    Psi_Q1 = np.array(
        [
            [s11, s12, s13, s14, 0.0, 0.0],
            [s12, s11, s13, -s14, 0.0, 0.0],
            [s13, s13, s33, 0.0, 0.0, 0.0],
            [s14, -s14, 0.0, s44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, s44, 2.0 * s14],
            [0.0, 0.0, 0.0, 0.0, 2.0 * s14, s66],
        ]
    )

    aniso_Psi_terms = transform_terms(2, [0, 1, 2], Psi_Q0)
    aniso_Psi_terms.extend(transform_terms(2, [0, 2, 1], Psi_Q1))
    c = np.zeros((6, 6, 3, 3))
    for term in aniso_Psi_terms:
        c[:, :, term[6], term[7]] = c[:, :, term[6], term[7]] + np.array(term[:6])

    params = {
        "a": a,
        "c": c,
        "lnV_0": np.log(alpha1.params["V_0"]),
    }

    qtz_a = AnisotropicSolution(
        name="qtz",
        scalar_solution=qtz,
        anisotropic_parameters=params,
        psi_excess_function=psi_xs_func,
        dXdQ=np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]]).T,
        orthotropic=True,
        relaxed=True,
    )

    qtz_a.equilibrate = equilibrate_m(qtz_a)

    return qtz_a


def make_array(total, s1, p11, p33, p44, p14):

    p13 = (total - 2.0 * s1 - p33) / 2.0
    p12 = s1 - p11 - p13

    # s1 = p11 + p12 + p13
    # 2p13 + p33 + 2s1 = total

    p66 = 2.0 * (p11 - p12)

    return np.array(
        [
            [p11, p12, p13, p14, 0.0, 0.0],
            [p12, p11, p13, -p14, 0.0, 0.0],
            [p13, p13, p33, 0.0, 0.0, 0.0],
            [p14, -p14, 0.0, p44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, p44, 2.0 * p14],
            [0.0, 0.0, 0.0, 0.0, 2.0 * p14, p66],
        ]
    )


def make_power_array(p11, p44, p14):

    return np.array(
        [
            [p11, p11, p11, p14, 0.0, 0.0],
            [p11, p11, p11, p14, 0.0, 0.0],
            [p11, p11, p11, 0.0, 0.0, 0.0],
            [p14, p14, 0.0, p44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, p44, p14],
            [0.0, 0.0, 0.0, 0.0, p14, p11],
        ]
    )


def make_anisotropic_model_3(args):
    """
    Psi_11ij delta_ij =
      a0
        + a1 * f
        + a2 * np.exp(d1 * f)
        + a3 * Pth
        + a4 * (Q1sqr - 1.)
        + a5 * (Q2sqr - 1.)
    )
    """

    # a0, a1, a2, a3, a4, a5, d
    ac = [
        0.3025364,
        -0.11589704,
        -0.70119436,
        0.00855154,
        -0.00358752,
        0.00098826,
        -0.63552351,
    ]

    a11, a33, a44, a14 = args[0:4]
    b11, b33, b44, b14 = args[4:8]
    c44, c14 = args[8:10]
    d11, d33, d44, d14 = args[10:14]
    e11, e33, e44 = args[14:17]
    f11, f33, f44 = args[17:20]

    a_0 = np.exp(ac[0]) / np.exp(1.0 - 2.0 * (ac[0]))
    c_0 = 1.0
    f = np.cbrt(alpha.params["V_0"] / (np.sqrt(3.0) / 2.0) / (a_0 * a_0 * c_0))
    cell_parameters_alpha = np.array([a_0 * f, a_0 * f, c_0 * f, 90.0, 90.0, 120.0])

    alpha1_params = {
        "a": make_array(1.0, ac[1], a11, a33, a44, a14),
        "b_1": make_array(0.0, ac[2], b11, b33, b44, b14),
        "c_1": make_power_array(ac[-1], c44, c14),
        "d": make_array(0.0, ac[3], d11, d33, d44, d14),
    }

    alpha2_params = {
        "a": make_array(1.0, ac[1], a11, a33, a44, -a14),
        "b_1": make_array(0.0, ac[2], b11, b33, b44, -b14),
        "c_1": make_power_array(ac[-1], c44, c14),
        "d": make_array(0.0, ac[3], d11, d33, d44, -d14),
    }

    alpha1 = AnisotropicMineral(
        alpha,
        cell_parameters=cell_parameters_alpha,
        anisotropic_parameters=alpha1_params,
        psi_function=psi_func_mbr,
        orthotropic=True,
    )

    alpha2 = AnisotropicMineral(
        alpha,
        cell_parameters=cell_parameters_alpha,
        anisotropic_parameters=alpha2_params,
        psi_function=psi_func_mbr,
        orthotropic=True,
    )

    model = PolynomialSolution(
        endmembers=[[alpha1, ""], [alpha1, ""], [alpha2, ""]],
        ESV_interactions=ESV_terms,
        interaction_endmembers=[alpha, beta],
        endmember_coefficients_and_interactions=mbr_terms,
    )

    qtz = Solution(name="qtz", solution_model=model)
    qtz.equilibrate = equilibrate_m(qtz)

    eps_Q0 = ac[4] * np.diag([1.0, 1.0, -2.0])
    eps_Q1 = ac[5] * np.diag([1.0, 1.0, -2.0])
    aniso_mbr_terms = transform_terms(2, [0, 1, 2], eps_Q0)
    aniso_mbr_terms.extend(transform_terms(2, [0, 2, 1], eps_Q1))
    a = np.zeros((6, 6, 3, 3))
    for term in aniso_mbr_terms:
        a[:, :, term[3], term[4]] = a[:, :, term[3], term[4]] + contract_compliances(
            np.einsum("ij, kl->ijkl", np.array(term[:3]), np.eye(3) / 3.0)
        )

    Psi_Q0 = make_array(0.0, 0.0, e11, e33, e44, 0.0)
    Psi_Q1 = make_array(0.0, 0.0, f11, f33, f44, 0.0)
    aniso_Psi_terms = transform_terms(2, [0, 1, 2], Psi_Q0)
    aniso_Psi_terms.extend(transform_terms(2, [0, 2, 1], Psi_Q1))
    c = np.zeros((6, 6, 3, 3))
    for term in aniso_Psi_terms:
        c[:, :, term[6], term[7]] = c[:, :, term[6], term[7]] + np.array(term[:6])

    c = np.zeros((6, 6, 3, 3))
    params = {
        "a": a,
        "c": c,
        "lnV_0": np.log(alpha1.params["V_0"]),
    }

    qtz_a = AnisotropicSolution(
        name="qtz",
        scalar_solution=qtz,
        anisotropic_parameters=params,
        psi_excess_function=psi_xs_func,
        dXdQ=np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]]).T,
        orthotropic=True,
        relaxed=True,
    )

    qtz_a.equilibrate = equilibrate_m(qtz_a)

    return qtz_a


def make_anisotropic_model_4(args):
    """
    Psi_11ij delta_ij =
      a0
        + a1 * f
        + a2 * np.exp(d1 * f)
        + a3 * Pth
        + a4 * (Q1sqr - 1.)
        + a5 * (Q2sqr - 1.)
    )
    """

    a11, a33, a44, a14 = args[0:4]
    b11, b33, b44, b14 = args[4:8]
    c44, c14 = args[8:10]
    d11, d33, d44, d14 = args[10:14]
    e11, e33, e44 = args[14:17]
    f11, f33, f44 = args[17:20]
    # a0, a1, a2, a3, a4, a5, d
    ac = args[20:27]
    es1, fs1 = args[27:29]

    a_0 = np.exp(ac[0]) / np.exp(1.0 - 2.0 * (ac[0]))
    c_0 = 1.0
    f = np.cbrt(alpha.params["V_0"] / (np.sqrt(3.0) / 2.0) / (a_0 * a_0 * c_0))
    cell_parameters_alpha = np.array([a_0 * f, a_0 * f, c_0 * f, 90.0, 90.0, 120.0])

    alpha1_params = {
        "a": make_array(1.0, ac[1], a11, a33, a44, a14),
        "b_1": make_array(0.0, ac[2], b11, b33, b44, b14),
        "c_1": make_power_array(ac[-1], c44, c14),
        "d": make_array(0.0, ac[3], d11, d33, d44, d14),
    }

    alpha2_params = {
        "a": make_array(1.0, ac[1], a11, a33, a44, -a14),
        "b_1": make_array(0.0, ac[2], b11, b33, b44, -b14),
        "c_1": make_power_array(ac[-1], c44, c14),
        "d": make_array(0.0, ac[3], d11, d33, d44, -d14),
    }

    alpha1 = AnisotropicMineral(
        alpha,
        cell_parameters=cell_parameters_alpha,
        anisotropic_parameters=alpha1_params,
        psi_function=psi_func_mbr,
        orthotropic=True,
    )

    alpha2 = AnisotropicMineral(
        alpha,
        cell_parameters=cell_parameters_alpha,
        anisotropic_parameters=alpha2_params,
        psi_function=psi_func_mbr,
        orthotropic=True,
    )

    model = PolynomialSolution(
        endmembers=[[alpha1, ""], [alpha1, ""], [alpha2, ""]],
        ESV_interactions=ESV_terms,
        interaction_endmembers=[alpha, beta],
        endmember_coefficients_and_interactions=mbr_terms,
    )

    qtz = Solution(name="qtz", solution_model=model)
    qtz.equilibrate = equilibrate_m(qtz)

    eps_Q0 = ac[4] * np.diag([1.0, 1.0, -2.0])
    eps_Q1 = ac[5] * np.diag([1.0, 1.0, -2.0])
    aniso_mbr_terms = transform_terms(2, [0, 1, 2], eps_Q0)
    aniso_mbr_terms.extend(transform_terms(2, [0, 2, 1], eps_Q1))
    a = np.zeros((6, 6, 3, 3))
    for term in aniso_mbr_terms:
        a[:, :, term[3], term[4]] = a[:, :, term[3], term[4]] + contract_compliances(
            np.einsum("ij, kl->ijkl", np.array(term[:3]), np.eye(3) / 3.0)
        )

    Psi_Q0 = make_array(0.0, es1, e11, e33, e44, 0.0)
    Psi_Q1 = make_array(0.0, fs1, f11, f33, f44, 0.0)
    aniso_Psi_terms = transform_terms(2, [0, 1, 2], Psi_Q0)
    aniso_Psi_terms.extend(transform_terms(2, [0, 2, 1], Psi_Q1))
    c = np.zeros((6, 6, 3, 3))
    for term in aniso_Psi_terms:
        c[:, :, term[6], term[7]] = c[:, :, term[6], term[7]] + np.array(term[:6])

    params = {
        "a": a,
        "c": c,
        "lnV_0": np.log(alpha1.params["V_0"]),
    }

    qtz_a = AnisotropicSolution(
        name="qtz",
        scalar_solution=qtz,
        anisotropic_parameters=params,
        psi_excess_function=psi_xs_func,
        dXdQ=np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]]).T,
        orthotropic=True,
        relaxed=True,
    )

    qtz_a.equilibrate = equilibrate_m(qtz_a)

    return qtz_a


def make_anisotropic_model_5(args):
    """
    Psi_11ij delta_ij =
      a0
        + a1 * f
        + a2 * (np.exp(d1 * f) - 1)
        + a3 * Pth
        + a4 * (Q1sqr - 1.)
        + a5 * (Q2sqr - 1.)
        + a6 * (Q1sqr - 1.) * f
        + a7 * (Q2sqr - 1.) * f
        )
    """

    a11, a33, a44, a14 = args[0:4]
    b11, b33, b44, b14 = args[4:8]
    c44, c14 = args[8:10]
    d11, d33, d44, d14 = args[10:14]
    e11, e33, e44 = args[14:17]
    f11, f33, f44 = args[17:21]
    # a0, a1, a2, a3, a4, a5, a6, a7, d
    ac = [
        3.02276161e-01,
        2.78273058e00,
        -8.86626017e00,
        -2.97357175e-03,
        -5.50793786e-04,
        6.79575848e-04,
        -2.12598726e-02,
        -1.49051977e-02,
        2.75266172e-01,
    ]

    a_0 = np.exp(ac[0]) / np.exp(1.0 - 2.0 * (ac[0]))
    c_0 = 1.0
    f = np.cbrt(alpha.params["V_0"] / (np.sqrt(3.0) / 2.0) / (a_0 * a_0 * c_0))
    cell_parameters_alpha = np.array([a_0 * f, a_0 * f, c_0 * f, 90.0, 90.0, 120.0])

    alpha1_params = {
        "a": make_array(1.0, ac[1], a11, a33, a44, a14),
        "b_1": make_array(0.0, ac[2], b11, b33, b44, b14),
        "c_1": make_power_array(ac[-1], c44, c14),
        "d": make_array(0.0, ac[3], d11, d33, d44, d14),
    }

    alpha2_params = {
        "a": make_array(1.0, ac[1], a11, a33, a44, -a14),
        "b_1": make_array(0.0, ac[2], b11, b33, b44, -b14),
        "c_1": make_power_array(ac[-1], c44, c14),
        "d": make_array(0.0, ac[3], d11, d33, d44, -d14),
    }

    alpha1 = AnisotropicMineral(
        alpha,
        cell_parameters=cell_parameters_alpha,
        anisotropic_parameters=alpha1_params,
        psi_function=psi_func_mbr,
        orthotropic=True,
    )

    alpha2 = AnisotropicMineral(
        alpha,
        cell_parameters=cell_parameters_alpha,
        anisotropic_parameters=alpha2_params,
        psi_function=psi_func_mbr,
        orthotropic=True,
    )

    model = PolynomialSolution(
        endmembers=[[alpha1, ""], [alpha1, ""], [alpha2, ""]],
        ESV_interactions=ESV_terms,
        interaction_endmembers=[alpha, beta],
        endmember_coefficients_and_interactions=mbr_terms,
    )

    qtz = Solution(name="qtz", solution_model=model)
    qtz.equilibrate = equilibrate_m(qtz)

    eps_Q0 = ac[4] * np.diag([1.0, 1.0, -2.0])
    eps_Q1 = ac[5] * np.diag([1.0, 1.0, -2.0])
    aniso_mbr_terms = transform_terms(2, [0, 1, 2], eps_Q0)
    aniso_mbr_terms.extend(transform_terms(2, [0, 2, 1], eps_Q1))
    a = np.zeros((6, 6, 3, 3))
    for term in aniso_mbr_terms:
        a[:, :, term[3], term[4]] = a[:, :, term[3], term[4]] + contract_compliances(
            np.einsum("ij, kl->ijkl", np.array(term[:3]), np.eye(3) / 3.0)
        )

    Psi_Q0 = make_array(0.0, ac[6], e11, e33, e44, 0.0)
    Psi_Q1 = make_array(0.0, ac[7], f11, f33, f44, 0.0)
    aniso_Psi_terms = transform_terms(2, [0, 1, 2], Psi_Q0)
    aniso_Psi_terms.extend(transform_terms(2, [0, 2, 1], Psi_Q1))
    c = np.zeros((6, 6, 3, 3))
    for term in aniso_Psi_terms:
        c[:, :, term[6], term[7]] = c[:, :, term[6], term[7]] + np.array(term[:6])

    params = {
        "a": a,
        "c": c,
        "lnV_0": np.log(alpha1.params["V_0"]),
    }

    qtz_a = AnisotropicSolution(
        name="qtz",
        scalar_solution=qtz,
        anisotropic_parameters=params,
        psi_excess_function=psi_xs_func,
        dXdQ=np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]]).T,
        orthotropic=True,
        relaxed=True,
    )

    qtz_a.equilibrate = equilibrate_m(qtz_a)

    return qtz_a
