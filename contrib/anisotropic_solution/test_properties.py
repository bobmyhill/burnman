import numpy as np
from copy import deepcopy
from scipy.linalg import logm
from scipy.optimize import minimize
from burnman.utils.anisotropy import (
    voigt_notation_to_compliance_tensor,
    contract_stresses,
    contract_stiffnesses,
)

from quartz_model import make_anisotropic_model_3

args = [
    0.66483636,
    0.82201773,
    1.7035123,
    -0.16841388,
    -0.09017583,
    0.03277347,
    0.11496859,
    0.18327479,
    0.10710242,
    0.13515666,
    0.1399706,
    -0.37319184,
    -1.17929474,
    0.43856306,
    0.21361123,
    -0.01124113,
    0.21850874,
    0.03008858,
    -0.04118708,
    0.01021032,
]

qtz_a = make_anisotropic_model_3(args)


def test_dPSmudQ():
    P_0 = 1.0e5
    T_0 = 800.0
    qtz_a.equilibrate(P_0, T_0)

    dPdQ_0 = qtz_a.dPdQ
    dSdQ_0 = qtz_a.dSdQ
    dmudQ_0 = qtz_a.d2FdQdQ_fixed_volume

    dx = 0.0001
    V = qtz_a.V

    dPdX = np.zeros(3)
    dSdX = np.zeros(3)
    dmudX = np.zeros((3, 3))

    X_0 = qtz_a.molar_fractions

    for i, v in enumerate(np.eye(3)):

        X = X_0 - v * dx / 2.0
        qtz_a.set_composition(X / np.sum(X))
        qtz_a.set_state_with_volume(V / np.sum(X), T_0)

        P_0 = qtz_a.pressure
        S_0 = qtz_a.S
        mu_0 = qtz_a.partial_gibbs

        X = X_0 + v * dx / 2.0
        qtz_a.set_composition(X / np.sum(X))
        qtz_a.set_state_with_volume(V / np.sum(X), T_0)

        P_1 = qtz_a.pressure
        S_1 = qtz_a.S
        mu_1 = qtz_a.partial_gibbs

        dPdX[i] = (P_1 - P_0) / dx
        dSdX[i] = (S_1 - S_0) / dx
        dmudX[i] = (mu_1 - mu_0) / dx

    print(dPdX.dot(qtz_a.dXdQ))
    print(dPdQ_0)
    print()
    print(dSdX.dot(qtz_a.dXdQ))
    print(dSdQ_0)
    print()
    print(qtz_a.dXdQ.T.dot(dmudX.dot(qtz_a.dXdQ)))
    print(dmudQ_0)


print("dPSmudQ")
test_dPSmudQ()


def test_Psi_mbr():
    P_0 = 1.0e5
    T_0 = 900.0

    qtz_a.set_composition([1.0, 0.0, 0.0])
    qtz_a.set_state(P_0, T_0)

    a = qtz_a.scalar_solution.endmembers[0][0]
    print(np.max((a.Psi - qtz_a.Psi).flatten()))


print("Psi_mbr")
test_Psi_mbr()


def test_state():
    P_0 = 1.0e5
    T_0 = 900.0

    qtz_a.equilibrate(P_0, T_0)
    qtz_a.set_composition([1.0, 0.0, 0.0])

    a = qtz_a.scalar_solution.endmembers[0][0]
    print(np.max(np.abs((a.Psi - qtz_a.Psi).flatten())))
    print(np.max(np.abs((a.alpha - qtz_a.scalar_solution.alpha).flatten())))
    print(
        np.max(np.abs((a.dPsidP_Voigt - qtz_a.dPsidP_fixed_T_Voigt).flatten())) * 1.0e9
    )
    print(np.max(np.abs((a.dPsidT_Voigt - qtz_a.dPsidT_fixed_P_Voigt).flatten())))


print("state")
test_state()


def test_dPsidPT1a():
    P_0 = 1.0e5
    T_0 = 800.0

    qtz_a.set_composition([1.0, 0.0, 0.0])
    qtz_a.set_state(P_0, T_0)
    dPsidP_0 = voigt_notation_to_compliance_tensor(qtz_a.dPsidP_fixed_T_Voigt)
    dPsidT_0 = voigt_notation_to_compliance_tensor(qtz_a.dPsidT_fixed_P_Voigt)

    dP = 1.0e5
    dT = 0.1

    qtz_a.set_state(P_0 - dP / 2.0, T_0)
    Psi_0 = qtz_a.Psi
    qtz_a.set_state(P_0 + dP / 2.0, T_0)
    Psi_1 = qtz_a.Psi
    dPsidP = (Psi_1 - Psi_0) / dP

    qtz_a.set_state(P_0, T_0 - dT / 2.0)
    Psi_0 = qtz_a.Psi
    qtz_a.set_state(P_0, T_0 + dT / 2.0)
    Psi_1 = qtz_a.Psi
    dPsidT = (Psi_1 - Psi_0) / dT

    print(np.max(np.abs((dPsidP - dPsidP_0).flatten())) * 1.0e9)
    print(np.max(np.abs((dPsidT - dPsidT_0).flatten())))


print("dPsidPT1a")
test_dPsidPT1a()


def test_dPsidPT2a():
    P_0 = 1.0e5
    T_0 = 800.0

    qtz_a.equilibrate(P_0, T_0)
    dPsidP_0 = voigt_notation_to_compliance_tensor(qtz_a.dPsidP_fixed_T_Voigt)
    dPsidT_0 = voigt_notation_to_compliance_tensor(qtz_a.dPsidT_fixed_P_Voigt)

    dP = 1.0e5
    dT = 0.1

    qtz_a.set_state(P_0 - dP / 2.0, T_0)
    Psi_0 = qtz_a.Psi
    qtz_a.set_state(P_0 + dP / 2.0, T_0)
    Psi_1 = qtz_a.Psi
    dPsidP = (Psi_1 - Psi_0) / dP

    qtz_a.set_state(P_0, T_0 - dT / 2.0)
    Psi_0 = qtz_a.Psi
    qtz_a.set_state(P_0, T_0 + dT / 2.0)
    Psi_1 = qtz_a.Psi
    dPsidT = (Psi_1 - Psi_0) / dT

    print(np.max(np.abs((dPsidP - dPsidP_0).flatten()) * 1.0e9))
    print(np.max(np.abs((dPsidT - dPsidT_0).flatten())))


print("dPsidPT2a")
test_dPsidPT2a()


def test_dPsidPT1b():
    P_0 = 1.0e5
    T_0 = 900.0

    qtz_a.set_composition([1.0, 0.0, 0.0])
    qtz_a.set_state(P_0, T_0)
    dPsidP_0 = voigt_notation_to_compliance_tensor(qtz_a.dPsidP_fixed_T_Voigt)
    dPsidT_0 = voigt_notation_to_compliance_tensor(qtz_a.dPsidT_fixed_P_Voigt)

    dP = 1.0e5
    dT = 0.1

    qtz_a.set_state(P_0 - dP / 2.0, T_0)
    Psi_0 = qtz_a.Psi
    qtz_a.set_state(P_0 + dP / 2.0, T_0)
    Psi_1 = qtz_a.Psi
    dPsidP = (Psi_1 - Psi_0) / dP

    qtz_a.set_state(P_0, T_0 - dT / 2.0)
    Psi_0 = qtz_a.Psi
    qtz_a.set_state(P_0, T_0 + dT / 2.0)
    Psi_1 = qtz_a.Psi
    dPsidT = (Psi_1 - Psi_0) / dT

    print(np.max(np.abs((dPsidP - dPsidP_0).flatten())) * 1.0e9)
    print(np.max(np.abs((dPsidT - dPsidT_0).flatten())))


print("dPsidPT1b")
test_dPsidPT1b()


def test_dPsidPT2b():
    P_0 = 1.0e5
    T_0 = 900.0

    qtz_a.equilibrate(P_0, T_0)
    dPsidP_0 = voigt_notation_to_compliance_tensor(qtz_a.dPsidP_fixed_T_Voigt)
    dPsidT_0 = voigt_notation_to_compliance_tensor(qtz_a.dPsidT_fixed_P_Voigt)

    dP = 1.0e5
    dT = 0.1

    qtz_a.set_state(P_0 - dP / 2.0, T_0)
    Psi_0 = qtz_a.Psi
    qtz_a.set_state(P_0 + dP / 2.0, T_0)
    Psi_1 = qtz_a.Psi
    dPsidP = (Psi_1 - Psi_0) / dP

    qtz_a.set_state(P_0, T_0 - dT / 2.0)
    Psi_0 = qtz_a.Psi
    qtz_a.set_state(P_0, T_0 + dT / 2.0)
    Psi_1 = qtz_a.Psi
    dPsidT = (Psi_1 - Psi_0) / dT

    print(np.max(np.abs((dPsidP - dPsidP_0).flatten()) * 1.0e9))
    print(np.max(np.abs((dPsidT - dPsidT_0).flatten())))


print("dPsidPT2b")
test_dPsidPT2b()


def test_depsdQT():
    P_0 = 1.0e5
    T_0 = 800.0
    qtz_a.equilibrate(P_0, T_0)

    X_0 = qtz_a.molar_fractions
    eps_0 = logm(qtz_a.deformation_gradient_tensor)
    V_0 = qtz_a.V

    depsdQ_0 = qtz_a.depsdQ
    depsdT_0 = qtz_a.depsdT

    def depsdQordT(dQ, dT):
        X_1 = X_0 + np.array([-dQ[0] / 2.0 - dQ[1] / 2.0, +dQ[0] / 2.0, dQ[1] / 2.0])
        qtz_a.set_composition(X_1)
        qtz_a.set_state_with_volume(V_0, T_0 - dT / 2.0)
        eps_1 = logm(qtz_a.deformation_gradient_tensor)

        X_2 = X_0 + np.array([dQ[0] / 2.0 + dQ[1] / 2.0, -dQ[0] / 2.0, -dQ[1] / 2.0])
        qtz_a.set_composition(X_2)
        qtz_a.set_state_with_volume(V_0, T_0 + dT / 2.0)
        eps_2 = logm(qtz_a.deformation_gradient_tensor)
        return eps_2 - eps_1

    deps = 1.0e-1
    dT = 0.1
    i, v = [1, np.eye(2)[1]]
    print(
        np.max(
            np.abs(((depsdQordT(v * deps, 0.0) / deps) - depsdQ_0[:, :, i]).flatten())
        )
    )
    print(np.max(np.abs(((depsdQordT([0.0, 0.0], dT) / dT) - depsdT_0).flatten())))


print("depsdQT")
test_depsdQT()


def helmholtz_fixed_Q(dQ, deps, dT, V_0, T_0, X_0, eps_0, helmholtz_0):
    try:
        V_1 = V_0 * (np.trace(deps) + 1.0)
        T_1 = T_0 + dT

        dQ2 = -np.sum(dQ)
        dP = np.array([(dQ[0] + dQ[1]) / 2.0, (dQ[0] + dQ2) / 2.0, (dQ[1] + dQ2) / 2.0])
        X_1 = X_0 + dP
        qtz_a.set_composition(X_1)
        qtz_a.set_state_with_volume(V_1, T_1)

        eps_1 = logm(qtz_a.deformation_gradient_tensor)
        eps_prime = eps_0 + deps - eps_1
        Fel = (
            0.5
            * qtz_a.V
            * np.einsum(
                "ij, ijkl, kl->",
                eps_prime,
                qtz_a.full_isothermal_stiffness_tensor_unrelaxed,
                eps_prime,
            )
        )
        dF = qtz_a.helmholtz + Fel - helmholtz_0
        return dF
    except ValueError:
        return 1.0e10


def helmholtz_fixed_Q1(args, deps, dT, V_0, T_0, X_0, eps_0, helmholtz_0):
    return helmholtz_fixed_Q(
        np.array([0.0, args[0]]), deps, dT, V_0, T_0, X_0, eps_0, helmholtz_0
    )


def test_d2FdQdZQ():
    # Fixed strain and/or temperature
    P_0 = 1.0e5
    T_0 = 800.0

    qtz_a.equilibrate(P_0, T_0)
    qtz_a.set_composition([1.0, 0.0, 0.0])
    X_0 = qtz_a.molar_fractions
    eps_0 = logm(qtz_a.deformation_gradient_tensor)
    helmholtz_0 = qtz_a.helmholtz
    V_0 = qtz_a.V
    d2FdQdZ_0 = qtz_a.d2FdQdZ
    d2FdQdQ_0 = qtz_a.d2FdQdQ_fixed_strain

    dQ = 1.0e-5
    deps_0 = np.zeros((3, 3))
    deps = 1.0e-5
    dT = 1.0

    F_T1Q1 = helmholtz_fixed_Q(
        [0.0, dQ / 2.0], deps_0, dT / 2.0, V_0, T_0, X_0, eps_0, helmholtz_0
    )
    F_T1Q0 = helmholtz_fixed_Q(
        [0.0, -dQ / 2.0], deps_0, dT / 2.0, V_0, T_0, X_0, eps_0, helmholtz_0
    )
    F_T0Q1 = helmholtz_fixed_Q(
        [0.0, dQ / 2.0], deps_0, -dT / 2.0, V_0, T_0, X_0, eps_0, helmholtz_0
    )
    F_T0Q0 = helmholtz_fixed_Q(
        [0.0, -dQ / 2.0], deps_0, -dT / 2.0, V_0, T_0, X_0, eps_0, helmholtz_0
    )

    d2FdQdeps = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            de0 = deepcopy(deps_0)
            de0[i, j] = de0[i, j] - deps / 2.0

            de1 = deepcopy(deps_0)
            de1[i, j] = de1[i, j] + deps / 2.0

            F_e1Q1 = helmholtz_fixed_Q(
                [0.0, dQ / 2.0], de1, 0.0, V_0, T_0, X_0, eps_0, helmholtz_0
            )
            F_e1Q0 = helmholtz_fixed_Q(
                [0.0, -dQ / 2.0], de1, 0.0, V_0, T_0, X_0, eps_0, helmholtz_0
            )
            F_e0Q1 = helmholtz_fixed_Q(
                [0.0, dQ / 2.0], de0, 0.0, V_0, T_0, X_0, eps_0, helmholtz_0
            )
            F_e0Q0 = helmholtz_fixed_Q(
                [0.0, -dQ / 2.0], de0, 0.0, V_0, T_0, X_0, eps_0, helmholtz_0
            )

            d2FdQdeps[i, j] = ((F_e1Q1 - F_e1Q0) - (F_e0Q1 - F_e0Q0)) / (dQ * deps)

    d2FdQdZ = np.zeros(7)
    d2FdQdZ[:-1] = contract_stresses(d2FdQdeps)
    d2FdQdZ[-1] = ((F_T1Q1 - F_T1Q0) - (F_T0Q1 - F_T0Q0)) / (dT * dQ)

    # Constant strain components and/or temperature
    print(d2FdQdZ)
    print(d2FdQdZ_0[1, :])

    dQ = 1.0e-4
    deps_0 = np.zeros((3, 3))
    F = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            dQ0 = (i - 1.0) * dQ
            dQ1 = (j - 1.0) * dQ
            F[i, j] = helmholtz_fixed_Q(
                [dQ0, dQ1], deps_0, 0.0, V_0, T_0, X_0, eps_0, helmholtz_0
            )

    dFdQ0 = np.gradient(F, axis=0, edge_order=2)
    dFdQ1 = np.gradient(F, axis=1, edge_order=2)

    d2FdQdQ = np.array(
        [
            [
                np.gradient(dFdQ0, axis=0, edge_order=2)[1, 1],
                np.gradient(dFdQ0, axis=1, edge_order=2)[1, 1],
            ],
            [
                np.gradient(dFdQ1, axis=0, edge_order=2)[1, 1],
                np.gradient(dFdQ1, axis=1, edge_order=2)[1, 1],
            ],
        ]
    ) / (dQ * dQ)
    print(d2FdQdQ_0)
    print(d2FdQdQ)


print("d2FdQdZQ")
test_d2FdQdZQ()


def test_ceps_unrelaxed():
    P_0 = 1.0e5
    T_0 = 900.0

    qtz_a.equilibrate(P_0, T_0)

    qtz_a.set_relaxation(True)
    X_0 = qtz_a.molar_fractions
    eps_0 = logm(qtz_a.deformation_gradient_tensor)
    helmholtz_0 = qtz_a.helmholtz
    V_0 = qtz_a.V

    ceps_unrelaxed = qtz_a.molar_isometric_heat_capacity_unrelaxed

    def helmholtz_fixed_T(dT):
        return helmholtz_fixed_Q(
            np.array([0.0, 0.0]),
            np.zeros((3, 3)),
            dT,
            V_0,
            T_0,
            X_0,
            eps_0,
            helmholtz_0,
        )

    dT = 1.0
    F0 = helmholtz_fixed_T(-dT)
    F1 = helmholtz_fixed_T(0)
    F2 = helmholtz_fixed_T(dT)

    ceps = -T_0 * ((F2 - F1) - (F1 - F0)) / (dT * dT)

    print(ceps)
    print(ceps_unrelaxed)


print("ceps_unrelaxed")
test_ceps_unrelaxed()


def test_ceps_relaxed():
    P_0 = 1.0e5
    T_0 = 900.0

    qtz_a.equilibrate(P_0, T_0)

    qtz_a.set_relaxation(True)
    X_0 = qtz_a.molar_fractions
    eps_0 = logm(qtz_a.deformation_gradient_tensor)
    helmholtz_0 = qtz_a.helmholtz
    V_0 = qtz_a.V

    ceps_relaxed = qtz_a.molar_isometric_heat_capacity

    dT = 1.0
    F0 = minimize(
        helmholtz_fixed_Q1,
        [0.001],
        args=(np.zeros((3, 3)), -dT, V_0, T_0, X_0, eps_0, helmholtz_0),
    ).fun
    F1 = 0.0
    F2 = minimize(
        helmholtz_fixed_Q1,
        [0.001],
        args=(np.zeros((3, 3)), dT, V_0, T_0, X_0, eps_0, helmholtz_0),
    ).fun

    ceps = -T_0 * ((F2 - F1) - (F1 - F0)) / (dT * dT)

    print(ceps)
    print(ceps_relaxed)


print("ceps_relaxed")
test_ceps_relaxed()


def test_pi_unrelaxed():
    P_0 = 1.0e5
    T_0 = 900.0

    qtz_a.equilibrate(P_0, T_0)

    qtz_a.set_relaxation(True)
    X_0 = qtz_a.molar_fractions
    eps_0 = logm(qtz_a.deformation_gradient_tensor)
    helmholtz_0 = qtz_a.helmholtz
    V_0 = qtz_a.V

    pi_unrelaxed = qtz_a.thermal_stress_tensor_unrelaxed

    def helmholtz_fixed_T(dT):
        return helmholtz_fixed_Q(
            np.array([0.0, 0.0]),
            np.zeros((3, 3)),
            dT,
            V_0,
            T_0,
            X_0,
            eps_0,
            helmholtz_0,
        )

    dT = 1.0
    helmholtz_fixed_T(-dT / 2.0)
    eps_1 = logm(qtz_a.deformation_gradient_tensor)
    eps_prime_1 = eps_0 - eps_1
    sigma_1 = -qtz_a.pressure * np.eye(3) + np.einsum(
        "ijkl, kl->ij",
        qtz_a.full_isothermal_stiffness_tensor_unrelaxed,
        eps_prime_1,
    )

    helmholtz_fixed_T(dT / 2.0)
    eps_2 = logm(qtz_a.deformation_gradient_tensor)
    eps_prime_2 = eps_0 - eps_2
    sigma_2 = -qtz_a.pressure * np.eye(3) + np.einsum(
        "ijkl, kl->ij",
        qtz_a.full_isothermal_stiffness_tensor_unrelaxed,
        eps_prime_2,
    )
    # assert sol.success
    pi = (sigma_2 - sigma_1) / dT

    np.printoptions(precision=4)
    print(pi)
    print(pi_unrelaxed)

    pi_diff = (pi - pi_unrelaxed).flatten()
    max_diff = 0.0
    vals_max_diff = [0.0, 0.0]
    for i, p in enumerate(pi_unrelaxed.flatten()):
        if np.abs(p) > 1.0:
            if np.abs(pi_diff[i] / p) > max_diff:
                max_diff = np.abs(pi_diff[i] / p)
                vals_max_diff = [pi_diff[i] / 1.0e9, p / 1.0e9]

    print(
        f"Percent error: {max_diff*100.}, diff: {vals_max_diff[0]:.2f} GPa, val: {vals_max_diff[1]:.2f} GPa"
    )


print("pi_unrelaxed")
test_pi_unrelaxed()


def test_pi_relaxed():
    P_0 = 1.0e5
    T_0 = 900.0

    qtz_a.equilibrate(P_0, T_0)

    qtz_a.set_relaxation(True)
    X_0 = qtz_a.molar_fractions
    eps_0 = logm(qtz_a.deformation_gradient_tensor)
    helmholtz_0 = qtz_a.helmholtz
    V_0 = qtz_a.V

    pi_relaxed = qtz_a.thermal_stress_tensor

    dT = 1.0
    sol = minimize(
        helmholtz_fixed_Q1,
        [0.001],
        args=(np.zeros((3, 3)), -dT / 2.0, V_0, T_0, X_0, eps_0, helmholtz_0),
    )
    Q1 = sol.x
    helmholtz_fixed_Q1(
        Q1, np.zeros((3, 3)), -dT / 2.0, V_0, T_0, X_0, eps_0, helmholtz_0
    )

    eps_1 = logm(qtz_a.deformation_gradient_tensor)
    eps_prime_1 = eps_0 - eps_1
    sigma_1 = -qtz_a.pressure * np.eye(3) + np.einsum(
        "ijkl, kl->ij",
        qtz_a.full_isothermal_stiffness_tensor_unrelaxed,
        eps_prime_1,
    )

    sol = minimize(
        helmholtz_fixed_Q1,
        [0.001],
        args=(np.zeros((3, 3)), dT / 2.0, V_0, T_0, X_0, eps_0, helmholtz_0),
    )
    Q1 = sol.x
    helmholtz_fixed_Q1(
        Q1, np.zeros((3, 3)), dT / 2.0, V_0, T_0, X_0, eps_0, helmholtz_0
    )
    eps_2 = logm(qtz_a.deformation_gradient_tensor)
    eps_prime_2 = eps_0 - eps_2
    sigma_2 = -qtz_a.pressure * np.eye(3) + np.einsum(
        "ijkl, kl->ij",
        qtz_a.full_isothermal_stiffness_tensor_unrelaxed,
        eps_prime_2,
    )
    # assert sol.success
    pi = (sigma_2 - sigma_1) / dT

    np.printoptions(precision=4)
    print(pi)
    print(pi_relaxed)

    pi_diff = (pi - pi_relaxed).flatten()
    max_diff = 0.0
    vals_max_diff = [0.0, 0.0]
    for i, p in enumerate(pi_relaxed.flatten()):
        if np.abs(p) > 1.0:
            if np.abs(pi_diff[i] / p) > max_diff:
                max_diff = np.abs(pi_diff[i] / p)
                vals_max_diff = [pi_diff[i] / 1.0e3, p / 1.0e3]

    print(
        f"Percent error: {max_diff*100.}, diff: {vals_max_diff[0]:.2f} kPa, val: {vals_max_diff[1]:.2f} kPa"
    )


print("pi_relaxed")
test_pi_relaxed()


def test_CT_unrelaxed():
    P_0 = 1.0e5
    T_0 = 900.0

    d_eps = 1.0e-5
    qtz_a.equilibrate(P_0, T_0)

    qtz_a.set_relaxation(True)
    X_0 = qtz_a.molar_fractions
    eps_0 = logm(qtz_a.deformation_gradient_tensor)
    helmholtz_0 = qtz_a.helmholtz
    V_0 = qtz_a.V

    CT_unrelaxed_Voigt = qtz_a.isothermal_stiffness_tensor_unrelaxed

    CT = np.empty((3, 3, 3, 3))

    for i in range(3):
        for j in range(3):

            deps = np.zeros((3, 3))
            deps[i][j] -= d_eps / 2.0
            helmholtz_fixed_Q1([0.0], deps, 0.0, V_0, T_0, X_0, eps_0, helmholtz_0)

            eps_1 = logm(qtz_a.deformation_gradient_tensor)
            eps_prime_1 = eps_0 + deps - eps_1
            sigma_1 = -qtz_a.pressure * np.eye(3) + np.einsum(
                "ijkl, kl->ij",
                qtz_a.full_isothermal_stiffness_tensor_unrelaxed,
                eps_prime_1,
            )
            deps = np.zeros((3, 3))
            deps[i][j] += d_eps / 2.0
            helmholtz_fixed_Q1([0.0], deps, 0.0, V_0, T_0, X_0, eps_0, helmholtz_0)
            eps_2 = logm(qtz_a.deformation_gradient_tensor)
            eps_prime_2 = eps_0 + deps - eps_2
            sigma_2 = -qtz_a.pressure * np.eye(3) + np.einsum(
                "ijkl, kl->ij",
                qtz_a.full_isothermal_stiffness_tensor_unrelaxed,
                eps_prime_2,
            )
            # assert sol.success
            CT[:, :, i, j] = (sigma_2 - sigma_1) / d_eps

    CT = contract_stiffnesses(CT)

    np.printoptions(precision=4)
    print(CT)
    print(CT_unrelaxed_Voigt)

    CT_diff = (CT - CT_unrelaxed_Voigt).flatten()
    max_diff = 0.0
    vals_max_diff = [0.0, 0.0]
    for i, C in enumerate(CT_unrelaxed_Voigt.flatten()):
        if np.abs(C) > 1.0:
            if np.abs(CT_diff[i] / C) > max_diff:
                max_diff = np.abs(CT_diff[i] / C)
                vals_max_diff = [CT_diff[i] / 1.0e9, C / 1.0e9]

    print(
        f"Percent error: {max_diff*100.}, diff: {vals_max_diff[0]:.2f} GPa, val: {vals_max_diff[1]:.2f} GPa"
    )


print("CT_unrelaxed")
test_CT_unrelaxed()


def test_CT_relaxed():
    P_0 = 1.0e5
    T_0 = 900.0

    d_eps = 1.0e-5
    qtz_a.equilibrate(P_0, T_0)

    qtz_a.set_relaxation(True)
    X_0 = qtz_a.molar_fractions
    eps_0 = logm(qtz_a.deformation_gradient_tensor)
    helmholtz_0 = qtz_a.helmholtz
    V_0 = qtz_a.V

    CT_relaxed_Voigt = qtz_a.isothermal_stiffness_tensor

    CT = np.empty((3, 3, 3, 3))

    for i in range(3):
        for j in range(3):
            print(i, j)
            deps = np.zeros((3, 3))
            deps[i][j] -= d_eps / 2.0
            sol = minimize(
                helmholtz_fixed_Q1,
                [0.001],
                args=(deps, 0.0, V_0, T_0, X_0, eps_0, helmholtz_0),
            )
            Q1 = sol.x
            helmholtz_fixed_Q1(Q1, deps, 0.0, V_0, T_0, X_0, eps_0, helmholtz_0)
            eps_1 = logm(qtz_a.deformation_gradient_tensor)
            eps_prime_1 = eps_0 + deps - eps_1
            sigma_1 = -qtz_a.pressure * np.eye(3) + np.einsum(
                "ijkl, kl->ij",
                qtz_a.full_isothermal_stiffness_tensor_unrelaxed,
                eps_prime_1,
            )
            deps = np.zeros((3, 3))
            deps[i][j] += d_eps / 2.0
            sol = minimize(
                helmholtz_fixed_Q1,
                [0.001],
                args=(deps, 0.0, V_0, T_0, X_0, eps_0, helmholtz_0),
            )
            Q2 = sol.x
            helmholtz_fixed_Q1(Q2, deps, 0.0, V_0, T_0, X_0, eps_0, helmholtz_0)
            eps_2 = logm(qtz_a.deformation_gradient_tensor)
            eps_prime_2 = eps_0 + deps - eps_2
            sigma_2 = -qtz_a.pressure * np.eye(3) + np.einsum(
                "ijkl, kl->ij",
                qtz_a.full_isothermal_stiffness_tensor_unrelaxed,
                eps_prime_2,
            )
            # assert sol.success
            CT[:, :, i, j] = (sigma_2 - sigma_1) / d_eps

    CT = contract_stiffnesses(CT)

    np.printoptions(precision=4)
    print(CT)
    print(CT_relaxed_Voigt)

    CT_diff = (CT - CT_relaxed_Voigt).flatten()
    max_diff = 0.0
    vals_max_diff = [0.0, 0.0]
    for i, C in enumerate(CT_relaxed_Voigt.flatten()):
        if np.abs(C) > 1.0:
            if np.abs(CT_diff[i] / C) > max_diff:
                max_diff = np.abs(CT_diff[i] / C)
                vals_max_diff = [CT_diff[i] / 1.0e9, C / 1.0e9]

    print(
        f"Percent error: {max_diff*100.}, diff: {vals_max_diff[0]:.2f} GPa, val: {vals_max_diff[1]:.2f} GPa"
    )


print("CT_relaxed")
test_CT_relaxed()
