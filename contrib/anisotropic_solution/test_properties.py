from __future__ import absolute_import
from __future__ import print_function
import unittest
import numpy as np
from copy import deepcopy
from scipy.linalg import logm
from scipy.optimize import minimize
from burnman.utils.anisotropy import (
    voigt_notation_to_compliance_tensor,
    contract_stresses,
    contract_stiffnesses,
    contract_compliances,
)
from quartz_model import make_anisotropic_model_3


class BurnManTest(unittest.TestCase):
    def assertFloatEqual(self, a, b, tol=1e-5, tol_zero=1e-16):
        self.assertAlmostEqual(a, b, delta=max(tol_zero, max(abs(a), abs(b)) * tol))

    def assertArraysAlmostEqual(self, a, b, tol=1e-5, tol_zero=1e-16):
        self.assertEqual(len(a), len(b))
        for (i1, i2) in zip(a, b):
            self.assertFloatEqual(i1, i2, tol, tol_zero)


def helmholtz_fixed_Q(dQ, deps, dT, V_0, T_0, X_0, eps_0, helmholtz_0):
    try:
        V_1 = V_0 * (np.trace(deps) + 1.0)
        T_1 = T_0 + dT

        dX = np.array([dQ[0] + dQ[1], -dQ[0], -dQ[1]])
        X_1 = X_0 + dX
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
        np.array([args[0] / 2.0, 0.0]), deps, dT, V_0, T_0, X_0, eps_0, helmholtz_0
    )


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

if False:
    P_0 = 1.0e5
    T_0 = 900.0
    dT = 1.0
    qtz_a.equilibrate(P_0, T_0)

    S_N = qtz_a.isothermal_compliance_tensor
    beta_NR = qtz_a.isothermal_compressibility_reuss

    dTdP = qtz_a.isentropic_thermal_gradient
    dP = 1.0e4
    P1 = P_0 - dP / 2.0
    P2 = P_0 + dP / 2.0

    T1 = T_0  # - dTdP * dP / 2.0
    T2 = T_0  # + dTdP * dP / 2.0

    qtz_a.set_state(P1, T1)

    Psi_1 = qtz_a.Psi
    f_1 = np.log(qtz_a.V)
    qtz_a.set_state(P2, T2)
    f_2 = np.log(qtz_a.V)

    Psi_2 = qtz_a.Psi

    dPsidf_Voigt = contract_compliances((Psi_2 - Psi_1) / (f_2 - f_1))
    print(dPsidf_Voigt)
    print(S_N / beta_NR)
    print(np.max(np.abs((dPsidf_Voigt.flatten() - (S_N / beta_NR).flatten()))))

    exit()


class anisotropic_derivatives(BurnManTest):
    def test_dPsidf_unrelaxed(self):
        P0 = 1.0e5
        T0 = 900.0
        qtz_a.equilibrate(P0, T0)

        qtz_a.set_relaxation(False)
        S_T = qtz_a.isothermal_compliance_tensor
        beta_TR = qtz_a.isothermal_compressibility_reuss
        qtz_a.set_relaxation(True)

        dP = 1.0e4
        P1 = P0 - dP / 2.0
        P2 = P0 + dP / 2.0

        qtz_a.set_state(P1, T0)
        Psi_1 = qtz_a.Psi
        f_1 = np.log(qtz_a.V)

        qtz_a.set_state(P2, T0)
        f_2 = np.log(qtz_a.V)
        Psi_2 = qtz_a.Psi

        dPsidf_Voigt = contract_compliances((Psi_2 - Psi_1) / (f_2 - f_1))
        self.assertArraysAlmostEqual(dPsidf_Voigt.flatten(), (S_T / beta_TR).flatten())

    def test_dPSmudQ(self):
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

        self.assertArraysAlmostEqual(dPdX.dot(qtz_a.dXdQ), dPdQ_0, tol_zero=1.0)
        self.assertArraysAlmostEqual(dSdX.dot(qtz_a.dXdQ), dSdQ_0, tol_zero=1.0e-8)
        self.assertArraysAlmostEqual(
            qtz_a.dXdQ.T.dot(dmudX.dot(qtz_a.dXdQ)).flatten(),
            dmudQ_0.flatten(),
            tol_zero=1.0e-4,
        )

    def test_Psi_mbr(self):
        P_0 = 1.0e5
        T_0 = 900.0

        qtz_a.set_composition([1.0, 0.0, 0.0])
        qtz_a.set_state(P_0, T_0)

        a = qtz_a.scalar_solution.endmembers[0][0]
        self.assertArraysAlmostEqual(a.Psi.flatten(), qtz_a.Psi.flatten())

    def test_state(self):
        P_0 = 1.0e5
        T_0 = 900.0
        qtz_a.set_relaxation(False)
        qtz_a.equilibrate(P_0, T_0)
        qtz_a.set_composition([1.0, 0.0, 0.0])

        a = qtz_a.scalar_solution.endmembers[0][0]

        self.assertArraysAlmostEqual(a.Psi.flatten(), qtz_a.Psi.flatten())
        self.assertFloatEqual(a.alpha, qtz_a.alpha)
        self.assertFloatEqual(a.alpha, qtz_a.alpha)
        self.assertArraysAlmostEqual(
            a.dPsidP_Voigt.flatten(), qtz_a.dPsidP_fixed_T_Voigt.flatten()
        )
        self.assertArraysAlmostEqual(
            a.dPsidT_Voigt.flatten(), qtz_a.dPsidT_fixed_P_Voigt.flatten()
        )
        qtz_a.set_relaxation(True)

    def test_dPsidPT1a(self):
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

        self.assertArraysAlmostEqual(
            dPsidP.flatten() * 1.0e9, dPsidP_0.flatten() * 1.0e9, tol=1.0e-4
        )
        self.assertArraysAlmostEqual(dPsidT.flatten(), dPsidT_0.flatten())

    def test_dPsidPT2a(self):
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

        self.assertArraysAlmostEqual(
            dPsidP.flatten() * 1.0e9, dPsidP_0.flatten() * 1.0e9
        )
        self.assertArraysAlmostEqual(dPsidT.flatten(), dPsidT_0.flatten())

    def test_dPsidPT1b(self):
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

        self.assertArraysAlmostEqual(
            dPsidP.flatten() * 1.0e9, dPsidP_0.flatten() * 1.0e9, tol=1.0e-4
        )
        self.assertArraysAlmostEqual(dPsidT.flatten(), dPsidT_0.flatten())

    def test_dPsidPT2b(self):
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

        self.assertArraysAlmostEqual(
            dPsidP.flatten() * 1.0e9, dPsidP_0.flatten() * 1.0e9
        )
        self.assertArraysAlmostEqual(dPsidT.flatten(), dPsidT_0.flatten())

    def test_depsdQT(self):
        P_0 = 1.0e5
        T_0 = 800.0
        qtz_a.equilibrate(P_0, T_0)

        X_0 = qtz_a.molar_fractions
        V_0 = qtz_a.V

        depsdQ_0 = qtz_a.depsdQ
        depsdT_0 = qtz_a.depsdT

        def depsdQordT(dQ, dT):
            X_1 = X_0 + np.array(
                [-dQ[0] / 2.0 - dQ[1] / 2.0, +dQ[0] / 2.0, dQ[1] / 2.0]
            )
            qtz_a.set_composition(X_1)
            qtz_a.set_state_with_volume(V_0, T_0 - dT / 2.0)
            eps_1 = logm(qtz_a.deformation_gradient_tensor)

            X_2 = X_0 + np.array(
                [dQ[0] / 2.0 + dQ[1] / 2.0, -dQ[0] / 2.0, -dQ[1] / 2.0]
            )
            qtz_a.set_composition(X_2)
            qtz_a.set_state_with_volume(V_0, T_0 + dT / 2.0)
            eps_2 = logm(qtz_a.deformation_gradient_tensor)
            return eps_2 - eps_1

        deps = 1.0e-1
        dT = 0.1
        i, v = [1, np.eye(2)[1]]

        v1 = depsdQordT(v * deps, 0.0) / deps
        v2 = depsdQ_0[:, :, i]
        self.assertArraysAlmostEqual(v1.flatten(), v2.flatten(), tol_zero=1.0e-10)

        v1 = depsdQordT([0.0, 0.0], dT) / dT
        v2 = depsdT_0
        self.assertArraysAlmostEqual(v1.flatten(), v2.flatten(), tol_zero=1.0e-10)

    def test_d2FdXdX(self):
        qtz_a.equilibrate(1.0e5, 800.0)
        qtz_a.set_composition([1.0, 0.0, 0.0])
        X0 = qtz_a.molar_fractions
        V = qtz_a.V
        dX = 1.0e-4
        F = np.zeros((3, 3, 3))

        d2FdXdX_0 = qtz_a.d2FdXdX

        for i in range(3):
            for j in range(3):
                for k in range(3):

                    dXa = np.array([(i - 1.0) * dX, (j - 1.0) * dX, (k - 1.0) * dX])
                    sumX = np.sum(X0 + dXa)
                    X = (X0 + dXa) / sumX

                    qtz_a.set_composition(X)
                    qtz_a.set_state_with_volume(V / sumX, 800.0)
                    F[i, j, k] = qtz_a.helmholtz * sumX

        dFdX0 = np.gradient(F, axis=0, edge_order=2)
        dFdX1 = np.gradient(F, axis=1, edge_order=2)
        dFdX2 = np.gradient(F, axis=2, edge_order=2)

        d2FdXdX = np.array(
            [
                [
                    np.gradient(dFdX0, axis=0, edge_order=2)[1, 1, 1],
                    np.gradient(dFdX0, axis=1, edge_order=2)[1, 1, 1],
                    np.gradient(dFdX0, axis=2, edge_order=2)[1, 1, 1],
                ],
                [
                    np.gradient(dFdX1, axis=0, edge_order=2)[1, 1, 1],
                    np.gradient(dFdX1, axis=1, edge_order=2)[1, 1, 1],
                    np.gradient(dFdX1, axis=2, edge_order=2)[1, 1, 1],
                ],
                [
                    np.gradient(dFdX2, axis=0, edge_order=2)[1, 1, 1],
                    np.gradient(dFdX2, axis=1, edge_order=2)[1, 1, 1],
                    np.gradient(dFdX2, axis=2, edge_order=2)[1, 1, 1],
                ],
            ]
        ) / (dX * dX)

        self.assertArraysAlmostEqual(d2FdXdX_0.flatten(), d2FdXdX.flatten())

    def test_d2FdQdZQ(self):
        # Fixed strain and/or temperature
        P_0 = 1.0e5
        T_0 = 900.0

        qtz_a.set_relaxation(True)
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
        self.assertArraysAlmostEqual(
            d2FdQdZ.flatten(), d2FdQdZ_0[1, :].flatten(), tol_zero=1.0e-7
        )

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

        self.assertArraysAlmostEqual(d2FdQdQ_0.flatten(), d2FdQdQ.flatten())

        qtz_a.set_relaxation(True)

    def test_ceps_unrelaxed(self):
        P_0 = 1.0e5
        T_0 = 900.0

        qtz_a.equilibrate(P_0, T_0)

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

        self.assertFloatEqual(ceps, ceps_unrelaxed)

    def test_ceps_relaxed(self):
        P_0 = 1.0e5
        T_0 = 900.0

        qtz_a.equilibrate(P_0, T_0)

        X_0 = qtz_a.molar_fractions
        eps_0 = logm(qtz_a.deformation_gradient_tensor)
        helmholtz_0 = qtz_a.helmholtz
        V_0 = qtz_a.V

        ceps_relaxed = qtz_a.molar_isometric_heat_capacity

        # NOTE: If this test fails, it may be because the minimizer step
        # is rather finicky.
        dT = 1.0
        F0 = minimize(
            helmholtz_fixed_Q1,
            [0.1],
            args=(np.zeros((3, 3)), -dT, V_0, T_0, X_0, eps_0, helmholtz_0),
        ).fun
        F1 = 0.0
        F2 = minimize(
            helmholtz_fixed_Q1,
            [0.1],
            args=(np.zeros((3, 3)), dT, V_0, T_0, X_0, eps_0, helmholtz_0),
        ).fun

        ceps = -T_0 * ((F2 - F1) - (F1 - F0)) / (dT * dT)

        self.assertFloatEqual(ceps, ceps_relaxed)

    def test_pi_unrelaxed(self):
        P_0 = 1.0e5
        T_0 = 900.0

        qtz_a.equilibrate(P_0, T_0)

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

        self.assertArraysAlmostEqual(
            pi.flatten(), pi_unrelaxed.flatten(), tol_zero=1.0e-3
        )

    def test_pi_relaxed(self):
        P_0 = 1.0e5
        T_0 = 900.0

        qtz_a.equilibrate(P_0, T_0)

        X_0 = qtz_a.molar_fractions
        eps_0 = logm(qtz_a.deformation_gradient_tensor)
        helmholtz_0 = qtz_a.helmholtz
        V_0 = qtz_a.V

        pi_relaxed = qtz_a.thermal_stress_tensor

        # NOTE: If this test fails, it may be because the minimizer step
        # is rather finicky.
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

        self.assertArraysAlmostEqual(
            pi.flatten(), pi_relaxed.flatten(), tol=3.0e-3, tol_zero=1.0e-2
        )

    def test_CT_unrelaxed(self):
        P_0 = 1.0e5
        T_0 = 900.0

        d_eps = 1.0e-5
        qtz_a.equilibrate(P_0, T_0)

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
        self.assertArraysAlmostEqual(
            CT.flatten(), CT_unrelaxed_Voigt.flatten(), tol_zero=100.0
        )

    def test_CT_relaxed(self):
        P_0 = 1.0e5
        T_0 = 900.0

        d_eps = 1.0e-5
        qtz_a.equilibrate(P_0, T_0)

        X_0 = qtz_a.molar_fractions
        eps_0 = logm(qtz_a.deformation_gradient_tensor)
        helmholtz_0 = qtz_a.helmholtz
        V_0 = qtz_a.V

        CT_relaxed = qtz_a.full_isothermal_stiffness_tensor

        CT = np.empty((3, 3, 3, 3))

        # Note change the i and j ranges to 3 to test the full tensor
        # NOTE: If this test fails, it may be because the minimizer step
        # is rather finicky.
        for i in range(1):
            for j in range(1):
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

                self.assertArraysAlmostEqual(
                    CT[:, :, i, j].flatten(),
                    CT_relaxed[:, :, i, j].flatten(),
                    tol=3.0e-3,
                    tol_zero=100.0,
                )


if __name__ == "__main__":
    unittest.main()
