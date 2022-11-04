import matplotlib.pyplot as plt
import numpy as np
import string
from data_tables import data
from quartz_model import (
    make_anisotropic_model_4,
    make_anisotropic_model_5,
    equilibrium_Q,
)
from burnman.utils.anisotropy import contract_compliances

#    a11, a33, a44, a14 = args[0:4]
#    b11, b33, b44, b14 = args[4:8]
#    c44, c14 = args[8:10]
#    d11, d33, d44, d14 = args[10:14]


# 5238
args = [
    -4.41413951e-01,
    8.01640333e-01,
    1.51285308e00,
    -2.43434388e-03,
    -2.28445678e00,
    1.50825700e-01,
    2.94153157e-02,
    -9.46031419e00,
    5.37303170e-03,
    3.22659916e-02,
    -1.19700530e-01,
    -1.12143669e-01,
    -9.44354466e-01,
    4.51094817e-01,
    2.14961826e-01,
    -2.12340655e-02,
    3.59672487e-01,
    -3.29118453e-02,
    -4.91200424e-03,
    -6.52961965e-03,
    3.01520115e-01,
    -2.60152302e-02,
    -6.94684348e-01,
    9.29285883e-03,
    -3.11614913e-03,
    -7.03987125e-05,
    -5.43785736e-01,
    -1.61745972e-03,
    6.74159701e-03,
]


"""
    a11, a33, a44, a14 = args[0:4]
    b11, b33, b44, b14 = args[4:8]
    c44, c14 = args[8:10]
    d11, d33, d44, d14 = args[10:14]
    e11, e33, e44 = args[14:17]
    f11, f33, f44 = args[17:20]
"""

args2 = [
    -3.84561803e-01,
    7.38299245e-01,
    1.5674533e00,  #
    -4.95936133e-02 * 8.0,
    -2.32624286e00,
    6.42776707e-02,
    2.35100601e-01,  #
    5.54699868e00 * 4.0,
    3.89844677e-01,  #
    1.0e-0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.01387550e-01,
    -4.43459975e-02,
    -1e-1,  #
    0.0,
    3.95126381e-02,
    -1.64471070e-02,
    0.0,  #
]


qtz_a = make_anisotropic_model_4(args)
m = qtz_a


def get_dPsidf(P, T):
    m.equilibrate(P, T)
    dTdP = m.isentropic_thermal_gradient
    dP = 1.0e4

    m.equilibrate(P - dP / 2.0, T - dTdP * dP / 2.0)
    Psi0 = m.Psi
    f0 = np.log(m.V)
    m.equilibrate(P + dP / 2.0, T + dTdP * dP / 2.0)
    Psi1 = m.Psi
    f1 = np.log(m.V)

    return contract_compliances(Psi1 - Psi0) / (f1 - f0)


if False:
    a = qtz_a.scalar_solution.endmembers[0][0]
    a.set_state(1.0e5, 300.0)
    print(a.V, a.Pth, a.cell_parameters[0] / a.cell_parameters[2])
    a.set_state(2866498400.932131, 300.0)
    print(a.V, a.Pth, a.cell_parameters[0] / a.cell_parameters[2])

    dQdX = np.array([[1.0, 1.0, -1.0], [1.0, -1.0, 1.0]]).T
    qtz_a.equilibrate(1.0e5, 300.0)
    print(
        qtz_a.V,
        qtz_a._Pth_mbr.dot(qtz_a.molar_fractions),
        qtz_a.molar_fractions.dot(dQdX),
        qtz_a.cell_parameters[0] / qtz_a.cell_parameters[2],
    )

    qtz_a.equilibrate(1.0e5, 800.0)
    print(
        qtz_a.V,
        qtz_a._Pth_mbr.dot(qtz_a.molar_fractions),
        qtz_a.molar_fractions.dot(dQdX),
        qtz_a.cell_parameters[0] / qtz_a.cell_parameters[2],
    )

    qtz_a.equilibrate(1.0e10, 300.0)
    print(
        qtz_a.V,
        qtz_a._Pth_mbr.dot(qtz_a.molar_fractions),
        qtz_a.molar_fractions.dot(dQdX),
        qtz_a.cell_parameters[0] / qtz_a.cell_parameters[2],
    )

    temperatures = np.linspace(1.0, 2000.0, 100)
    cell_parameters = np.empty((len(temperatures), 6))

    for P in [1.0e5, 0.3e9, 1.0e9]:
        for i, T in enumerate(temperatures):
            if i % 100 == 0:
                print(T)
            qtz_a.equilibrate(P, T)
            cell_parameters[i] = qtz_a.cell_parameters

        plt.plot(temperatures, cell_parameters[:, 0] / cell_parameters[:, 2])

    for ds in ["Carpenter_neutron", "Carpenter_XRD", "Raz"]:
        plt.scatter(data[ds]["T_K"], data[ds]["a"] / data[ds]["c"], label=ds)

    plt.show()

    pressures = np.linspace(1.0e5, 20.0e9, 100)
    for T in [300.0]:
        for i, P in enumerate(pressures):
            if i % 100 == 0:
                print(P)
            qtz_a.equilibrate(P, T)
            cell_parameters[i] = qtz_a.cell_parameters

        plt.plot(pressures / 1.0e9, cell_parameters[:, 0] / cell_parameters[:, 2])

    for ds in ["Scheidl"]:
        plt.scatter(data[ds]["P_GPa"], data[ds]["a"] / data[ds]["c"], label=ds)
    plt.show()


if True:
    temperatures = np.linspace(300.0, 1300.0, 100)
    SS = np.empty((len(temperatures), 6, 6))
    betaS = np.empty(len(temperatures))
    betaT = np.empty(len(temperatures))
    betaTest = np.empty(len(temperatures))
    V = np.empty(len(temperatures))
    cell_parameters = np.empty((len(temperatures), 6))
    Q_active_sq = np.empty(len(temperatures))

    for i, T in enumerate(temperatures):
        if i % 100 == 0:
            print(T)
        qtz_a.equilibrate(1.0e5, T)
        Qs = equilibrium_Q(1.0e5, T)
        Q_active_sq[i] = Qs[0] * Qs[0] if Qs[0] > 0.0 else -Qs[1] * Qs[1]
        SS[i] = qtz_a.isentropic_compliance_tensor
        betaS[i] = qtz_a.isentropic_compressibility_reuss
        betaT[i] = qtz_a.isothermal_compressibility_reuss
        V[i] = qtz_a.V
        cell_parameters[i] = qtz_a.cell_parameters

        dP = 1.0e4
        qtz_a.equilibrate(1.0e5 + dP, T)
        # qtz_a.set_state(1.0e5 + dP, T)
        V2 = qtz_a.V
        betaTest[i] = 1.0 / (-((V2 + V[i]) / 2.0) * dP / (V2 - V[i]))

        # print(1.0 - np.linalg.det(qtz_a.cell_vectors) / qtz_a.V)

    for i in range(3):
        cell_params = cell_parameters[:, i]
        plt.plot(temperatures, cell_params, label=f"{string.ascii_lowercase[i]}")

    for ds in ["Carpenter_neutron", "Carpenter_XRD"]:
        plt.scatter(data[ds]["T_K"], data[ds]["a"], label="a")
        plt.scatter(data[ds]["T_K"], data[ds]["a"], label="b")
        plt.scatter(data[ds]["T_K"], data[ds]["c"], label="c")

    plt.plot(temperatures, np.cbrt(V), linestyle=":", label="cbrt(V)")
    plt.legend()
    plt.show()

    aoverc = cell_parameters[:, 0] / cell_parameters[:, 2]
    plt.plot(temperatures, aoverc, label="a/c")

    for ds in ["Carpenter_neutron", "Carpenter_XRD"]:
        plt.scatter(data[ds]["T_K"], data[ds]["a"] / data[ds]["c"], label=ds)

    plt.legend()
    plt.show()

    L_Qs = [
        equilibrium_Q(P_GPa * 1.0e9, T)
        for P_GPa, T in zip(data["Lakshtanov"]["P_GPa"], data["Lakshtanov"]["T_K"])
    ]
    L_Qs.extend(
        [
            equilibrium_Q(P_GPa * 1.0e9, T)
            for P_GPa, T in zip(data["Wang"]["P_GPa"], data["Wang"]["T_K"])
        ]
    )
    L_Q_active_sq = np.array([Q[0] * Q[0] if Q[0] > 0 else -Q[1] * Q[1] for Q in L_Qs])

    dPsidf = [
        get_dPsidf(P_GPa * 1.0e9, T)
        for P_GPa, T in zip(data["Lakshtanov"]["P_GPa"], data["Lakshtanov"]["T_K"])
    ]
    dPsidf.extend(
        [
            get_dPsidf(P_GPa * 1.0e9, T)
            for P_GPa, T in zip(data["Wang"]["P_GPa"], data["Wang"]["T_K"])
        ]
    )

    dPsidf = np.array(dPsidf)

    for (i, j) in [(1, 1), (3, 3), (4, 4), (1, 2), (1, 3), (1, 4)]:
        plt.plot(Q_active_sq, SS[:, i - 1, j - 1] / betaS, label=f"$S_{{S{i}{j}}}$")

        beta_ratio = np.concatenate(
            (
                data["Lakshtanov"][f"S{i}{j}"] * data["Lakshtanov"]["K_S"],
                data["Wang"][f"S{i}{j}"] * data["Wang"]["K_S"],
            )
        )
        plt.scatter(L_Q_active_sq, beta_ratio)

        plt.scatter(L_Q_active_sq, dPsidf[:, i - 1, j - 1], marker="+")

    plt.plot(Q_active_sq, betaS * 0.0 + 1.0, label="$\\beta_S$")
    plt.plot(Q_active_sq, betaT / betaS, label="$\\beta_T$")
    plt.plot(Q_active_sq, betaTest / betaS, linestyle="--", label="$\\beta_T est$")

    plt.legend()
    plt.show()

if True:
    pressures = np.linspace(1.0e5, 20.0e9, 101)
    CS = np.empty((len(pressures), 6, 6))
    KS = np.empty(len(pressures))
    KT = np.empty(len(pressures))
    KTest = np.empty(len(pressures))
    V = np.empty(len(pressures))
    cell_parameters = np.empty((len(pressures), 6))

    T = 300.0
    for i, P in enumerate(pressures):
        if i % 100 == 0:
            print(P / 1.0e9)
        qtz_a.equilibrate(P, T)

        CS[i] = qtz_a.isentropic_stiffness_tensor
        KS[i] = qtz_a.isentropic_bulk_modulus_reuss
        KT[i] = qtz_a.isothermal_bulk_modulus_reuss
        V[i] = qtz_a.V
        cell_parameters[i] = qtz_a.cell_parameters

        # print(1.0 - np.linalg.det(qtz_a.cell_vectors) / qtz_a.V)

    KTest = -np.gradient(pressures, np.log(V), edge_order=2)

    for i in range(3):
        cell_params = cell_parameters[:, i]
        plt.plot(pressures / 1.0e9, cell_params, label=f"{string.ascii_lowercase[i]}")

    for ds in ["Scheidl"]:
        plt.scatter(data[ds]["P_GPa"], data[ds]["a"], label="a")
        plt.scatter(data[ds]["P_GPa"], data[ds]["a"], label="b")
        plt.scatter(data[ds]["P_GPa"], data[ds]["c"], label="c")

        Vobs = (np.sqrt(3) / 2) * (data[ds]["a"] * data[ds]["a"] * data[ds]["c"])

        plt.scatter(data[ds]["P_GPa"], np.cbrt(Vobs), label="cbrt(V)")

    Vmod = (np.sqrt(3) / 2) * (
        cell_parameters[:, 0] * cell_parameters[:, 0] * cell_parameters[:, 2]
    )
    plt.plot(
        pressures / 1.0e9, np.cbrt(Vmod), linestyle=":", label="cbrt(V from a, a, c)"
    )
    plt.plot(pressures / 1.0e9, np.cbrt(V), linestyle=":", label="cbrt(V from V)")
    plt.legend()
    plt.show()

    aoverc = cell_parameters[:, 0] / cell_parameters[:, 2]
    plt.plot(pressures / 1.0e9, aoverc, label="a/c")

    for ds in ["Scheidl"]:
        plt.scatter(data[ds]["P_GPa"], data[ds]["a"] / data[ds]["c"], label=ds)

    plt.legend()
    plt.show()

    for (i, j) in [(1, 1), (3, 3), (4, 4), (1, 2), (1, 3), (1, 4)]:
        plt.plot(
            pressures / 1.0e9, CS[:, i - 1, j - 1] / 1.0e9, label=f"$C_{{S{i}{j}}}$"
        )
        plt.scatter(data["Wang"]["P_GPa"], data["Wang"][f"C{i}{j}"])

    plt.plot(pressures / 1.0e9, KS / 1.0e9, label="$K_S$")
    plt.plot(pressures / 1.0e9, KT / 1.0e9, label="$K_T$")
    plt.plot(pressures / 1.0e9, KTest / 1.0e9, linestyle="--", label="$K_T est$")

    plt.legend()
    plt.show()
