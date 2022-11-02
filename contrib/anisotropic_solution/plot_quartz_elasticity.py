import matplotlib.pyplot as plt
import numpy as np
import string
from data_tables import data
from quartz_model import make_anisotropic_model_4

#    a11, a33, a44, a14 = args[0:4]
#    b11, b33, b44, b14 = args[4:8]
#    c44, c14 = args[8:10]
#    d11, d33, d44, d14 = args[10:14]

# 6342
args = [
    3.08023469e-02,
    7.11196588e-01,
    1.43134903e00,
    -9.41929127e-02,
    -1.43295172e00,
    1.20198198e-02,
    -7.29519270e-02,
    -1.24960228e00,
    2.35467349e-02,
    1.87706618e-02,
    3.69723158e-02,
    -1.16882690e-01,
    -6.42735902e-01,
    5.63749967e-01,
    4.35422899e-02,
    -2.47530735e-02,
    7.97487126e-03,
    2.20609966e-02,
    -1.12603343e-02,
    -9.38895870e-05,
    3.01706139e-01,
    -3.31787168e-02,
    -7.12771915e-01,
    7.93822184e-03,
    -2.51231281e-03,
    8.46341094e-05,
    -5.43208946e-01,
    -3.81465199e-03,
    5.31775526e-04,
]


qtz_a = make_anisotropic_model_4(args)

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
    temperatures = np.linspace(1.0, 1400.0, 100)
    CS = np.empty((len(temperatures), 6, 6))
    KS = np.empty(len(temperatures))
    KT = np.empty(len(temperatures))
    KTest = np.empty(len(temperatures))
    V = np.empty(len(temperatures))
    cell_parameters = np.empty((len(temperatures), 6))

    for i, T in enumerate(temperatures):
        if i % 100 == 0:
            print(T)
        qtz_a.equilibrate(1.0e5, T)

        CS[i] = qtz_a.isentropic_stiffness_tensor
        KS[i] = qtz_a.isentropic_bulk_modulus_reuss
        KT[i] = qtz_a.isothermal_bulk_modulus_reuss
        V[i] = qtz_a.V
        cell_parameters[i] = qtz_a.cell_parameters

        dP = 1.0e4
        qtz_a.equilibrate(1.0e5 + dP, T)
        # qtz_a.set_state(1.0e5 + dP, T)
        V2 = qtz_a.V
        KTest[i] = -((V2 + V[i]) / 2.0) * dP / (V2 - V[i])

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

    for (i, j) in [(1, 1), (3, 3), (4, 4), (1, 2), (1, 3), (1, 4)]:
        plt.plot(temperatures, CS[:, i - 1, j - 1] / 1.0e9, label=f"$C_{{S{i}{j}}}$")
        plt.scatter(data["Lakshtanov"]["T_K"], data["Lakshtanov"][f"C{i}{j}"])

    plt.plot(temperatures, KS / 1.0e9, label="$K_S$")
    plt.plot(temperatures, KT / 1.0e9, label="$K_T$")
    plt.plot(temperatures, KTest / 1.0e9, linestyle="--", label="$K_T est$")

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
