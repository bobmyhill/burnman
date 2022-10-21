import matplotlib.pyplot as plt
import numpy as np
import string
from data_tables import data
from quartz_model import make_anisotropic_model_4

#    a11, a33, a44, a14 = args[0:4]
#    b11, b33, b44, b14 = args[4:8]
#    c44, c14 = args[8:10]
#    d11, d33, d44, d14 = args[10:14]

# 7863
args = [
    4.24392314e-01,
    7.08578306e-01,
    1.45793555e00,
    -1.68461729e-01,
    -6.09522821e-01,
    -1.12003432e-02,
    6.36812270e-02,
    6.13831431e-01,
    9.00394247e-02,
    9.54158204e-02,
    8.25444467e-02,
    -1.40061003e-01,
    -6.12857181e-01,
    6.60442257e-01,
    -1.45975698e-01,
    -1.96139630e-02,
    2.26315254e-01,
    1.41928737e-02,
    8.37321521e-02,
    -5.22903121e-03,
    3.01841019e-01,
    -8.33393895e-02,
    -7.10436331e-01,
    6.89974826e-03,
    -3.16804519e-03,
    3.01915087e-04,
    -5.93263316e-01,
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
