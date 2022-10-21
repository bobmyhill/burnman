import numpy as np
from scipy.optimize import minimize
from data_tables import data
from quartz_model import make_anisotropic_model

min_misfit = [np.inf]
min_args = [0.0]


def printe(s, a, b, m):
    if False:
        print(s, a, b, m)


CNTobs = data["Carpenter_neutron"]["T_K"].to_numpy()[::5]
CNPobs = 1.0e5 * np.ones_like(CNTobs)
CNac = (
    data["Carpenter_neutron"]["a"].to_numpy()[::5]
    / data["Carpenter_neutron"]["c"].to_numpy()[::5]
)
CNacerr = np.ones_like(CNac) * 1.0e-4

CXTobs = data["Carpenter_XRD"]["T_K"].to_numpy()[::5]
CXPobs = 1.0e5 * np.ones_like(CNTobs)
CXac = (
    data["Carpenter_XRD"]["a"].to_numpy()[::5]
    / data["Carpenter_XRD"]["c"].to_numpy()[::5]
)
CXacerr = np.ones_like(CXac) * 1.0e-4

SPobs = data["Scheidl"]["P_GPa"].to_numpy()[::5] * 1.0e9
STobs = 300.0 * np.ones_like(SPobs)
Sac = data["Scheidl"]["a"].to_numpy()[::5] / data["Scheidl"]["c"].to_numpy()[::5]
Sacerr = np.ones_like(Sac) * 5.0e-4

LTobs = data["Lakshtanov"]["T_K"].to_numpy()[::5]
LPobs = 1.0e5 * np.ones_like(LTobs)

LCobs = {}
LCerr = {}
for (j, k) in [(1, 1), (3, 3), (4, 4), (1, 2), (1, 3), (1, 4)]:
    LCobs[f"{j}{k}"] = data["Lakshtanov"][f"C{j}{k}"].to_numpy()[::5]
    LCerr[f"{j}{k}"] = np.ones_like(LCobs[f"{j}{k}"]) * 1.0

# Only 5 data points in Wang
WPobs = data["Wang"]["P_GPa"].to_numpy() * 1.0e9
WTobs = 300.0 * np.ones_like(WPobs)

WCobs = {}
WCerr = {}
for (j, k) in [(1, 1), (3, 3), (4, 4), (1, 2), (1, 3), (1, 4)]:
    WCobs[f"{j}{k}"] = data["Wang"][f"C{j}{k}"].to_numpy()
    WCerr[f"{j}{k}"] = np.ones_like(WCobs[f"{j}{k}"]) * (1.0 + 0.2e-9 * WPobs)


def model_misfit(args):

    # print(repr(args))
    try:
        qtz_a = make_anisotropic_model(args)
        misfit = 0.0

        # equilibrated_properties(pressure, temperature, params)

        # Start the misfit calculation here

        # a/c at 1 bar (Carpenter neutron)
        Tobs = CNTobs
        Pobs = CNPobs
        acobs = CNac
        acerr = CNacerr
        nobs = len(Pobs)

        for i in range(nobs):
            qtz_a.equilibrate(Pobs[i], Tobs[i])
            acmod = qtz_a.cell_parameters[0] / qtz_a.cell_parameters[2]
            misfit += np.power((acmod - acobs[i]) / acerr[i], 2.0)
            printe("1", acmod, acobs[i], misfit)

        # a/c at 1 bar (Carpenter XRD)
        Tobs = CXTobs
        Pobs = CXPobs
        acobs = CXac
        acerr = CXacerr
        nobs = len(Pobs)

        for i in range(nobs):
            qtz_a.equilibrate(Pobs[i], Tobs[i])
            acmod = qtz_a.cell_parameters[0] / qtz_a.cell_parameters[2]
            misfit += np.power((acmod - acobs[i]) / acerr[i], 2.0)
            printe("2", acmod, acobs[i], misfit)

        # a/c at RT (Scheidl)
        Pobs = SPobs
        Tobs = STobs
        acobs = Sac
        acerr = Sacerr
        nobs = len(Pobs)

        for i in range(nobs):
            qtz_a.equilibrate(Pobs[i], Tobs[i])
            acmod = qtz_a.cell_parameters[0] / qtz_a.cell_parameters[2]
            misfit += np.power((acmod - acobs[i]) / acerr[i], 2.0)
            printe("2", acmod, acobs[i], misfit)

        # CSijs at 1 bar (Lakshtanov)
        Tobs = LTobs
        Pobs = LPobs
        nobs = len(Pobs)

        Cobs = LCobs
        Cerr = LCerr

        for i in range(nobs):
            qtz_a.equilibrate(Pobs[i], Tobs[i])
            Cmod = qtz_a.isentropic_stiffness_tensor / 1.0e9
            for (j, k) in [(1, 1), (3, 3), (4, 4), (1, 2), (1, 3), (1, 4)]:
                misfit += np.power(
                    (Cmod[j - 1, k - 1] - Cobs[f"{j}{k}"][i]) / Cerr[f"{j}{k}"][i], 2.0
                )
                printe("3", Cmod[j - 1, k - 1], Cobs[f"{j}{k}"][i], misfit)

        # CSijs at RT (Wang)
        Pobs = WPobs
        Tobs = WTobs
        nobs = len(Pobs)

        Cobs = WCobs
        Cerr = WCerr

        for i in range(nobs):
            qtz_a.equilibrate(Pobs[i], Tobs[i])
            Cmod = qtz_a.isentropic_stiffness_tensor / 1.0e9
            for (j, k) in [(1, 1), (3, 3), (4, 4), (1, 2), (1, 3), (1, 4)]:
                misfit += np.power(
                    (Cmod[j - 1, k - 1] - Cobs[f"{j}{k}"][i]) / Cerr[f"{j}{k}"][i], 2.0
                )
                printe("4", Cmod[j - 1, k - 1], Cobs[f"{j}{k}"][i], misfit)

    except AssertionError:
        misfit = 1.0e8

    if misfit < min_misfit[0]:
        min_misfit[0] = misfit
        min_args[0] = args
        print(repr(args))
        print(misfit)
    return misfit


args = [
    4.9137 / 5.4047,
    4.79501845e-01,
    -6.91663267e-02,
    -1.64139386e-01,
    3.64442730e-01,
    7.31250186e-01,
    0.0,
    0.0,
]


# 4620
args = [
    1.00999997e00,
    9.09178765e-01,
    -1.71455900e-01,
    2.42203080e00,
    8.02960854e-01,
    1.28647221e00,
    8.64154226e-04,
    4.50238945e-04,
    -1.07952119e-01,
    2.04573762e-02,
    -2.76997617e00,
    -3.32321275e-02,
    5.09027356e-01,
]

qtz_a = make_anisotropic_model(args)
model_misfit(args)


if False:
    print(qtz_a)
else:
    args_2 = args[13:]

    min_args[0] = args
    for i in range(10):
        print(i)
        if False:
            print("NM")
            min_misfit = [model_misfit(args)]
            args = minimize(
                model_misfit,
                min_args[0],
                method="Nelder-Mead",
                options={"adaptive": False},
            ).x
        if True:
            print("NM-adaptive")
            # Adaptive tends to be slower but maybe more robust?
            args = minimize(
                model_misfit,
                min_args[0],
                method="Nelder-Mead",
                options={"adaptive": True},
            ).x

        if False:
            print("COBYLA")
            # COBYLA can be cycled with Nelder-Mead
            args = minimize(
                model_misfit, min_args[0], method="COBYLA", options={"rhobeg": 0.1}
            ).x
