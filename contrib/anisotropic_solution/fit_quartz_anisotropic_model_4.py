import numpy as np
from scipy.optimize import minimize
from data_tables import data
from quartz_model import make_anisotropic_model_4

min_misfit = [np.inf]
min_args = [0.0]


def printe(s, a, b, m):
    if False:
        print(s, a, b, m)


CNTobs = data["Carpenter_neutron"]["T_K"].to_numpy()[::1]
CNPobs = 1.0e5 * np.ones_like(CNTobs)
CNac = (
    data["Carpenter_neutron"]["a"].to_numpy()[::1]
    / data["Carpenter_neutron"]["c"].to_numpy()[::1]
)
CNacerr = np.ones_like(CNac) * 1.0e-4

CXTobs = data["Carpenter_XRD"]["T_K"].to_numpy()[::1]
CXPobs = 1.0e5 * np.ones_like(CNTobs)
CXac = (
    data["Carpenter_XRD"]["a"].to_numpy()[::1]
    / data["Carpenter_XRD"]["c"].to_numpy()[::1]
)
CXacerr = np.ones_like(CXac) * 1.0e-4

RTobs = data["Raz"]["T_K"].to_numpy()[::2]
RPobs = data["Raz"]["P_GPa"].to_numpy()[::2] * 1.0e9
Rac = data["Raz"]["a"].to_numpy()[::2] / data["Raz"]["c"].to_numpy()[::2]
Racerr = np.ones_like(Rac) * 1.0e-4


SPobs = data["Scheidl"]["P_GPa"].to_numpy()[::3] * 1.0e9
STobs = 300.0 * np.ones_like(SPobs)
Sac = data["Scheidl"]["a"].to_numpy()[::3] / data["Scheidl"]["c"].to_numpy()[::3]
Sacerr = np.ones_like(Sac) * 5.0e-4

LTobs = data["Lakshtanov"]["T_K"].to_numpy()[::1]
LPobs = 1.0e5 * np.ones_like(LTobs)

LCobs = {}
LCerr = {}
for (j, k) in [(1, 1), (3, 3), (4, 4), (1, 2), (1, 3), (1, 4)]:
    LCobs[f"{j}{k}"] = data["Lakshtanov"][f"C{j}{k}"].to_numpy()[::1]
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
        qtz_a = make_anisotropic_model_4(args)
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
            printe("3", acmod, acobs[i], misfit)

        # a/c (Raz)
        Pobs = RPobs
        Tobs = RTobs
        acobs = Rac
        acerr = Racerr
        nobs = len(RPobs)

        for i in range(nobs):
            qtz_a.equilibrate(Pobs[i], Tobs[i])
            acmod = qtz_a.cell_parameters[0] / qtz_a.cell_parameters[2]
            misfit += np.power((acmod - acobs[i]) / acerr[i], 2.0)
            printe("4", acmod, acobs[i], misfit)

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
                printe("5", Cmod[j - 1, k - 1], Cobs[f"{j}{k}"][i], misfit)

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
                printe("6", Cmod[j - 1, k - 1], Cobs[f"{j}{k}"][i], misfit)

    except AssertionError:
        misfit = 1.0e8

    if misfit < min_misfit[0]:
        min_misfit[0] = misfit
        min_args[0] = args
        print()
        print(repr(args))
        print(misfit)
    else:
        print(".", end="")
    return misfit


# 7535
args = [
    2.95213230e-01,
    6.99471420e-01,
    1.44946936e00,
    -1.57155755e-01,
    -8.77602188e-01,
    -2.31876622e-02,
    -5.19997394e-02,
    8.83115841e-01,
    1.97784949e-02,
    6.52290602e-02,
    6.04301637e-02,
    -1.36589908e-01,
    -6.32042423e-01,
    6.41027325e-01,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    3.01789661e-01,
    -4.25055139e-02,
    -7.13029446e-01,
    6.34294615e-03,
    -2.76454411e-03,
    2.24248663e-04,
    -5.44019331e-01,
    0.0,
    0.0,
]

qtz_a = make_anisotropic_model_4(args)
model_misfit(args)


if False:
    print(qtz_a)
else:

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

        if True:
            print("COBYLA")
            # COBYLA can be cycled with Nelder-Mead
            args = minimize(
                model_misfit, min_args[0], method="COBYLA", options={"rhobeg": 0.1}
            ).x
