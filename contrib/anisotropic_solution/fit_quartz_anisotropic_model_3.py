import numpy as np
from scipy.optimize import minimize
from data_tables import data
from quartz_model import make_anisotropic_model_3

min_misfit = [np.inf]
min_args = [0.0]


def printe(s, a, b, m):
    if False:
        print(s, a, b, m)


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
        qtz_a = make_anisotropic_model_3(args)
        misfit = 0.0

        # equilibrated_properties(pressure, temperature, params)

        # Start the misfit calculation here

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


# 10976
args =[ 0.6478425 ,  0.65696392,  1.43534873, -0.17552329, -0.19389298,
       -0.06631256,  0.15772346,  0.15235251,  0.20120924,  0.1201691 ,
        0.07339388, -0.19922402, -0.60847529,  0.48797835,  0.22759003,
       -0.04301525,  0.21753109,  0.02321081, -0.07238275, -0.00298102]
qtz_a = make_anisotropic_model_3(args)
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
