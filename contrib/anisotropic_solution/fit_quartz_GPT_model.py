import numpy as np
import matplotlib.pyplot as plt
import burnman
from scipy.optimize import minimize, root, basinhopping
from scipy import interpolate
from data_tables import data, colours, labels
from copy import deepcopy
from utils.transition_temperatures import (
    transition_temperature,
    transition_temperature_Angel,
    transition_temperature_MM1980,
)


"""
Assume quadratic V(P,T) and S(P,T) with respect to Q

"Standard state" quartz has mean tilt,
finite size domains
V_11

The effect of tilt
(-ve V, -ve S)
V_11 - V_01

The effect of small domains
(+ve V, +ve S)
V_11 - V_10
"""


q_SLB = burnman.minerals.SLB_2011.quartz()
q_HP = burnman.minerals.HP_2011_ds62.q()

# Tweak Antao data
data["Antao"]["V"] = data["Carpenter_neutron"]["V"][0] + 1.03 * (
    data["Antao"]["V"] - data["Carpenter_neutron"]["V"][0]
)

min_misfit = [np.inf]
min_args = [0.0]


def excess_gibbs(Q, pressure, temperature, properties, params):
    """
    alpha is the instance with Q = [1., 1.]
    beta is the instance with Q = [0., 0.]

    gibbs is the gibbs energy

    Q_0 = static tilt (decreasing with temperature, 0 in beta field)
    Q_1 = tilt domain surface area, volume normalised (increasing with temperature, 0 in alpha field)
    """

    alpha, beta, p_10, c_10, c_01, n_01 = params

    p_01 = 1.0 - p_10

    alpha.set_state(pressure, temperature)
    beta.set_state(pressure, temperature)

    delta_gibbs = alpha.gibbs - beta.gibbs
    delta_S = alpha.S - beta.S
    delta_V = alpha.V - beta.V

    # Static tilt (alpha only)
    cs = [[2, 0, p_10 * delta_gibbs, p_10 * delta_S, p_10 * delta_V]]
    cs.append([6, 0, p_10 * c_10, 0.0, 0.0])

    # Tilt domains (beta only)
    cs.append([0, 2, p_01 * delta_gibbs, p_01 * delta_S, p_01 * delta_V])
    cs.append([0, n_01, p_01 * c_01, 0.0, 0.0])
    Gex = np.sum(
        np.array(
            [
                np.power(Q[0], c[0]) * np.power(np.abs(Q[1]), c[1]) * np.array(c[2:])
                for c in cs
            ]
        ),
        axis=0,
    )

    properties[:] = Gex[1:] + np.array([beta.S, beta.V])

    return Gex[0]


def equilibrium_Q(pressure, temperature, params):
    alpha, beta, p_10, c_10, c_01, n_01 = params

    p_01 = 1.0 - p_10

    alpha.set_state(pressure, temperature)
    beta.set_state(pressure, temperature)

    delta_gibbs = alpha.gibbs - beta.gibbs
    delta_S = alpha.S - beta.S
    delta_V = alpha.V - beta.V

    # Static tilt (alpha only)
    cs = [[2, 0, p_10 * delta_gibbs, p_10 * delta_S, p_10 * delta_V]]
    cs.append([6, 0, p_10 * c_10, 0.0, 0.0])

    # Tilt domains (beta only)
    cs.append([0, 2, p_01 * delta_gibbs, p_01 * delta_S, p_01 * delta_V])
    cs.append([0, n_01, p_01 * c_01, 0.0, 0.0])

    Q = np.zeros(2)
    if delta_gibbs < 0.0:
        # Q_0 > 0, Q_1 = 0
        Q[0] = np.power(np.abs(2.0 * delta_gibbs / (6.0 * c_10)), 1.0 / (4.0))
    else:
        Q[1] = np.power(np.abs(2.0 * delta_gibbs / (n_01 * c_01)), 1.0 / (n_01 - 2.0))
    return Q


def equilibrated_properties(pressure, temperature, params):
    properties = np.zeros(2)

    Q_eqm = equilibrium_Q(pressure, temperature, params)
    Gxs = excess_gibbs(Q_eqm, pressure, temperature, properties, params)
    return {"G": Gxs, "Q": np.abs(Q_eqm), "S": properties[0], "V": properties[1]}


def make_params(args_scale):

    args = np.array(args_scale) * 1.0
    alpha = burnman.minerals.SLB_2011.quartz()

    alpha.params["F_0"] = 0.0
    alpha.params["V_0"] = args[0] * 1.0e-7
    alpha.params["K_0"] = args[1] * 1.0e11
    alpha.params["Kprime_0"] = args[2]
    alpha.params["grueneisen_0"] = args[3]
    alpha.params["Debye_0"] = args[4] * 1.0e3
    alpha.params["q_0"] = args[5]

    alpha.property_modifiers = []

    beta = deepcopy(alpha)
    beta.params["V_0"] = args[6] * 1.0e-7
    beta.params["K_0"] = args[7] * 1.0e11
    beta.params["grueneisen_0"] = args[8]
    beta.params["Debye_0"] = args[9] * 1.0e3

    alpha.property_modifiers = [
        [
            "einstein",
            {"Theta_0": 400.0 + args[13] * 1.0e3, "Cv_inf": 40.0 + args[16] * 1.0e2},
        ],
        [
            "einstein",
            {"Theta_0": 800.0 + args[14] * 1.0e3, "Cv_inf": -70.0 + args[17] * 1.0e2},
        ],
        [
            "einstein",
            {
                "Theta_0": 1400.0 + args[15] * 1.0e3,
                "Cv_inf": 30.0 - args[16] * 1.0e2 - args[17] * 1.0e2,
            },
        ],
    ]
    beta.property_modifiers = [
        [
            "einstein_delta",
            {"Theta_0": args[10] * 1.0e3, "S_inf": args[11]},
        ],  # This is like an order-disorder term. S_inf is fitted
        [
            "einstein_delta",
            {"Theta_0": args[18] * 1.0e3, "S_inf": args[19]},
        ],  # This is like an order-disorder term. S_inf is fitted
        [
            "einstein",
            {"Theta_0": 400.0 + args[13] * 1.0e3, "Cv_inf": 40.0 + args[16] * 1.0e2},
        ],
        [
            "einstein",
            {"Theta_0": 800.0 + args[14] * 1.0e3, "Cv_inf": -70.0 + args[17] * 1.0e2},
        ],
        [
            "einstein",
            {
                "Theta_0": 1400.0 + args[15] * 1.0e3,
                "Cv_inf": 30.0 - args[16] * 1.0e2 - args[17] * 1.0e2,
            },
        ],
    ]

    p_10 = args[12]

    F_0_orig = alpha.params["F_0"]

    def delta_T(args_dT):
        alpha.params["F_0"] = F_0_orig + args_dT[0]
        return [
            transition_temperature(P, alpha, beta) - transition_temperature_MM1980(P)
            for P in [0.0]
        ]

    # Fit to the transition
    root(delta_T, [-1000.0])

    alpha.set_state(1.0e5, 298.15)
    beta.set_state(1.0e5, 298.15)
    c_10 = -(alpha.gibbs - beta.gibbs) / 3.0

    alpha.set_state(1.0e5, 1246.00)
    beta.set_state(1.0e5, 1246.00)
    c_01 = -(alpha.gibbs - beta.gibbs) / 5.0
    n_01 = 6.0  # args[18] * 10.0

    return [alpha, beta, p_10, c_10, c_01, n_01]


def printe(s, a, b, m):
    if False:
        print(s, a, b, m)


def model_misfit(args):

    # print(repr(args))
    try:
        params = make_params(args)
        alpha, beta = params[:2]
        misfit = 0.0

        # equilibrated_properties(pressure, temperature, params)

        # Start the misfit calculation here

        if False:
            # Volume at 1 bar (Antao)
            Tobs = data["Antao"]["T_K"][::3].to_numpy()
            mask = Tobs > 846.0
            Tobs = data["Antao"]["T_K"][::3].to_numpy()[mask]
            Pobs = 1.0e5 * np.ones_like(Tobs)
            Vobs = data["Antao"]["V"][::3].to_numpy()[mask]
            Verr = data["Antao"]["V"][::3].to_numpy()[mask] * 0.0005
            nobs = len(Pobs)

            for i in range(nobs):
                p = equilibrated_properties(Pobs[i], Tobs[i], params)
                misfit += np.power((p["V"] - Vobs[i]) / Verr[i], 2.0)
                printe("1", p["V"], Vobs[i], misfit)

        # Volume at 1 bar (Carpenter neutron)
        # i = [0, 3, 6, 9]
        Tobs = data["Carpenter_neutron"]["T_K"].to_numpy()
        Pobs = 1.0e5 * np.ones_like(Tobs)
        Vobs = data["Carpenter_neutron"]["V"].to_numpy()
        Verr = data["Carpenter_neutron"]["V"].to_numpy() * 0.0002
        nobs = len(Pobs)

        for i in range(nobs):
            p = equilibrated_properties(Pobs[i], Tobs[i], params)
            misfit += np.power((p["V"] - Vobs[i]) / Verr[i], 2.0)
            printe("2", p["V"], Vobs[i], misfit)

        # Volume at 1 bar (Carpenter XRD)
        Tobs = data["Carpenter_XRD"]["T_K"].to_numpy()
        Pobs = 1.0e5 * np.ones_like(Tobs)
        Vobs = data["Carpenter_XRD"]["V"].to_numpy()
        Verr = data["Carpenter_XRD"]["V"].to_numpy() * 0.0005
        nobs = len(Pobs)

        for i in range(nobs):
            p = equilibrated_properties(Pobs[i], Tobs[i], params)
            misfit += np.power((p["V"] - Vobs[i]) / Verr[i], 2.0)
            printe("2", p["V"], Vobs[i], misfit)

        # Volume at RT (Scheidl)
        Pobs = data["Scheidl"]["P_GPa"].to_numpy() * 1.0e9
        Tobs = 298.15 * np.ones_like(Pobs)
        Vobs = data["Scheidl"]["V"].to_numpy()
        Verr = data["Scheidl"]["unc_V"].to_numpy() * 10.0
        nobs = len(Pobs)

        for i in range(nobs):
            p = equilibrated_properties(Pobs[i], Tobs[i], params)
            misfit += np.power((p["V"] - Vobs[i]) / Verr[i], 2.0)
            printe("3", p["V"], Vobs[i], misfit)

        if True:
            # C_p at 1 bar (Richet)
            i = [4, 17, 20, 23, 26]
            Tobs = data["Richet"]["T_K"][i].to_numpy()
            Pobs = 1.0e5 * np.ones_like(Tobs)
            CPobs = data["Richet"]["CP"][i].to_numpy()
            CPerr = data["Richet"]["CP"][i].to_numpy() * 0.002
            nobs = len(Pobs)

            for i in range(nobs):
                p0 = equilibrated_properties(Pobs[i], Tobs[i] - 1.0, params)
                p1 = equilibrated_properties(Pobs[i], Tobs[i] + 1.0, params)
                Cp_model = Tobs[i] * (p1["S"] - p0["S"]) / 2.0
                misfit += np.power((Cp_model - CPobs[i]) / CPerr[i], 2.0)
                printe("4", Cp_model, CPobs[i], misfit)

            # C_p at 1 bar (Gronvold)
            Tobs = data["Gronvold"]["T_K"][::5].to_numpy()
            Pobs = 1.0e5 * np.ones_like(Tobs)
            CPobs = data["Gronvold"]["CP"][::5].to_numpy()
            CPerr = data["Gronvold"]["CP"][::5].to_numpy() * 0.002
            nobs = len(Pobs)

            for i in range(nobs):
                p0 = equilibrated_properties(Pobs[i], Tobs[i] - 1.0, params)
                p1 = equilibrated_properties(Pobs[i], Tobs[i] + 1.0, params)
                Cp_model = Tobs[i] * (p1["S"] - p0["S"]) / 2.0
                misfit += np.power((Cp_model - CPobs[i]) / CPerr[i], 2.0)
                printe("5", Cp_model, CPobs[i], misfit)

        if True:
            # K_S at 1 bar (Lakshtanov)
            Tobs = data["Lakshtanov"]["T_K"].to_numpy()
            mask = np.logical_or((Tobs < 750.0), (Tobs > 900.0))
            Tobs = Tobs[mask]
            Pobs = 1.0e5 * np.ones_like(Tobs)
            KSobs = data["Lakshtanov"]["K_S"].to_numpy()[mask]
            KSerr = 0.05e9 + data["Lakshtanov"]["K_S"].to_numpy()[mask] * 0.002
            nobs = len(Pobs)

            # KS = KT*(1. - alpha*alpha*KT*V*T/Cp)^-1
            # betaS = betaT*(1. + (dV/dT)^2 / ((dV/dP)*(dS/dT)))
            # betaS = -1/V*((dV/dP) + (dV/dT)^2 / ((dS/dT))
            # KS = -V/((dV/dP) + (dV/dT)^2 / ((dS/dT))

            for i in range(nobs):
                p = equilibrated_properties(Pobs[i], Tobs[i], params)
                pT0 = equilibrated_properties(Pobs[i], Tobs[i] - 1.0, params)
                pT1 = equilibrated_properties(Pobs[i], Tobs[i] + 1.0, params)
                pP0 = equilibrated_properties(Pobs[i] - 1.0e5, Tobs[i], params)
                pP1 = equilibrated_properties(Pobs[i] + 1.0e5, Tobs[i], params)

                V = p["V"]
                dVdP = (pP1["V"] - pP0["V"]) / 2.0e5
                dVdT = (pT1["V"] - pT0["V"]) / 2.0
                dSdT = (pT1["S"] - pT0["S"]) / 2.0
                KS_model = -V / (dVdP + dVdT * dVdT / dSdT)

                misfit += np.power((KS_model - KSobs[i]) / KSerr[i], 2.0)
                printe("6", KS_model, KSobs[i], misfit)

        for P in np.linspace(0.0, 2.0e9, 11):
            Tmodel = transition_temperature(P, alpha, beta)
            Tobs = transition_temperature_MM1980(P)
            Terr = 5.0
            misfit += np.power(
                (Tmodel - Tobs) / Terr,
                2.0,
            )
            printe("7", Tmodel, Tobs, misfit)
    except AssertionError:
        misfit = 1.0e8

    if misfit < min_misfit[0]:
        min_misfit[0] = misfit
        min_args[0] = args
        print(repr(args))
        print(misfit)
    return misfit


def model_misfit_2(args):
    args_full = np.array(
        [
            2.27834880e02,
            8.00175710e-01,
            5.30461449e00,
            -3.28342735e-02,
            1.06592403e00,
            1.23266155e00,
            2.37512306e02,
            7.59606671e-01,
            -2.65741400e-01,
            1.07740447e00,
            6.51009276e-01,
            3.82700751e00,
            1.12972550e00,
            -2.43967181e-03,
            2.00425092e-02,
            8.44850254e-02,
            6.82057310e-05,
            -6.14710600e-03,
            10.0,
        ]
    )
    args_full[13:] = args
    return model_misfit(args_full)


alpha_V_0 = 2.27796e-05
alpha_K_0 = 7.992e10
alpha_Kprime_0 = 5.287
alpha_grueneisen_0 = -5.70e-02
alpha_Debye_0 = 1066.22155
alpha_q_0 = 1.24113567
beta_V_0 = 2.37407924871e-05
beta_K_0 = 76558522000.0
beta_grueneisen_0 = -0.27061281000000004
beta_Debye_0 = 1077.12882
beta_pm_Theta_0 = 656.03989
beta_pm_S_inf = 3.8
p_10 = 1.13
n_01 = 10.0

args = [
    alpha_V_0 / 1.0e-7,
    alpha_K_0 / 1.0e11,
    alpha_Kprime_0,
    alpha_grueneisen_0,
    alpha_Debye_0 / 1.0e3,
    alpha_q_0,
    beta_V_0 / 1.0e-7,
    beta_K_0 / 1.0e11,
    beta_grueneisen_0,
    beta_Debye_0 / 1.0e3,
    beta_pm_Theta_0 / 1.0e3,
    beta_pm_S_inf,
    p_10,
    n_01 / 10.0,
]

# 1287
args = [
    2.27818089e02,
    8.00692238e-01,
    5.31458857e00,
    -3.61584066e-02,
    1.06607481e00,
    1.23922499e00,
    2.37458465e02,
    7.61123107e-01,
    -2.58985446e-01,
    1.07734707e00,
    6.67025347e-01,
    3.83469933e00,
    1.13243437e00,
    6.67687256e-03,
    9.02008224e-03,
    1.33316200e-02,
    -8.41044737e-04,
    -3.75265932e-03,
]

# 935.48
args = [
    2.27616157e02,
    7.48032237e-01,
    7.01334832e00,
    -6.15759576e-02,
    1.06902139e00,
    1.09400170e01,
    2.37899663e02,
    7.22349659e-01,
    -2.37705826e-01,
    1.07625537e00,
    2.43517385e-01,
    2.00297001e00,
    1.07586176e00,
    1.10158911e-01,
    1.41137578e-03,
    7.09445943e-01,
    -6.33075145e-04,
    1.70130911e-01,
    8.73841765e-01,
    2.19210207e00,
]

params = make_params(args)
model_misfit(args)
alpha, beta, p_10, c_10, c_01, n_01 = params

if True:
    print(alpha.params)
    print(beta.params)
    print(alpha.property_modifiers)
    print(beta.property_modifiers)
    print(p_10, c_10, c_01, n_01)
    exit()
else:
    args_2 = args[13:]

    min_args[0] = args
    for i in range(10):
        print(i)
        if False:
            min_misfit = [model_misfit(args)]
            args = minimize(
                model_misfit,
                min_args[0],
                method="Nelder-Mead",
                options={"adaptive": False},
            ).x
        if False:
            # Adaptive tends to be slower but maybe more robust?
            args = minimize(
                model_misfit,
                min_args[0],
                method="Nelder-Mead",
                options={"adaptive": True},
            ).x

        if True:
            # COBYLA can be cycled with Nelder-Mead
            args = minimize(
                model_misfit, min_args[0], method="COBYLA", options={"rhobeg": 0.1}
            ).x


if False:
    plt.scatter(data["Scheidl"]["P_GPa"][::5], data["Scheidl"]["V"][::5], s=5)
    plt.errorbar(
        data["Scheidl"]["P_GPa"][::5],
        data["Scheidl"]["V"][::5],
        xerr=data["Scheidl"]["unc_P"][::5],
        yerr=data["Scheidl"]["unc_V"][::5] * 30.0,
        fmt="None",
    )
    plt.show()

    plt.scatter(data["Lakshtanov"]["T_K"][::2], data["Lakshtanov"]["K_S"][::2], s=5)
    plt.errorbar(
        data["Lakshtanov"]["T_K"][::2],
        data["Lakshtanov"]["K_S"][::2],
        yerr=1e9 + data["Lakshtanov"]["K_S"][::2] * 0.02,
        fmt="None",
    )
    plt.show()

    i = [4, 17, 20, 23, 26]
    plt.scatter(data["Richet"]["T_K"][i], data["Richet"]["CP"][i], s=5)

    plt.errorbar(
        data["Richet"]["T_K"][i],
        data["Richet"]["CP"][i],
        yerr=data["Richet"]["CP"][i] * 0.01,
        fmt="None",
    )

    plt.scatter(data["Gronvold"]["T_K"][::5], data["Gronvold"]["CP"][::5], s=5)

    plt.errorbar(
        data["Gronvold"]["T_K"][::5],
        data["Gronvold"]["CP"][::5],
        yerr=data["Gronvold"]["CP"][::5] * 0.01,
        fmt="None",
    )

    plt.show()

    plt.scatter(
        data["Carpenter_neutron"]["T_K"][::3], data["Carpenter_neutron"]["V"][::3], s=5
    )

    plt.errorbar(
        data["Carpenter_neutron"]["T_K"][::3],
        data["Carpenter_neutron"]["V"][::3],
        yerr=data["Carpenter_neutron"]["V"][::3] * 0.0005,
        fmt="None",
    )

    Tobs = data["Antao"]["T_K"][::3].to_numpy()
    mask = Tobs > 846.0
    Tobs = data["Antao"]["T_K"][::3].to_numpy()[mask]
    Pobs = 1.0e5 * np.ones_like(Tobs)
    Vobs = data["Antao"]["V"][::3].to_numpy()[mask]
    Verr = data["Antao"]["V"][::3].to_numpy()[mask] * 0.0005
    nobs = len(Pobs)

    plt.scatter(Tobs, Vobs, s=5)

    plt.errorbar(
        Tobs,
        Vobs,
        yerr=Verr,
        fmt="None",
    )

    plt.show()


pressures = np.linspace(1.0e5, 3.0e9, 11)
temperatures = np.array([transition_temperature(P, alpha, beta) for P in pressures])

plt.plot(pressures / 1.0e9, temperatures, label="Model")
plt.plot(
    pressures / 1.0e9,
    transition_temperature_MM1980(pressures),
    linestyle=":",
    label="MM1980",
)

plt.plot(
    pressures / 1.0e9,
    transition_temperature_Angel(pressures),
    linestyle="--",
    label="Angel",
)

plt.legend()
plt.show()


fig = plt.figure(figsize=(16, 8))
ax = [fig.add_subplot(2, 4, i) for i in range(1, 9)]

for temperatures in [
    np.linspace(1, 845.0, 101),
    np.linspace(845, 847.0, 21),
    np.linspace(847.0, 2000.0, 101),
]:
    Q_mins = np.zeros((len(temperatures), 2))
    Gs = np.zeros(len(temperatures))
    Ss = np.zeros((2, len(temperatures)))
    Vs = np.zeros((2, len(temperatures)))

    dP = 1.0e6
    for i, P in enumerate([1.0e3 - dP / 2.0, 1.0e3 + dP / 2.0]):

        pressures = np.ones_like(temperatures) * P
        prps = np.array([0.0, 0.0])

        for j, T in enumerate(temperatures):
            print(T)
            p = equilibrated_properties(P, T, params)
            Gs[i] = p["G"]
            Ss[i, j], Vs[i, j] = (p["S"], p["V"])
            Q_mins[i] = np.abs(p["Q"])

    V1_fn = interpolate.interp1d(
        Ss[1, :], Vs[1, :], kind="linear", fill_value="extrapolate"
    )
    Vav = (V1_fn(Ss[0, :]) + Vs[0, :]) / 2.0
    KS = -Vav * dP / (V1_fn(Ss[0, :]) - Vs[0, :])
    ax[6].plot(temperatures, KS, color="black")


ax[6].scatter(data["Lakshtanov"]["T_K"], data["Lakshtanov"]["K_S"], s=5)

temperatures = np.linspace(1.0, 2000.0, 151)
Q_mins = np.zeros((len(temperatures), 2))
Gs = np.zeros(len(temperatures))
Ss = np.zeros(len(temperatures))
Vs = np.zeros(len(temperatures))

for P in [1.0e5, 0.26e9, 1.0e9, 3.0e9]:

    pressures = np.ones_like(temperatures) * P
    prps = np.array([0.0, 0.0])

    for i, T in enumerate(temperatures):
        print(T)
        p = equilibrated_properties(P, T, params)
        Gs[i] = p["G"]
        Ss[i], Vs[i] = (p["S"], p["V"])
        Q_mins[i] = np.abs(p["Q"])

    ax[0].plot(temperatures, Q_mins[:, 0], label=f"$Q_1$ ({P/1.e9} GPa)")
    ax[0].plot(temperatures, Q_mins[:, 1], label=f"$Q_2$ ({P/1.e9} GPa)")
    # ax[0].plot(temperatures, Q_mins[:, 2], label="$Q_3$")

    Cps = temperatures * np.gradient(Ss, temperatures, edge_order=2)

    ax[4].plot(temperatures, Ss, label=f"S ({P/1.e9} GPa)")
    ax[1].plot(temperatures, Cps, label=f"Cp ({P/1.e9} GPa)")
    ax[2].plot(
        temperatures,
        Vs,
        label=f"V ({P/1.e9} GPa)",
    )

    V_SLB, S_SLB, Cp_SLB = q_SLB.evaluate(
        ["V", "S", "molar_heat_capacity_p"], pressures, temperatures
    )
    V_HP, S_HP, Cp_HP, KS_HP = q_HP.evaluate(
        ["V", "S", "molar_heat_capacity_p", "K_S"], pressures, temperatures
    )
    """
    ax[1].plot(
        temperatures,
        Cp_HP,
        linestyle="--",
        label=f"HP ({P/1.e9} GPa)",
    )
    """
    ax[4].plot(
        temperatures,
        S_HP,
        linestyle="--",
        label=f"HP ({P/1.e9} GPa)",
    )
    ax[6].plot(
        temperatures,
        KS_HP,
        linestyle="--",
        label=f"HP ({P/1.e9} GPa)",
    )
    ax[2].plot(
        temperatures,
        V_HP,
        linestyle="--",
        label=f"HP ({P/1.e9} GPa)",
    )

pressures = np.linspace(1.0e5, 20.0e9, 51)
Q_mins = np.zeros((len(pressures), 2))
Gs = np.zeros(len(pressures))
Ss = np.zeros(len(pressures))
Vs = np.zeros(len(pressures))

for T in [300.0]:

    temperatures = np.ones_like(pressures) * T
    prps = np.array([0.0, 0.0])

    for i, P in enumerate(pressures):
        print(P / 1.0e9)
        p = equilibrated_properties(P, T, params)
        Gs[i] = p["G"]
        Ss[i], Vs[i] = (p["S"], p["V"])
        Q_mins[i] = np.abs(p["Q"])

    ax[3].plot(
        pressures / 1.0e9,
        Vs,
        label="V",
    )
    V_HP, S_HP, Cp_HP = q_HP.evaluate(
        ["V", "S", "molar_heat_capacity_p"], pressures, temperatures
    )
    ax[3].plot(
        pressures / 1.0e9,
        V_HP,
        linestyle="--",
        label="HP",
    )

    ax[5].plot(Vs, Q_mins[:, 0])


def tilt_distorted(c_over_a, x, z):
    # c_over_a is the c/a ratio
    # x and z are the 6c positions (Oxygen atom positions (x,y,z))
    # of space group P3_121
    # typically x~0.41, z~0.22
    return np.degrees(
        np.arctan(2.0 * np.sqrt(3.0) / 9.0 * c_over_a * (6.0 * z - 1.0) / x)
    )


V_fn = interpolate.interp1d(
    data["Antao"]["T_K"][1:],
    data["Antao"]["V"][1:],
    kind="linear",
    fill_value="extrapolate",
)

f_tilt = 1.00 / data["Antao"]["tilt"][0] / 1.02
ax[5].scatter(
    data["Jorgensen"]["V"],
    data["Jorgensen"]["tilt"] * f_tilt,
    c=colours["Jorgensen"],
    label=labels["Jorgensen"],
    s=5,
)

ax[5].scatter(
    data["Ogata"]["V"],
    data["Ogata"]["tilt"] * f_tilt,
    c=colours["Ogata"],
    label=labels["Ogata"],
    s=5,
)

Hazen_tilt = tilt_distorted(
    data["Hazen"]["c"] / data["Hazen"]["a"], data["Hazen"]["x"], data["Hazen"]["z"]
)
ax[5].scatter(
    data["Hazen"]["V"],
    Hazen_tilt * f_tilt,
    c=colours["Hazen"],
    label=labels["Hazen"],
    s=5,
)


ax[0].scatter(data["Antao"]["T_K"], data["Antao"]["tilt"] * f_tilt, s=5)
ax[0].scatter(
    data["Axe"]["T_K"],
    np.power(data["Axe"]["T_over_I"] / data["Axe"]["T_K"] * 20.0, 0.25),
    s=5,
)

ax[1].scatter(data["Gronvold"]["T_K"], data["Gronvold"]["CP"], s=5)
ax[1].scatter(data["Richet"]["T_K"], data["Richet"]["CP"], s=5)

ax[2].scatter(data["Antao"]["T_K"], data["Antao"]["V"], s=5)

for P in [1, 2600]:
    idx = data["Raz"]["P_bar"].isin([P])
    ax[2].scatter(
        data["Raz"]["T_C"][idx] + 273.15,
        (1.0 + 0.01 * data["Raz"]["dV_V0_pct"][idx]) * data["Antao"]["V"][0],
        s=5,
    )

ax[2].scatter(
    data["Carpenter_neutron"]["T_K"],
    data["Carpenter_neutron"]["V"],
    c=colours["Carpenter_neutron"],
    label=labels["Carpenter_neutron"],
    s=5,
)

ax[2].scatter(
    data["Carpenter_XRD"]["T_K"],
    data["Carpenter_XRD"]["V"],
    c=colours["Carpenter_XRD"],
    label=labels["Carpenter_XRD"],
    s=5,
)

ax[3].scatter(data["Scheidl"]["P_GPa"], data["Scheidl"]["V"], s=5)

ax[0].legend()
ax[1].legend()
ax[1].set_ylim(0.0, 100.0)
ax[4].set_ylim(-1.0, 200.0)
ax[2].legend()
plt.show()
