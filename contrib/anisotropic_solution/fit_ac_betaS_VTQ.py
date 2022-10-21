import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_tables import data
from quartz_model import equilibrium_Q, qtz_a
from scipy.optimize import minimize, basinhopping
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from copy import deepcopy

# Compile data into one array
cmap = "turbo"

Ts_K = np.concatenate(
    (
        data["Lakshtanov"]["T_K"].to_numpy(),
        data["Wang"]["T_K"].to_numpy(),
    )
)

Ps_K = (
    np.concatenate(
        (
            data["Lakshtanov"]["P_GPa"].to_numpy(),
            data["Wang"]["P_GPa"].to_numpy(),
        )
    )
    * 1.0e9
)
betaS11_K = np.concatenate(
    (
        data["Lakshtanov"]["beta_S11"].to_numpy(),
        data["Wang"]["beta_S11"].to_numpy(),
    )
)
betaS33_K = np.concatenate(
    (
        data["Lakshtanov"]["beta_S33"].to_numpy(),
        data["Wang"]["beta_S33"].to_numpy(),
    )
)
V_K = np.concatenate(
    (
        data["Lakshtanov"]["V"].to_numpy(),
        data["Wang"]["V"].to_numpy(),
    )
)

Qs_K = np.zeros((len(Ts_K), 2))
Qs_K1 = np.zeros((len(Ts_K), 2))
Qs_K2 = np.zeros((len(Ts_K), 2))

Vs_K1 = np.zeros(len(Ts_K))
Vs_K2 = np.zeros(len(Ts_K))

Pths_K = np.zeros(len(Ts_K))
Pths_K1 = np.zeros(len(Ts_K))
Pths_K2 = np.zeros(len(Ts_K))

for i in range(len(Ts_K)):
    Qs_K[i] = equilibrium_Q(Ps_K[i], Ts_K[i])
    qtz_a.equilibrate(Ps_K[i], Ts_K[i])
    Pths_K[i] = qtz_a._Pth_mbr.dot(qtz_a.molar_fractions)

    dTdP = qtz_a.isentropic_thermal_gradient
    dP = 1.0e4
    P1 = Ps_K[i] - dP / 2.0
    P2 = Ps_K[i] + dP / 2.0

    T1 = Ts_K[i] - dTdP * dP / 2.0
    T2 = Ts_K[i] + dTdP * dP / 2.0

    Qs_K1[i] = equilibrium_Q(P1, T1)
    qtz_a.equilibrate(P1, T1)
    Vs_K1[i] = qtz_a.V
    Pths_K1[i] = qtz_a._Pth_mbr.dot(qtz_a.molar_fractions)

    Qs_K2[i] = equilibrium_Q(P2, T2)
    qtz_a.equilibrate(P2, T2)
    Vs_K2[i] = qtz_a.V
    Pths_K2[i] = qtz_a._Pth_mbr.dot(qtz_a.molar_fractions)


Q1sqrs_K1, Q2sqrs_K1 = (Qs_K1 * Qs_K1).T
Q1sqrs_K2, Q2sqrs_K2 = (Qs_K2 * Qs_K2).T
deps_over_df_K = (betaS11_K - betaS33_K) / (betaS11_K + betaS11_K + betaS33_K)

Ts = np.concatenate(
    (
        data["Raz"]["T_K"].to_numpy(),
        data["Scheidl"]["T_K"].to_numpy(),
        data["Carpenter_XRD"]["T_K"].to_numpy(),
        data["Carpenter_neutron"]["T_K"].to_numpy(),
    )
)
Ps = (
    np.concatenate(
        (
            data["Raz"]["P_GPa"].to_numpy(),
            data["Scheidl"]["P_GPa"].to_numpy(),
            data["Carpenter_XRD"]["P_GPa"].to_numpy(),
            data["Carpenter_neutron"]["P_GPa"].to_numpy(),
        )
    )
    * 1.0e9
)
a = np.concatenate(
    (
        data["Raz"]["a"].to_numpy(),
        data["Scheidl"]["a"].to_numpy(),
        data["Carpenter_XRD"]["a"].to_numpy(),
        data["Carpenter_neutron"]["a"].to_numpy(),
    )
)
c = np.concatenate(
    (
        data["Raz"]["c"].to_numpy(),
        data["Scheidl"]["c"].to_numpy(),
        data["Carpenter_XRD"]["c"].to_numpy(),
        data["Carpenter_neutron"]["c"].to_numpy(),
    )
)
V = np.concatenate(
    (
        data["Raz"]["V"].to_numpy(),
        data["Scheidl"]["V"].to_numpy(),
        data["Carpenter_XRD"]["V"].to_numpy(),
        data["Carpenter_neutron"]["V"].to_numpy(),
    )
)

Qs = np.zeros((len(Ts), 2))
Pths = np.zeros(len(Ts))

for i in range(len(Ts)):
    Qs[i] = equilibrium_Q(Ps[i], Ts[i])
    qtz_a.equilibrate(Ps[i], Ts[i])
    Pths[i] = qtz_a._Pth_mbr.dot(qtz_a.molar_fractions)
Q1sqrs, Q2sqrs = (Qs * Qs).T


def a_over_c(V, Pth, Q1sqr, Q2sqr, a1, a2, a3, a4, a5, a6, a7, a8, d1):

    Vrel = V / 2.2761615699999998e-05  # value from scalar model
    f = np.log(Vrel)
    lna = (
        a1
        + a2 * f
        + a3 * (np.exp(d1 * f) - 1.0)
        + a4 * Pth
        + a5 * (Q1sqr - 1.0)
        + a6 * (Q2sqr - 1.0)
        + a7 * (Q1sqr - 1.0) * f
        + a8 * (Q2sqr - 1.0) * f
    )

    c1 = 1 - 2.0 * a1
    c2 = 1 - 2.0 * a2
    c3 = -2.0 * a3
    c4 = -2.0 * a4
    c5 = -2.0 * a5
    c6 = -2.0 * a6
    c7 = -2.0 * a7
    c8 = -2.0 * a8

    lnc = (
        c1
        + c2 * f
        + c3 * (np.exp(d1 * f) - 1.0)
        + c4 * Pth
        + c5 * (Q1sqr - 1.0)
        + c6 * (Q2sqr - 1.0)
        + c7 * (Q1sqr - 1.0) * f
        + c8 * (Q2sqr - 1.0) * f
    )
    # a_over_c = np.log(expa) / np.log(expc)

    a = np.nan_to_num(np.exp(lna), 1000.0)
    c = np.nan_to_num(np.exp(lnc), 1.0)

    return a / c


a_obs = a
c_obs = c
a_over_c_obs = a / c


def misfit(args, K_factor, valargs):
    try:
        a_over_c_mod = a_over_c(V, Pths / 1.0e9, Q1sqrs, Q2sqrs, *args)
        misfit = np.sum(np.power(a_over_c_mod - a_over_c_obs, 2.0))

        ln_a_over_c_mod_K1 = np.log(
            a_over_c(Vs_K1, Pths_K1 / 1.0e9, Q1sqrs_K1, Q2sqrs_K1, *args)
        )
        ln_a_over_c_mod_K2 = np.log(
            a_over_c(Vs_K2, Pths_K2 / 1.0e9, Q1sqrs_K2, Q2sqrs_K2, *args)
        )

        # deps1 - deps3
        deps = ln_a_over_c_mod_K2 - ln_a_over_c_mod_K1
        df = np.log(Vs_K2) - np.log(Vs_K1)

        deps_over_df_mod = deps / df

        misfit += np.sum(np.power(deps_over_df_mod - deps_over_df_K, 2.0)) * K_factor
    except RuntimeWarning:
        misfit = 1.0e5

    if misfit < valargs[0]:
        valargs[0] = misfit
        print(misfit)
    return misfit


warnings.filterwarnings("error")
if True:
    sol = minimize(
        misfit,
        [
            0.33,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            3.0,
        ],
        args=(1.0e-10, [1.0e5]),
    )
    print("trying basin hopping")
    store = [sol.fun, deepcopy(sol.x)]
    sol = basinhopping(
        misfit,
        sol.x,
        minimizer_kwargs={
            "args": (1.0e-2, [1.0e5]),
            "method": "Nelder-Mead",
            "options": {"adaptive": True},
        },
    )
    print(f"solve: {sol.fun}")
    print(repr(sol.x))
    x = sol.x
else:
    x = [
        3.44552173e-01,
        -1.13315130e-01,
        1.07795628e-01,
        -3.91321772e-03,
        1.58623000e-03,
        -4.15582828e-04,
        1.09888347e00,
    ]

a_over_c_mod = a_over_c(V, Pths / 1.0e9, Q1sqrs, Q2sqrs, *x)


bench = np.array(
    [
        [2.276158527150651e-05, -1.5566693036817014e-05, 1, 1],
        [2.2e-05, 0.0, 1.0, 1.0],
        [2.2685112576222494e-05, 0.0, 0.99930396, 0.0],
        [2.338451197404028e-05, -101976948.06662083, 0.54971356, 0.0],
        [1.9207506108419904e-05, 0.0, 1.57630735, 0.0],
    ]
).T

warnings.filterwarnings("ignore")

print(
    a_over_c(bench[0], bench[1] / 1.0e9, bench[2] * bench[2], bench[3] * bench[3], *x)
)

ln_a_over_c_mod_K1 = np.log(a_over_c(Vs_K1, Pths_K1 / 1.0e9, Q1sqrs_K1, Q2sqrs_K1, *x))
ln_a_over_c_mod_K2 = np.log(a_over_c(Vs_K2, Pths_K2 / 1.0e9, Q1sqrs_K2, Q2sqrs_K2, *x))

# deps1 - deps3
deps = ln_a_over_c_mod_K2 - ln_a_over_c_mod_K1
df = np.log(Vs_K2) - np.log(Vs_K1)

deps_over_df_mod = deps / df


fig = plt.figure(figsize=(8, 3))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

mask = V_K > V[0]
ax[0].scatter(
    Ts_K[mask], deps_over_df_mod[mask], c=Pths_K[mask] / 1.0e9, s=60, cmap=cmap
)
ax[0].scatter(Ts_K[mask], deps_over_df_K[mask], s=20, color="black")
d = ax[0].scatter(
    Ts_K[mask], deps_over_df_K[mask], c=Pths_K[mask] / 1.0e9, s=10, cmap=cmap
)

divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(d, cax=cax)
cbar.set_label("$P_{{th}}$ (GPa)")

ax[0].set_xlabel("$T$ (K)")
ax[0].set_ylabel("$(\\beta_{S11} - \\beta_{S33})/\\beta_{SR}$")


mask = Ts_K < 301.0
ax[1].scatter(
    Ps_K[mask] / 1.0e9, deps_over_df_mod[mask], c=V_K[mask] * 1.0e6, s=60, cmap=cmap
)
ax[1].scatter(Ps_K[mask] / 1.0e9, deps_over_df_K[mask], s=20, color="black")
d = ax[1].scatter(
    Ps_K[mask] / 1.0e9, deps_over_df_K[mask], c=V_K[mask] * 1.0e6, s=10, cmap=cmap
)

divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(d, cax=cax)
cbar.set_label("$V$ (cm$^3$/mol)")


ax[1].set_xlabel("$P$ (GPa)")
ax[1].set_ylabel("$(\\beta_{S11} - \\beta_{S33})/\\beta_{SR}$")


fig.set_tight_layout(True)
fig.savefig("figures/relative_isentropic_compressibilities.pdf")

plt.show()

# 0.9120037807609471
# 0.9127638400630655
# 0.9090477876401244
# 0.9128135222023198
# 0.8849093116003766

fig = plt.figure(figsize=(12, 6))
ax = [fig.add_subplot(2, 3, i) for i in range(1, 7)]


mask = V > V[0]

ax[0].scatter(np.log(V / V[0]), a_over_c_mod, c=Pths / 1.0e9, s=60, cmap=cmap)
ax[0].scatter(np.log(V / V[0]), a / c, s=20, color="black")
d = ax[0].scatter(np.log(V / V[0]), a / c, c=Pths / 1.0e9, s=10, cmap=cmap)


divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(d, cax=cax)
cbar.set_label("$P_{{th}}$ (GPa)")


ax[0].set_xlabel("$\\ln (V)$")
ax[0].set_ylabel("$a/c$")

ax[3].scatter(
    np.log(V / V[0])[mask], a_over_c_mod[mask], c=Pths[mask] / 1.0e9, s=60, cmap=cmap
)
ax[3].scatter(np.log(V / V[0])[mask], (a / c)[mask], s=20, color="black")
d = ax[3].scatter(
    np.log(V / V[0])[mask], (a / c)[mask], c=Pths[mask] / 1.0e9, s=10, cmap=cmap
)

divider = make_axes_locatable(ax[3])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(d, cax=cax)
cbar.set_label("$P_{{th}}$ (GPa)")

ax[3].set_xlabel("$\\ln (V)$")
ax[3].set_ylabel("$a/c$")


vmin = np.min((a / c))
vmax = np.max((a / c))

ax[1].scatter(
    np.log(V / V[0]),
    Pths / 1.0e9,
    c=a_over_c_mod,
    cmap=cmap,
    s=60,
    vmin=vmin,
    vmax=vmax,
)

ax[1].scatter(np.log(V / V[0]), Pths / 1.0e9, color="black", s=20)

d = ax[1].scatter(
    np.log(V / V[0]), Pths / 1.0e9, c=(a / c), cmap=cmap, s=10, vmin=vmin, vmax=vmax
)

divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(d, cax=cax)
cbar.set_label("$a/c$")

ax[1].set_xlabel("$\\ln (V)$")
ax[1].set_ylabel("$P_{{th}}$ (GPa)")


vmin = np.min((a / c)[mask])
vmax = np.max((a / c)[mask])

ax[4].scatter(
    np.log(V[mask] / V[0]),
    Pths[mask] / 1.0e9,
    c=a_over_c_mod[mask],
    cmap=cmap,
    s=60,
    vmin=vmin,
    vmax=vmax,
)

ax[4].scatter(np.log(V[mask] / V[0]), Pths[mask] / 1.0e9, color="black", s=20)

d = ax[4].scatter(
    np.log(V / V[0])[mask], Pths[mask] / 1.0e9, c=(a / c)[mask], s=10, cmap=cmap
)


divider = make_axes_locatable(ax[4])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(d, cax=cax)
cbar.set_label("$a/c$")

ax[4].set_xlabel("$\\ln (V)$")
ax[4].set_ylabel("$P_{{th}}$ (GPa)")


vmin = np.min((a / c))
vmax = np.max((a / c))
Q_active = np.maximum(Qs[:, 0], Qs[:, 1])

mask = Qs[:, 0] > 1.0e-5
ax[2].scatter(
    np.log(V / V[0]),
    Q_active,
    c=a_over_c_mod,
    vmin=vmin,
    vmax=vmax,
    s=60.0,
    cmap=cmap,
)

ax[2].scatter(
    np.log(V / V[0]),
    Q_active,
    color="black",
    s=20.0,
)

d = ax[2].scatter(
    np.log(V / V[0]),
    Q_active,
    c=(a / c),
    vmin=vmin,
    vmax=vmax,
    s=10.0,
    cmap=cmap,
)

divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(d, cax=cax)
cbar.set_label("$a/c$")

ax[2].set_xlabel("$\\ln (V)$")
ax[2].set_ylabel("$Q$")


if False:
    mask = Qs[:, 1] > 1.0e-5
    vmin, vmax = [0.9153, 0.9157]

    ax[5].scatter(
        np.log(V[mask] / V[0]),
        Qs[:, 1][mask],
        c=a_over_c_mod[mask],
        vmin=vmin,
        vmax=vmax,
        s=60,
        cmap=cmap,
    )

    ax[5].scatter(
        np.log(V[mask] / V[0]),
        Qs[:, 1][mask],
        color="black",
        s=20,
    )

    ax[5].scatter(
        np.log(V[mask] / V[0]),
        Qs[:, 1][mask],
        c=(a / c)[mask],
        vmin=vmin,
        vmax=vmax,
        s=10,
        cmap=cmap,
    )

    ax[5].set_xlabel("$\\ln (V)$")
    ax[5].set_ylabel("Q1")
else:
    mask = V > V[0]
    vmin = np.min((a / c)[mask])
    vmax = np.max((a / c)[mask])

    ax[5].scatter(
        np.log(V[mask] / V[0]),
        Q_active[mask],
        c=a_over_c_mod[mask],
        vmin=vmin,
        vmax=vmax,
        s=60,
        cmap=cmap,
    )

    ax[5].scatter(
        np.log(V[mask] / V[0]),
        Q_active[mask],
        color="black",
        s=20,
    )

    d = ax[5].scatter(
        np.log(V[mask] / V[0]),
        Q_active[mask],
        c=(a / c)[mask],
        vmin=vmin,
        vmax=vmax,
        s=10,
        cmap=cmap,
    )

    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(d, cax=cax)
    cbar.set_label("$a/c$")

    ax[5].set_xlabel("$\\ln (V)$")
    ax[5].set_ylabel("$Q$")

fig.set_tight_layout(True)
fig.savefig("figures/a_over_c.pdf")
plt.show()

mask = Ps < 4.0e8
vmin = np.min((a / c)[mask])
vmax = np.max((a / c)[mask])

plt.scatter(
    Ts[mask],
    Ps[mask] / 1.0e9,
    c=a_over_c_mod[mask],
    vmin=vmin,
    vmax=vmax,
    s=60,
    cmap=cmap,
)

plt.scatter(
    Ts[mask],
    Ps[mask] / 1.0e9,
    color="black",
    s=20,
)

d = plt.scatter(
    Ts[mask],
    Ps[mask] / 1.0e9,
    c=(a / c)[mask],
    vmin=vmin,
    vmax=vmax,
    s=10,
    cmap=cmap,
)

plt.show()
