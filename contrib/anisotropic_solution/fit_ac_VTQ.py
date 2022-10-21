import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_tables import data
from quartz_model import equilibrium_Q, qtz_a
from scipy.optimize import minimize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Compile data into one array
cmap = "turbo"


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


def a_over_c(V, Pth, Q1sqr, Q2sqr, a1, a2, a3, a4, a5, a6, d1):

    Vrel = V / 2.2761615699999998e-05  # value from scalar model
    f = np.log(Vrel)
    lna = (
        a1
        + a2 * f
        + a3 * (np.exp(d1 * f) - 1.0)
        + a4 * Pth
        + a5 * (Q1sqr - 1.0)
        + a6 * (Q2sqr - 1.0)
    )

    c1 = 1 - 2.0 * a1
    c2 = 1 - 2.0 * a2
    c3 = -2.0 * a3
    c4 = -2.0 * a4
    c5 = -2.0 * a5
    c6 = -2.0 * a6

    lnc = (
        c1
        + c2 * f
        + c3 * (np.exp(d1 * f) - 1.0)
        + c4 * Pth
        + c5 * (Q1sqr - 1.0)
        + c6 * (Q2sqr - 1.0)
    )
    # a_over_c = np.log(expa) / np.log(expc)

    a = np.nan_to_num(np.exp(lna), 1000.0)
    c = np.nan_to_num(np.exp(lnc), 1.0)

    return a / c


a_obs = a
c_obs = c
a_over_c_obs = a / c


def misfit(args):
    a_over_c_mod = a_over_c(V, Pths / 1.0e9, Q1sqrs, Q2sqrs, *args)
    misfit = np.sum(np.power(a_over_c_mod - a_over_c_obs, 2.0))
    return misfit


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
            -10.0,
        ],
    )
    for i in range(30):
        sol = minimize(
            misfit,
            sol.x,
            method="Nelder-Mead",
            options={"adaptive": True},
        )
        sol = minimize(misfit, sol.x, method="SLSQP")
        print(f"solve {i}: {sol.fun}")
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

print(
    a_over_c(bench[0], bench[1] / 1.0e9, bench[2] * bench[2], bench[3] * bench[3], *x)
)

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
