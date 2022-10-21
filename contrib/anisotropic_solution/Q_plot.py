import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from scipy.integrate import cumtrapz

Antao = pd.read_csv(
    "data/Antao_2016_quartz_structure_1bar.dat", delim_whitespace=True, comment="#"
)

Wells = pd.read_csv("data/Wells_2002_disorder.dat", delim_whitespace=True, comment="#")


tilt_fn = interpolate.interp1d(
    Antao["T_K"][1:], Antao["tilt"][1:], kind="linear", fill_value="extrapolate"
)

V_fn = interpolate.interp1d(
    Antao["T_K"][1:], Antao["V"][1:], kind="linear", fill_value="extrapolate"
)


Ts = np.linspace(0.0, 1500.0, 1500)
tilts = tilt_fn(Ts)

dtiltdT = tilts[301] - tilts[300]

delta_tilts = cumtrapz(np.linspace(0.0, dtiltdT, 301), initial=0)
tilts[:301] = delta_tilts - delta_tilts[-1] + tilts[300]

tilt_fn = interpolate.interp1d(Ts, tilts, kind="linear")
T_fn = interpolate.interp1d(tilts, Ts, kind="linear", fill_value="extrapolate")


misfit_fn = interpolate.interp1d(
    Wells["T_K"],
    Wells["misfit_Asqr_per_bond"],
    kind="linear",
    fill_value="extrapolate",
)

fig = plt.figure()
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

Q = (np.sqrt(Wells["misfit_Asqr_per_bond"])) / np.cbrt(V_fn(Wells["T_K"])) * 3.0


t = ax[0].scatter(
    tilt_fn(Wells["T_K"]),
    Q,
    c=Wells["T_K"],
    cmap="rainbow",
)
"""
ax1.scatter(
    -tilt_fn(Wells["T_K"]),
    Wells["misfit_Asqr_per_bond"] - sm,
    c=Wells["T_K"],
    cmap="rainbow",
)
ax1.scatter(
    tilt_fn(Wells["T_K"]),
    -Wells["misfit_Asqr_per_bond"] + sm,
    c=Wells["T_K"],
    cmap="rainbow",
)
ax1.scatter(
    -tilt_fn(Wells["T_K"]),
    -Wells["misfit_Asqr_per_bond"] + sm,
    c=Wells["T_K"],
    cmap="rainbow",
)
"""

misfits = np.linspace(misfit_fn(0.0), misfit_fn(857.0), 1001)
x = misfits - misfit_fn(0.0)
tilts = tilt_fn(0.0) - 4.0 * x - 22.0 * x * x

# ax[0].plot(tilts, np.sqrt(misfits), c="grey", linestyle=":")

# ax[1].plot(T_fn(tilts), np.sqrt(misfits), c="grey", linestyle=":")

temperatures = np.linspace(0.0, 1200.0, 101)

# a = 130
# b = 10000000.0
# Q1 = misfit_fn(0.0) + a * np.sqrt((np.sqrt(1.0 + temperatures / b) - 1.0))
# ax[1].plot(temperatures, Q1, c="grey", linestyle="--")
# ax[1].plot(temperatures, np.power(temperatures / 1000, 0.25), c="grey", linestyle="--")
# ax[1].plot(
#    temperatures,
#    np.sqrt(misfit_fn(0.0)) + np.power(temperatures / 2050, 0.5),
#    linestyle="--",
##    c="grey",
# )


ax[1].scatter(
    Wells["T_K"],
    Q,
    c=Wells["T_K"],
    cmap="rainbow",
)

t = ax[2].scatter(
    tilt_fn(Wells["T_K"]),
    Wells["T_K"],
    c=Wells["T_K"],
    cmap="rainbow",
)

cm = t.get_cmap()


p = 4.0


sm = 0.0  # misfit_fn(0.0)
ax[3].scatter(
    tilt_fn(Wells["T_K"]),
    1.0 / np.power((Q / np.power(Wells["T_K"], 1.0 / p)), p),
    c=Wells["T_K"],
    cmap=cm,
)
ax[3].scatter(
    -tilt_fn(Wells["T_K"]),
    1.0 / np.power((Q / np.power(Wells["T_K"], 1.0 / p)), p),
    c=Wells["T_K"],
    cmap=cm,
)


def Q_factor(tilt):
    t = tilt / tilt_fn(0.0)
    # return 8.95e-6 * (1 - 1.8 * (np.power(t, 2.0)) + 0.9 * (np.power(t, 4.0)))
    # The following makes c (i.e. c * Q1^6) a quadratic function of tilt
    return 0.88 / (1.0e5 + 5.0e5 * t * t)


# ax[0].plot(
#    tilt_fn(Wells["T_K"]),
#    np.power(Q_factor(tilt_fn(Wells["T_K"])) * Wells["T_K"], 1.0 / p),
# )

# ax[1].plot(
#     Wells["T_K"], np.power(Q_factor(tilt_fn(Wells["T_K"])) * Wells["T_K"], 1.0 / p)
# )

# ax[1].plot(Antao["T_K"], np.power(Q_factor(Antao["tilt"]) * Antao["T_K"], 1.0 / p))

temperatures = np.linspace(0.0, 1500.0, 1001)

ax[0].plot(
    tilt_fn(temperatures),
    np.power(Q_factor(tilt_fn(temperatures)) * temperatures, 1.0 / p),
    color="orange",
)

ax[1].plot(
    temperatures,
    np.power(Q_factor(tilt_fn(temperatures)) * temperatures, 1.0 / p),
    color="orange",
)

ax[2].plot(
    tilt_fn(temperatures),
    temperatures,
    color="orange",
)

ax[3].plot(
    tilt_fn(temperatures),
    1.0 / Q_factor(tilt_fn(temperatures)),
    color="orange",
)
ax[3].plot(
    -tilt_fn(temperatures),
    1.0 / Q_factor(tilt_fn(temperatures)),
    color="orange",
)


mask = Antao["tilt"] > 5
ax[2].plot(Antao["tilt"][mask], Antao["T_K"][mask], c="grey")
mask = Antao["tilt"] < 5
ax[2].plot(Antao["tilt"][mask], Antao["T_K"][mask], c="grey")


ax[0].set_xlabel("Tilt ($^{\\circ}$)")
ax[0].set_ylabel("RMS tilt (rad, sqrt(M)/V$^{1/3}$)")

ax[1].set_xlabel("Temperature (K)")
ax[1].set_ylabel("RMS tilt (rad, sqrt(M)/V$^{1/3}$)")

ax[2].set_xlabel("Tilt ($^{\\circ}$)")
ax[2].set_ylabel("Temperature (K)")

ax[3].set_xlabel("Tilt ($^{\\circ}$)")
ax[3].set_ylabel("$T Q_1^{-4}$")


for i in range(2, 3):
    ax[i].set_xlim(
        0.0,
    )
    ax[i].set_ylim(
        0.0,
    )

# cbar = fig.colorbar(t, ax=ax[2])
# cbar.set_label("T (K)", rotation=0)
fig.set_tight_layout(True)
fig.savefig("figures/Q_plot.pdf")
plt.show()
