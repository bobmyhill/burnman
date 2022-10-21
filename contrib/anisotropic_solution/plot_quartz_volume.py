from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from quartz_model import qtz, alpha, beta
from data_tables import data, labels


temperatures = np.linspace(1.0, 2500.0, 2500)
volumes = np.empty_like(temperatures)
volumes1 = np.empty_like(temperatures)
volumes2 = np.empty_like(temperatures)

cs = ["red", "orange", "purple"]

plt.scatter(
    data["Carpenter_neutron"]["T_K"],
    data["Carpenter_neutron"]["V"] * 1.0e6,
    label=labels["Carpenter_neutron"],
    s=30,
    marker="+",
    c=cs[0],
)

plt.scatter(
    data["Carpenter_XRD"]["T_K"],
    data["Carpenter_XRD"]["V"] * 1.0e6,
    label=labels["Carpenter_XRD"],
    s=15,
    marker="*",
    c=cs[0],
)

plt.scatter(
    data["Antao"]["T_K"],
    data["Antao"]["V"] * 1.0e6,
    label=labels["Antao"],
    s=5,
    marker="^",
    c=cs[0],
)


for j, P in enumerate([1.0e5, 1.4e8, 3.0e8]):
    guess = [1.0, 1.0]
    for i, T in enumerate(temperatures):
        qtz.equilibrate(P, T)
        volumes[i] = qtz.V
        volumes1[i] = alpha.V
        volumes2[i] = beta.V

    idx = data["Raz"]["P_bar"].isin([int(P / 1.0e5)])
    plt.scatter(
        data["Raz"]["T_C"][idx] + 273.15,
        (1.0 + 0.01 * data["Raz"]["dV_V0_pct"][idx]) * data["Antao"]["V"][0] * 1.0e6,
        s=5,
        c=cs[j],
        label=f"Raz et al. ({P/1.e9} GPa)",
    )

    plt.plot(
        temperatures,
        volumes * 1.0e6,
        c=cs[j],
        label=f"$V_{{eqm}}$ ({P/1.e9} GPa)",
    )
    plt.plot(
        temperatures,
        volumes1 * 1.0e6,
        c=cs[j],
        linestyle=":",
        label=f"$V_{{\\alpha}}$ ({P/1.e9} GPa)",
    )
    plt.plot(
        temperatures,
        volumes2 * 1.0e6,
        c=cs[j],
        linestyle="--",
        label=f"$V_{{\\beta}}$ ({P/1.e9} GPa)",
    )

plt.xlim(0.0, 2500.0)
plt.xlabel("Temperature (K)")
plt.ylabel("Volume (cm$^3$/mol)")
plt.legend(loc="lower right")
plt.savefig("figures/qtz_volumes.pdf")
plt.show()
