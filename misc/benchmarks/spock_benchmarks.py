from __future__ import absolute_import

# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit
# for the Earth and Planetary Sciences.
# Copyright (C) 2012 - 2024 by the BurnMan team, released under the GNU
# GPL v2 or later.

from burnman.tools.eos import check_eos_consistency
from burnman import Mineral
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HMX_params = {
    "P_0": 1.0e5,
    "V_0": 1.0e-6,  # arbitrary value
    "K_0": 15.22e9,
    "Kprime_0": 7.54,
    "Kdprime_0": -7.54 / 15.22e9,
    "Kprime_inf": 2.63,
    "molar_mass": 0.296155,
    "equation_of_state": "spock",
}

HMX = Mineral(HMX_params)

if check_eos_consistency(HMX, tol=1.0e-5, including_shear_properties=False):
    print("The SPOCK EoS is internally consistent.\n")

pressures = np.linspace(0.0, 100.0e9, 6)
temperatures = 0.0 + 0.0 * pressures
V, K_T = HMX.evaluate(["V", "K_T"], pressures, temperatures)

for i in range(6):
    print(
        f"{pressures[i]/1.e9:3.0f} GPa: "
        f"V/V_0 = {V[i]/HMX_params['V_0']:.3f}, "
        f"K_T = {K_T[i]/1.e9:6.2f} GPa"
    )


fig1 = mpimg.imread("figures/Lozano_Aslam_2022_Fig6c_HMX.png")
plt.imshow(fig1, extent=[0.0, 100.0, 0, 500.0], aspect="auto")

for a in [1.0, 1.5, 2.0]:
    HMX_params["Kdprime_0"] = -a * HMX_params["Kprime_0"] / HMX_params["K_0"]

    pressures = np.linspace(0.0, 100.0e9, 101)
    temperatures = 0.0 + 0.0 * pressures
    K_T = HMX.evaluate(["K_T"], pressures, temperatures)[0]

    plt.plot(pressures / 1.0e9, K_T / 1.0e9, linestyle=":", label=f"SPOCK {a}")

plt.ylim(0.0, 500.0)
plt.legend(loc=(0.025, 0.5))
plt.show()