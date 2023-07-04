import os.path
import sys

sys.path.insert(1, os.path.abspath("../.."))

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from burnman import Mineral
from burnman.minerals.Pitzer_Sterner_1994 import H2O_Pitzer_Sterner

H2O = H2O_Pitzer_Sterner()

"""
rho in Pitzer and Sterner (1994) is the inverse of the molar_volume
"""

# critical density of water is 0.322 g/cm3, which is a specific density of
# 0.322 / 18.01528 = 0.01787 mol/cm^3

# In Sterner and Pitzer, density is in mol/cm^3,
# pressure is in MPa.
Tc = 647  # K
Pc = 22.064e6 / 1.0e6  # MPa
zc = 0.229  # from paper
rhoc = Pc / (zc * 8.31446 * Tc)

rhoc = 3.22e2  # kg/m^3

print(H2O.method.pressure(Tc, 1.0e-6 / 0.01787, H2O.params))

PS1994_Fig1 = mpimg.imread("data/Pitzer_Sterner_1994_Fig_1.png")
PS1994_Fig2a = mpimg.imread("data/Pitzer_Sterner_1994_Fig2a.png")
plt.imshow(PS1994_Fig1, extent=[0.0, 7.0, 0.0, 25.0], aspect="auto")
plt.imshow(PS1994_Fig2a, extent=[0.0, 4.0, 0.0, 5.0], aspect="auto")


densities = rhoc * np.linspace(0.1, 7, 101)  # kg/m^3
volumes = H2O.params["molar_mass"] / densities  # m^3/mol

reduced_pressures = np.empty_like(densities)
for T in [Tc, 1600.0]:
    for i, volume in enumerate(volumes):
        rho = H2O.params["molar_mass"] / volume
        reduced_pressures[i] = (
            H2O.method.pressure(T, volume, H2O.params) * volume / (8.31446 * T)
        )

    plt.plot(densities / rhoc, reduced_pressures, label=f"{T} K")
plt.legend()
plt.ylim(0, 25)
plt.xlabel("Reduced densities ($\\rho / \\rho_c$; GPa)")
plt.ylabel("Reduced pressures ($PV / (RT)$)")
plt.show()

H2O_props = np.loadtxt("data/Barin_H2O_properties.dat", unpack=True)
Ts, Cps, Ss, Grels, Gs = H2O_props


temperatures = np.linspace(500.0, 4000.0, 101)
volumes = np.empty_like(temperatures)
entropies = np.empty_like(temperatures)
helmholtz_energies = np.empty_like(temperatures)

fig = plt.figure()
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

for P in [1.0e5, 4.0e5, 16.0e5, 64.0e5, 256.0e5, 1.0e8, 4.0e8, 16.0e8, 64.0e8, 256.0e8]:
    pressures = temperatures * 0.0 + P
    volumes, entropies, helmholtz_energies = H2O.evaluate(
        ["V", "S", "molar_helmholtz"], pressures, temperatures
    )

    ax[0].plot(temperatures, H2O.params["molar_mass"] / volumes)
    ax[1].plot(temperatures, entropies, label=f"{P/1.e9} GPa")
    ax[2].plot(
        temperatures, (helmholtz_energies + P * volumes) / 1.0e3, label=f"{P/1.e9} GPa"
    )
    # ax[1].plot(temperatures, temperatures*np.gradient(entropies, temperatures), label=f'{P/1.e9} GPa')


ax[1].plot(Ts, Ss, linestyle=":", label="Barin entropies (1 bar)")
ax[2].plot(
    Ts,
    Gs / 1.0e3 + 233.2550 * 298.15 / 1.0e3,
    linestyle=":",
    label="Barin gibbs (1 bar; HSC->SUP converted)",
)

for i in range(3):
    ax[i].set_xlabel("Temperature (K)")
ax[0].set_ylabel("Density (kg/m$^3$)")
ax[1].set_ylabel("Entropy (J/K/mol)")
ax[2].set_ylabel("Gibbs (kJ/mol)")

ax[0].set_ylim(
    0,
)
ax[1].set_ylim(
    0,
)

ax[1].legend()
ax[2].legend()
plt.show()
