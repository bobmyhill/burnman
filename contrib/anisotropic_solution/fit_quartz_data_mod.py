import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def tilt_regular(theta):
    # theta is the Si-O-Si bond angle (degrees)
    return np.degrees(
        np.arccos(
            np.sqrt(0.75 - np.cos(np.radians(theta))) - 1.0 / (2.0 * np.sqrt(3.0))
        )
    )


def tilt_distorted(c_over_a, x, z):
    # c_over_a is the c/a ratio
    # x and z are the 6c positions (Oxygen atom positions (x,y,z))
    # of space group P3_121
    # typically x~0.41, z~0.22
    return np.degrees(
        np.arctan(2.0 * np.sqrt(3.0) / 9.0 * c_over_a * (6.0 * z - 1.0) / x)
    )


Antao = pd.read_csv(
    "data/Antao_2016_quartz_structure_1bar.dat", delim_whitespace=True, comment="#"
)
Bachheimer = pd.read_csv(
    "data/Bachheimer_Dolino_1975_quartz_Q.dat", delim_whitespace=True, comment="#"
)
Hazen = pd.read_csv(
    "data/Hazen_et_al_1989_quartz_cell.dat", delim_whitespace=True, comment="#"
)
Gronvold = pd.read_csv(
    "data/Gronvold_et_al_1989_quartz_Cp.dat", delim_whitespace=True, comment="#"
)
Richet = pd.read_csv(
    "data/Richet_et_al_1992_quartz_Cp.dat", delim_whitespace=True, comment="#"
)
Jorgensen = pd.read_csv(
    "data/Jorgensen_1978_quartz_tilts_high_pressure.dat",
    delim_whitespace=True,
    comment="#",
)
Scheidl = pd.read_csv(
    "data/Scheidl_et_al_2016_quartz_cell.dat", delim_whitespace=True, comment="#"
)
Raz = pd.read_csv(
    "data/Raz_et_al_2002_quartz_PVT.dat", delim_whitespace=True, comment="#"
)

Carpenter_neutron = pd.read_csv(
    "data/Carpenter_1998_quartz_unit_cell_neutron.dat",
    delim_whitespace=True,
    comment="#",
)

Carpenter_XRD = pd.read_csv(
    "data/Carpenter_1998_quartz_unit_cell_XRD.dat", delim_whitespace=True, comment="#"
)

Ogata = pd.read_csv(
    "data/Ogata_et_al_1987_quartz_cell.dat", delim_whitespace=True, comment="#"
)

Lakshtanov = pd.read_csv(
    "data/Lakshtanov_et_al_2007_Cijs_quartz.dat", delim_whitespace=True, comment="#"
)

L = Lakshtanov
# Make bigger than needed to index from 1
C = np.zeros((7, 7, len(L["C11"])))

C[1, 1] = L["C11"].to_numpy()
C[1, 2] = L["C12"].to_numpy()
C[1, 3] = L["C13"].to_numpy()
C[1, 4] = L["C14"].to_numpy()
C[3, 3] = L["C33"].to_numpy()
C[4, 4] = L["C44"].to_numpy()
C[6, 6] = L["C66"].to_numpy()

# Add dependent upper triangular elements
C[2, 2] = C[1, 1]
C[2, 3] = C[1, 3]
C[2, 4] = -C[1, 4]
C[5, 5] = C[4, 4]
C[5, 6] = C[1, 4]

# Clip to correct size
C = C[1:, 1:]

# Make symmetric
C = (
    np.einsum("ijk->kij", C)
    + np.einsum("jik->kij", C)
    - np.einsum("ijk, ij->kij", C, np.eye(6))
)

# Invert
S = np.linalg.inv(C)

# Get the Reuss bulk modulus
K_S = 1.0 / np.sum(S[:, :3, :3], axis=(1, 2))

c_Antao = "red"
c_Bachheimer = "orange"
c_Hazen = "yellow"
c_Gronvold = "green"
c_Richet = "pink"
c_Jorgensen = "magenta"
c_Scheidl = "purple"
c_Raz = "navy"
c_Raz_3 = "black"
c_Carpenter_neutron = "blue"
c_Carpenter_XRD = "cyan"
c_Ogata = "maroon"
c_Lakshtanov = "maroon"

l_Antao = "Antao"
l_Bachheimer = "Bachheimer"
l_Hazen = "Hazen"
l_Gronvold = "Gronvold"
l_Richet = "Richet"
l_Jorgensen = "Jorgensen"
l_Scheidl = "Scheidl"
l_Raz = "Raz"
l_Raz_3 = "Raz (3 kbar)"
l_Carpenter_neutron = "Carpenter (N)"
l_Carpenter_XRD = "Carpenter (X)"
l_Ogata = "Ogata"
l_Lakshtanov = "Lakshtanov"

fig1 = plt.figure(figsize=(10, 12))
fig2 = plt.figure(figsize=(10, 8))
fig3 = plt.figure(figsize=(5, 4))

ax = [fig1.add_subplot(3, 2, i) for i in range(1, 7)]
ax.extend([fig2.add_subplot(2, 2, i - 6) for i in range(7, 11)])
ax.append(fig3.add_subplot(1, 1, 1))

i = {
    "T_V": 0,
    "V_P": 1,
    "T_tilt": 2,
    "V_tilt": 3,
    "T_CP": 4,
    "T_KS": 5,
    "T_a": 6,
    "V_a": 7,
    "T_c": 8,
    "V_c": 9,
    "PVT": 10,
}

x = {
    "T_V": "T (K)",
    "V_P": "V (Angstrom$^3$)",
    "T_tilt": "T (K)",
    "V_tilt": "V (Angstrom$^3$)",
    "T_a": "V (Angstrom$^3$)",
    "V_a": "V (Angstrom$^3$)",
    "T_c": "V (Angstrom$^3$)",
    "V_c": "V (Angstrom$^3$)",
    "T_CP": "T (K)",
    "PVT": "T (K)",
    "T_KS": "T (K)",
}

y = {
    "T_V": "V (Angstrom$^3$)",
    "V_P": "P (GPa)",
    "T_tilt": "tilt ($^{\\circ}$)",
    "V_tilt": "tilt ($^{\\circ}$)",
    "T_a": "a (Angstrom)",
    "V_a": "a (Angstrom)",
    "T_c": "c (Angstrom)",
    "V_c": "c (Angstrom)",
    "T_CP": "CP (J/K/mol)",
    "PVT": "V (Angstrom)",
    "T_KS": "$K_S$ (GPa)",
}

x = {i[k]: x[k] for k in i.keys()}
y = {i[k]: y[k] for k in i.keys()}

i = {
    "T_V": 0,
    "V_P": 1,
    "T_tilt": 2,
    "V_tilt": 3,
    "T_CP": 4,
    "T_KS": 5,
    "T_a": 7,
    "V_a": 7,
    "T_c": 9,
    "V_c": 9,
    "PVT": 10,
}

idx = Raz["P_bar"].isin([1])
ax[i["T_V"]].scatter(
    Raz["T_C"][idx] + 273.15,
    (1.0 + 0.01 * Raz["dV_V0_pct"][idx]) * Antao["V"][0],
    c=c_Raz,
    label=l_Raz,
)
ax[i["T_a"]].scatter(
    (1.0 + 0.01 * Raz["dV_V0_pct"][idx]) * Antao["V"][0],
    ((1.0 + 0.01 * Raz["dx_x0_pct"][idx]) * Antao["a"][0])
    / np.power((1.0 + 0.01 * Raz["dV_V0_pct"][idx]) * Antao["V"][0], 2.0),
    c=c_Raz,
    label=l_Raz,
)
ax[i["T_c"]].scatter(
    (1.0 + 0.01 * Raz["dV_V0_pct"][idx]) * Antao["V"][0],
    (
        (1.0 + 0.01 * Raz["dz_z0_pct"][idx])
        * Antao["c"][0]
        / ((1.0 + 0.01 * Raz["dV_V0_pct"][idx]) * Antao["V"][0])
    ),
    c=c_Raz,
    label=l_Raz,
)

idx = Raz["P_bar"].isin([3000])
ax[i["T_V"]].scatter(
    Raz["T_C"][idx] + 273.15,
    (1.0 + 0.01 * Raz["dV_V0_pct"][idx]) * Antao["V"][0],
    c=c_Raz_3,
    label=l_Raz_3,
)
ax[i["T_a"]].scatter(
    (1.0 + 0.01 * Raz["dV_V0_pct"][idx]) * Antao["V"][0],
    ((1.0 + 0.01 * Raz["dx_x0_pct"][idx]) * Antao["a"][0])
    / np.power((1.0 + 0.01 * Raz["dV_V0_pct"][idx]) * Antao["V"][0], 2.0),
    c=c_Raz_3,
    label=l_Raz_3,
)
ax[i["T_c"]].scatter(
    (1.0 + 0.01 * Raz["dV_V0_pct"][idx]) * Antao["V"][0],
    (
        (1.0 + 0.01 * Raz["dz_z0_pct"][idx])
        * Antao["c"][0]
        / ((1.0 + 0.01 * Raz["dV_V0_pct"][idx]) * Antao["V"][0])
    ),
    c=c_Raz_3,
    label=l_Raz_3,
)

ax[i["T_V"]].scatter(
    Carpenter_neutron["T_K"],
    Carpenter_neutron["V"],
    c=c_Carpenter_neutron,
    label=l_Carpenter_neutron,
)
ax[i["T_a"]].scatter(
    Carpenter_neutron["V"],
    Carpenter_neutron["a"] / np.power(Carpenter_neutron["V"], 2.0),
    c=c_Carpenter_neutron,
    label=l_Carpenter_neutron,
)
ax[i["T_c"]].scatter(
    Carpenter_neutron["V"],
    Carpenter_neutron["c"] / np.power(Carpenter_neutron["V"], 1.0),
    c=c_Carpenter_neutron,
    label=l_Carpenter_neutron,
)
ax[i["T_V"]].scatter(
    Carpenter_XRD["T_K"], Carpenter_XRD["V"], c=c_Carpenter_XRD, label=l_Carpenter_XRD
)
ax[i["T_a"]].scatter(
    Carpenter_XRD["V"],
    Carpenter_XRD["a"] / np.power(Carpenter_XRD["V"], 2.0),
    c=c_Carpenter_XRD,
    label=l_Carpenter_XRD,
)
ax[i["T_c"]].scatter(
    Carpenter_XRD["V"],
    Carpenter_XRD["c"] / np.power(Carpenter_XRD["V"], 1.0),
    c=c_Carpenter_XRD,
    label=l_Carpenter_XRD,
)


idx = Raz["T_C"].isin([25])
ax[i["V_P"]].scatter(
    (1.0 + 0.01 * Raz["dV_V0_pct"][idx]) * Antao["V"][0],
    Raz["P_bar"][idx] / 1.0e4,
    c=c_Raz,
    label=l_Raz,
)
ax[i["V_a"]].scatter(
    (1.0 + 0.01 * Raz["dV_V0_pct"][idx]) * Antao["V"][0],
    ((1.0 + 0.01 * Raz["dx_x0_pct"][idx]) * Antao["a"][0])
    / np.power((1.0 + 0.01 * Raz["dV_V0_pct"][idx]) * Antao["V"][0], 2.0),
    c=c_Raz,
    label=l_Raz,
)
ax[i["V_c"]].scatter(
    (1.0 + 0.01 * Raz["dV_V0_pct"][idx]) * Antao["V"][0],
    (
        (1.0 + 0.01 * Raz["dz_z0_pct"][idx])
        * Antao["c"][0]
        / ((1.0 + 0.01 * Raz["dV_V0_pct"][idx]) * Antao["V"][0])
    ),
    c=c_Raz,
    label=l_Raz,
)


for P in Raz["P_bar"].unique():
    idx = Raz["P_bar"].isin([P])
    ax[i["PVT"]].scatter(
        Raz["T_C"][idx] + 273.15,
        Raz["V_cm3"][idx],
        label=f"{P}",
    )

ax[i["T_tilt"]].scatter(Antao["T_K"], Antao["tilt"], c=c_Antao, label=l_Antao)
ax[i["T_tilt"]].scatter(
    Bachheimer["T_K"],
    Bachheimer["Q_norm"] * Antao["tilt"][i["T_tilt"]],
    c=c_Bachheimer,
    label=l_Bachheimer,
)

ax[i["V_tilt"]].errorbar(
    Jorgensen["V"],
    Jorgensen["tilt"],
    yerr=Jorgensen["unc_tilt"],
    linestyle="None",
    c=c_Jorgensen,
)
ax[i["V_tilt"]].scatter(
    Jorgensen["V"], Jorgensen["tilt"], c=c_Jorgensen, label=l_Jorgensen
)


# !!!! 1 BAR DATA ON V-TILT FIGURE
ax[i["V_tilt"]].scatter(Antao["V"], Antao["tilt"], c=c_Antao, label=l_Antao)
ax[i["V_tilt"]].scatter(Ogata["V"], Ogata["tilt"], c=c_Ogata, label=l_Ogata)

Hazen_tilt = tilt_distorted(Hazen["c"] / Hazen["a"], Hazen["x"], Hazen["z"])
ax[i["V_tilt"]].scatter(Hazen["V"], Hazen_tilt, c=c_Hazen, label=l_Hazen)

ax[i["T_V"]].errorbar(
    Antao["T_K"], Antao["V"], yerr=Antao["unc_V"], linestyle="None", c=c_Antao
)
ax[i["T_V"]].scatter(Antao["T_K"], Antao["V"], c=c_Antao, label=l_Antao)


ax[i["T_CP"]].scatter(Gronvold["T_K"], Gronvold["CP"], c=c_Gronvold, label=l_Gronvold)
ax[i["T_CP"]].scatter(Richet["T_K"], Richet["CP"], c=c_Richet, label=l_Richet)


ax[i["V_P"]].errorbar(
    Jorgensen["V"],
    Jorgensen["P_kbar"] / 10.0,
    xerr=Jorgensen["unc_V"],
    linestyle="None",
    c=c_Jorgensen,
)
ax[i["V_P"]].scatter(
    Jorgensen["V"], Jorgensen["P_kbar"] / 10.0, c=c_Jorgensen, label=l_Jorgensen
)

ax[i["V_P"]].errorbar(
    Scheidl["V"],
    Scheidl["P_GPa"],
    yerr=Scheidl["unc_P"],
    xerr=Scheidl["unc_V"],
    linestyle="None",
    c=c_Scheidl,
)
ax[i["V_P"]].scatter(Scheidl["V"], Scheidl["P_GPa"], c=c_Scheidl, label=l_Scheidl)

ax[i["V_P"]].errorbar(
    Hazen["V"], Hazen["P_GPa"], xerr=Hazen["unc_V"], linestyle="None", c=c_Hazen
)
ax[i["V_P"]].scatter(Hazen["V"], Hazen["P_GPa"], c=c_Hazen, label=l_Hazen)


for axis in ["a", "c"]:
    if axis == "a":
        p = 2
    else:
        p = 1

    ax[i[f"T_{axis}"]].scatter(
        Antao["V"], Antao[axis] / np.power(Antao["V"], p), c=c_Antao, label=l_Antao
    )

    ax[i[f"V_{axis}"]].scatter(
        Jorgensen["V"],
        Jorgensen[axis] / np.power(Jorgensen["V"], p),
        c=c_Jorgensen,
        label=l_Jorgensen,
    )

    ax[i[f"V_{axis}"]].scatter(
        Scheidl["V"],
        Scheidl[axis] / np.power(Scheidl["V"], p),
        c=c_Scheidl,
        label=l_Scheidl,
    )

    ax[i[f"V_{axis}"]].scatter(
        Hazen["V"], Hazen[axis] / np.power(Hazen["V"], p), c=c_Hazen, label=l_Hazen
    )


ax[i["T_KS"]].scatter(
    Lakshtanov["T_C"] + 273.15, Lakshtanov["K"], c=[(0.9, 0.9, 0.9)], label=l_Lakshtanov
)
ax[i["T_KS"]].scatter(
    Lakshtanov["T_C"] + 273.15, K_S, c=c_Lakshtanov, label="Lakshtanov (from C$^{-1}$)"
)


for i in [0, 2, 4, 5, 10]:
    ax[i].set_xlim(0.0, 1400.0)
for i in [1, 3, 7, 9]:
    ax[i].set_xlim(85.0, 120.0)

if False:
    for i in [6, 8]:
        ax[i].set_xlim(116.5, 118.5)
    ax[6].set_ylim(4.96, 5.01)
    ax[8].set_ylim(5.44, 5.47)

for i in range(11):
    ax[i].legend()
    ax[i].set_xlabel(x[i])
    ax[i].set_ylabel(y[i])


fig1.set_tight_layout(True)
fig1.savefig("quartz_data.pdf")
fig2.set_tight_layout(True)
fig2.savefig("quartz_data_2.pdf")
fig3.set_tight_layout(True)
fig3.savefig("quartz_data_3.pdf")
plt.show()
