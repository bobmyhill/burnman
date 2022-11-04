import pandas as pd
from burnman.constants import Avogadro
import numpy as np

data = {}
data["Antao"] = pd.read_csv(
    "data/Antao_2016_quartz_structure_1bar.dat", delim_whitespace=True, comment="#"
)

data["Antao"]["V"] = data["Antao"]["V"] * Avogadro / 1.0e30 / 3
data["Antao"]["unc_V"] = data["Antao"]["unc_V"] * Avogadro / 1.0e30 / 3

data["Antao"]["a"] = data["Antao"]["a"] * np.cbrt(Avogadro / 1.0e30 / 3)
data["Antao"]["unc_a"] = data["Antao"]["unc_a"] * np.cbrt(Avogadro / 1.0e30 / 3)
data["Antao"]["c"] = data["Antao"]["c"] * np.cbrt(Avogadro / 1.0e30 / 3)
data["Antao"]["unc_c"] = data["Antao"]["unc_c"] * np.cbrt(Avogadro / 1.0e30 / 3)


data["Bachheimer"] = pd.read_csv(
    "data/Bachheimer_Dolino_1975_quartz_Q.dat", delim_whitespace=True, comment="#"
)
data["Hazen"] = pd.read_csv(
    "data/Hazen_et_al_1989_quartz_cell.dat", delim_whitespace=True, comment="#"
)

data["Hazen"]["V"] = data["Hazen"]["V"] * Avogadro / 1.0e30 / 3

data["Gronvold"] = pd.read_csv(
    "data/Gronvold_et_al_1989_quartz_Cp.dat", delim_whitespace=True, comment="#"
)
data["Richet"] = pd.read_csv(
    "data/Richet_et_al_1992_quartz_Cp.dat", delim_whitespace=True, comment="#"
)
data["Jorgensen"] = pd.read_csv(
    "data/Jorgensen_1978_quartz_tilts_high_pressure.dat",
    delim_whitespace=True,
    comment="#",
)

data["Jorgensen"]["V"] = data["Jorgensen"]["V"] * Avogadro / 1.0e30 / 3

data["Scheidl"] = pd.read_csv(
    "data/Scheidl_et_al_2016_quartz_cell.dat", delim_whitespace=True, comment="#"
)

data["Scheidl"]["T_K"] = data["Scheidl"]["P_GPa"] * 0.0 + 300.0
data["Scheidl"]["V"] = data["Scheidl"]["V"] * Avogadro / 1.0e30 / 3
data["Scheidl"]["unc_V"] = data["Scheidl"]["unc_V"] * Avogadro / 1.0e30 / 3

data["Scheidl"]["a"] = data["Scheidl"]["a"] * np.cbrt(Avogadro / 1.0e30 / 3)
data["Scheidl"]["unc_a"] = data["Scheidl"]["unc_a"] * np.cbrt(Avogadro / 1.0e30 / 3)
data["Scheidl"]["c"] = data["Scheidl"]["c"] * np.cbrt(Avogadro / 1.0e30 / 3)
data["Scheidl"]["unc_c"] = data["Scheidl"]["unc_c"] * np.cbrt(Avogadro / 1.0e30 / 3)


data["Raz"] = pd.read_csv(
    "data/Raz_et_al_2002_quartz_PVT.dat", delim_whitespace=True, comment="#"
)

data["Raz"]["T_K"] = data["Raz"]["T_C"] + 273.15
data["Raz"]["P_GPa"] = data["Raz"]["P_bar"] / 1.0e4
data["Raz"]["V"] = (1.0 + 0.01 * data["Raz"]["dV_V0_pct"]) * data["Antao"]["V"][0]
data["Raz"]["a"] = (1.0 + 0.01 * data["Raz"]["dx_x0_pct"]) * data["Antao"]["a"][0]
data["Raz"]["c"] = (1.0 + 0.01 * data["Raz"]["dz_z0_pct"]) * data["Antao"]["c"][0]


data["Carpenter_neutron"] = pd.read_csv(
    "data/Carpenter_1998_quartz_unit_cell_neutron.dat",
    delim_whitespace=True,
    comment="#",
)
data["Carpenter_neutron"]["P_GPa"] = data["Carpenter_neutron"]["T_K"] * 0.0 + 1.0e-4

data["Carpenter_neutron"]["V"] = data["Carpenter_neutron"]["V"] * Avogadro / 1.0e30 / 3
data["Carpenter_neutron"]["Verr"] = (
    data["Carpenter_neutron"]["Verr"] * Avogadro / 1.0e30 / 3
)
data["Carpenter_neutron"]["a"] = data["Carpenter_neutron"]["a"] * np.cbrt(
    Avogadro / 1.0e30 / 3
)
data["Carpenter_neutron"]["aerr"] = data["Carpenter_neutron"]["aerr"] * np.cbrt(
    Avogadro / 1.0e30 / 3
)
data["Carpenter_neutron"]["c"] = data["Carpenter_neutron"]["c"] * np.cbrt(
    Avogadro / 1.0e30 / 3
)
data["Carpenter_neutron"]["cerr"] = data["Carpenter_neutron"]["cerr"] * np.cbrt(
    Avogadro / 1.0e30 / 3
)

data["Carpenter_XRD"] = pd.read_csv(
    "data/Carpenter_1998_quartz_unit_cell_XRD.dat", delim_whitespace=True, comment="#"
)

data["Carpenter_XRD"]["P_GPa"] = data["Carpenter_XRD"]["T_K"] * 0.0 + 1.0e-4

data["Carpenter_XRD"]["V"] = data["Carpenter_XRD"]["V"] * Avogadro / 1.0e30 / 3
data["Carpenter_XRD"]["Verr"] = data["Carpenter_XRD"]["Verr"] * Avogadro / 1.0e30 / 3

data["Carpenter_XRD"]["a"] = data["Carpenter_XRD"]["a"] * np.cbrt(Avogadro / 1.0e30 / 3)
data["Carpenter_XRD"]["aerr"] = data["Carpenter_XRD"]["aerr"] * np.cbrt(
    Avogadro / 1.0e30 / 3
)
data["Carpenter_XRD"]["c"] = data["Carpenter_XRD"]["c"] * np.cbrt(Avogadro / 1.0e30 / 3)
data["Carpenter_XRD"]["cerr"] = data["Carpenter_XRD"]["cerr"] * np.cbrt(
    Avogadro / 1.0e30 / 3
)

data["Ogata"] = pd.read_csv(
    "data/Ogata_et_al_1987_quartz_cell.dat", delim_whitespace=True, comment="#"
)

data["Ogata"]["V"] = data["Ogata"]["V"] * Avogadro / 1.0e30 / 3

data["Lakshtanov"] = pd.read_csv(
    "data/Lakshtanov_et_al_2007_Cijs_quartz.dat", delim_whitespace=True, comment="#"
)

data["Lakshtanov"]["T_K"] = data["Lakshtanov"]["T_C"] + 273.15
data["Lakshtanov"]["P_GPa"] = data["Lakshtanov"]["T_C"] * 0.0 + 1.0e-4
data["Lakshtanov"]["V"] = 0.06008 / (data["Lakshtanov"]["rho"] * 1.0e3)

data["Axe"] = pd.read_csv(
    "data/Axe_Shirane_1970_quartz_scattering_103.dat",
    delim_whitespace=True,
    comment="#",
)

data["Wang"] = pd.read_csv(
    "data/Wang_2015_elastic_tensor_quartz_pressure.dat",
    delim_whitespace=True,
    comment="#",
)

data["Wang"]["T_K"] = data["Wang"]["P_GPa"] * 0.0 + 300.0
data["Wang"]["V"] = 0.06008 / (data["Wang"]["rho"] * 1.0e3)

colours = {}
colours["Antao"] = "red"
colours["Bachheimer"] = "orange"
colours["Hazen"] = "yellow"
colours["Gronvold"] = "green"
colours["Richet"] = "pink"
colours["Jorgensen"] = "magenta"
colours["Scheidl"] = "purple"
colours["Raz"] = "navy"
colours["Raz_3"] = "black"
colours["Carpenter_neutron"] = "blue"
colours["Carpenter_XRD"] = "cyan"
colours["Ogata"] = "maroon"
colours["Lakshtanov"] = "maroon"
colours["Wang"] = "navy"

labels = {}
labels["Antao"] = "Antao (XRD)"
labels["Bachheimer"] = "Bachheimer"
labels["Hazen"] = "Hazen"
labels["Gronvold"] = "Gronvold"
labels["Richet"] = "Richet"
labels["Jorgensen"] = "Jorgensen"
labels["Scheidl"] = "Scheidl"
labels["Raz"] = "Raz"
labels["Raz_3"] = "Raz (3 kbar)"
labels["Carpenter_neutron"] = "Carpenter (neutron)"
labels["Carpenter_XRD"] = "Carpenter (XRD)"
labels["Ogata"] = "Ogata"
labels["Lakshtanov"] = "Lakshtanov"
colours["Wang"] = "Wang"


def get_S(dataset):
    L = dataset
    # Make bigger than needed to index from 1
    C = np.zeros((7, 7, len(L["C11"])))

    C[1, 1] = L["C11"].to_numpy()
    C[1, 2] = L["C12"].to_numpy()
    C[1, 3] = L["C13"].to_numpy()
    C[1, 4] = L["C14"].to_numpy()
    C[3, 3] = L["C33"].to_numpy()
    C[4, 4] = L["C44"].to_numpy()
    C[6, 6] = (C[1, 1] - C[1, 2]) / 2.0

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
    return S / 1.0e9


def K_S(S):
    # Get the Reuss bulk modulus
    return 1.0 / np.sum(S[:, :3, :3], axis=(1, 2))


def calc_beta_S(S):
    # Get the isentropic compressibility tensor
    beta_S = np.sum(S[:, :, :3], axis=(2))
    return beta_S


for ds in [data["Lakshtanov"], data["Wang"]]:
    S = get_S(ds)
    ds["K_S"] = K_S(S)
    beta_S = calc_beta_S(S)
    ds["beta_S1"] = beta_S[:, 0]
    ds["beta_S3"] = beta_S[:, 2]
    ds["S11"] = S[:, 0, 0]
    ds["S33"] = S[:, 2, 2]
    ds["S44"] = S[:, 3, 3]
    ds["S12"] = S[:, 0, 1]
    ds["S13"] = S[:, 0, 2]
    ds["S14"] = S[:, 0, 3]
