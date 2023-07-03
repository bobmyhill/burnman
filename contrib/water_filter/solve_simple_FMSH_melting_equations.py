import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from FMSH_endmembers_linearised import fo, wad, ring, maj, Mg2SiO4L, H2OL
from FMSH_endmembers_linearised import H2MgSiO4fo as hyfo
from FMSH_endmembers_linearised import H2MgSiO4wad as hywad
from FMSH_endmembers_linearised import H2MgSiO4ring as hyring
from FMSH_melt_model import ax_melt

R = 8.31446
W = -91000.0


def gibbs(mineral, P, T):
    mineral.set_state(P, T)
    return mineral.gibbs


def delta_G(T, P, m1, m2):
    m1.set_state(P, T[0])
    m2.set_state(P, T[0])
    return m1.gibbs - m2.gibbs


def melting_temperature(fo_polymorph, P):
    return fsolve(delta_G, [2600.0], args=(P, fo_polymorph, Mg2SiO4L))[0]


def compositions(
    P,
    T,
    X_Mg2SiO4,
    X_H2O,
    X_MgSiO3,
    Tm,
    fo_polymorph,
    H2MgSiO4_polymorph,
    MgSiO3_polymorph,
):
    S_fusion = (gibbs(Mg2SiO4L, P, T) - gibbs(fo_polymorph, P, T)) / (Tm - T)
    a_H2OL, p_H2OL, _ = ax_melt(T, Tm, S_fusion)

    p_H2MgSiO4fo = a_H2OL * np.exp(
        -(
            gibbs(H2MgSiO4_polymorph, P, T)
            - gibbs(MgSiO3_polymorph, P, T)
            - gibbs(H2OL, P, T)
        )
        / (R * T)
    )

    # bulk_constraint_1 = (x_fo * p_H2MgSiO4fo) + (x_L * p_H2OL) - X_H2O
    # bulk_constraint_2 = x_en + (x_fo * p_H2MgSiO4fo) - X_MgSiO3
    # bulk_constraint_3 = (x_fo * p_Mg2SiO4fo) + (x_L * p_Mg2SiO4L) - X_Mg2SiO4

    x_fo = -(X_H2O * (p_H2OL - 1) + X_Mg2SiO4 * p_H2OL) / (p_H2MgSiO4fo - p_H2OL)
    x_en = X_MgSiO3 - p_H2MgSiO4fo * x_fo
    x_L = (X_H2O * (p_H2MgSiO4fo - 1) + X_Mg2SiO4 * p_H2MgSiO4fo) / (
        p_H2MgSiO4fo - p_H2OL
    )

    return (x_fo, x_en, x_L, p_H2OL, p_H2MgSiO4fo)


def proportion_H2MgSiO4_to_wtpercent_H2O(p_H2MgSiO4fo):
    p_Mg2SiO4fo = 1.0 - p_H2MgSiO4fo

    xMgOfo = 2.0 * p_Mg2SiO4fo + 1.0 * p_H2MgSiO4fo
    xSiO2fo = p_Mg2SiO4fo + p_H2MgSiO4fo
    xH2Ofo = p_H2MgSiO4fo

    mH2O = 18.01528 * xH2Ofo
    mMgO = 40.3044 * xMgOfo
    mSiO2 = 60.08 * xSiO2fo
    wtpctH2O = mH2O / (mH2O + mMgO + mSiO2) * 100.0
    return wtpctH2O


def proportion_H2O_to_wtpercent_H2O(p_H2OL):
    p_Mg2SiO4L = 1.0 - p_H2OL

    xMgOL = 2.0 * p_Mg2SiO4L
    xSiO2L = p_Mg2SiO4L
    xH2OL = p_H2OL

    mH2O = 18.01528 * xH2OL
    mMgO = 40.3044 * xMgOL
    mSiO2 = 60.08 * xSiO2L
    wtpctH2O = mH2O / (mH2O + mMgO + mSiO2) * 100.0
    return wtpctH2O


P1 = 8.0e9
Tm1 = melting_temperature(fo, P1)
P1b = 12.0e9
Tm1b = melting_temperature(fo, P1b)
P2 = 15.0e9
Tm2 = melting_temperature(wad, P2)
P2b = 17.0e9
Tm2b = melting_temperature(wad, P2b)
P3 = 20.0e9
Tm3 = melting_temperature(ring, P3)


fo_wtpctH2Os = np.empty(101)
fo_wtpctH2Osb = np.empty(101)
wad_wtpctH2Os = np.empty(101)
wad_wtpctH2Osb = np.empty(101)
ring_wtpctH2Os = np.empty(101)

foL_wtpctH2Os = np.empty(101)
foL_wtpctH2Osb = np.empty(101)
wadL_wtpctH2Os = np.empty(101)
wadL_wtpctH2Osb = np.empty(101)
ringL_wtpctH2Os = np.empty(101)


foL_H2O = np.empty(101)
foL_H2Ob = np.empty(101)
wadL_H2O = np.empty(101)
wadL_H2Ob = np.empty(101)
ringL_H2O = np.empty(101)

temperatures = np.linspace(500.0, 2500.0, 101)
for i, T in enumerate(temperatures):
    _, _, _, p_H2OL, p_H2MgSiO4fo = compositions(
        P=P1,
        T=T,
        X_Mg2SiO4=1.0,
        X_H2O=0.2,
        X_MgSiO3=1.0,
        Tm=Tm1,
        fo_polymorph=fo,
        H2MgSiO4_polymorph=hyfo,
        MgSiO3_polymorph=maj,
    )

    fo_wtpctH2Os[i] = proportion_H2MgSiO4_to_wtpercent_H2O(p_H2MgSiO4fo)
    foL_wtpctH2Os[i] = proportion_H2O_to_wtpercent_H2O(p_H2OL)
    foL_H2O[i] = p_H2OL

    _, _, _, p_H2OL, p_H2MgSiO4fob = compositions(
        P=P1b,
        T=T,
        X_Mg2SiO4=1.0,
        X_H2O=0.2,
        X_MgSiO3=1.0,
        Tm=Tm1b,
        fo_polymorph=fo,
        H2MgSiO4_polymorph=hyfo,
        MgSiO3_polymorph=maj,
    )

    fo_wtpctH2Osb[i] = proportion_H2MgSiO4_to_wtpercent_H2O(p_H2MgSiO4fob)
    foL_wtpctH2Osb[i] = proportion_H2O_to_wtpercent_H2O(p_H2OL)
    foL_H2Ob[i] = p_H2OL

    _, _, _, p_H2OL, p_H2MgSiO4wad = compositions(
        P=P2,
        T=T,
        X_Mg2SiO4=1.0,
        X_H2O=0.2,
        X_MgSiO3=1.0,
        Tm=Tm2,
        fo_polymorph=wad,
        H2MgSiO4_polymorph=hywad,
        MgSiO3_polymorph=maj,
    )

    wad_wtpctH2Os[i] = proportion_H2MgSiO4_to_wtpercent_H2O(p_H2MgSiO4wad)
    wadL_wtpctH2Os[i] = proportion_H2O_to_wtpercent_H2O(p_H2OL)
    wadL_H2O[i] = p_H2OL

    _, _, _, p_H2OL, p_H2MgSiO4wadb = compositions(
        P=P2b,
        T=T,
        X_Mg2SiO4=1.0,
        X_H2O=0.2,
        X_MgSiO3=1.0,
        Tm=Tm2b,
        fo_polymorph=wad,
        H2MgSiO4_polymorph=hywad,
        MgSiO3_polymorph=maj,
    )

    wad_wtpctH2Osb[i] = proportion_H2MgSiO4_to_wtpercent_H2O(p_H2MgSiO4wadb)
    wadL_wtpctH2Osb[i] = proportion_H2O_to_wtpercent_H2O(p_H2OL)
    wadL_H2Ob[i] = p_H2OL

    _, _, _, p_H2OL, p_H2MgSiO4ring = compositions(
        P=P3,
        T=T,
        X_Mg2SiO4=1.0,
        X_H2O=0.2,
        X_MgSiO3=1.0,
        Tm=Tm3,
        fo_polymorph=ring,
        H2MgSiO4_polymorph=hyring,
        MgSiO3_polymorph=maj,
    )

    ring_wtpctH2Os[i] = proportion_H2MgSiO4_to_wtpercent_H2O(p_H2MgSiO4ring)
    ringL_wtpctH2Os[i] = proportion_H2O_to_wtpercent_H2O(p_H2OL)
    ringL_H2O[i] = p_H2OL


data = np.genfromtxt(
    "data/13GPa_fo-H2O.dat", dtype=[float, float, float, (np.unicode_, 16)]
)
phases = list(set([d[3] for d in data]))
experiments = {
    ("melt + " + ph.replace("_", " "))
    .replace(" + liquid", ""): np.array(
        [[d[0], d[1], d[2]] for d in data if d[3] == ph]
    )
    .T
    for ph in phases
}
for phase, expts in sorted(experiments.items(), key=lambda item: item[0]):
    # plt.scatter(expts[2]/100., expts[0], label=phase) # on a 1-cation basis
    plt.scatter(expts[2] / (expts[1] + expts[2]), expts[0], label=phase)
# plt.plot(foL_H2O, temperatures-273.15, label=f'model melt ({P1/1.e9} GPa)', color='green', linestyle=':')
plt.plot(
    foL_H2Ob, temperatures - 273.15, label=f"model melt ({P1b/1.e9} GPa)", color="green"
)
plt.xlabel("x(Mg2SiO4) / (x(Mg2SiO4) + x(H2O)) (molar)")
plt.ylabel("Temperature (C)")
# plt.plot(wadL_H2O, temperatures-273.15, label=f'model melt ({P2/1.e9} GPa)', color='orange')
# plt.plot(wadL_H2Ob, temperatures-273.15, label=f'model melt ({P2b/1.e9} GPa)', color='orange', linestyle=':')
# plt.plot(ringL_H2O, temperatures-273.15, label=f'model melt ({P3/1.e9} GPa)', color='blue')
plt.legend()
plt.savefig("output_figures/melt_compositions_at_13GPa.pdf")
plt.show()


# Olivine
plt.plot(
    fo_wtpctH2Os,
    temperatures - 273.15,
    label=f"model melt ({P1/1.e9} GPa)",
    color="green",
    linestyle=":",
)
plt.plot(
    fo_wtpctH2Osb,
    temperatures - 273.15,
    label=f"model melt ({P1b/1.e9} GPa)",
    color="green",
)

plt.plot(foL_wtpctH2Os, temperatures - 273.15, color="green", linestyle=":")
plt.plot(foL_wtpctH2Osb, temperatures - 273.15, color="green")

Mdata = np.genfromtxt(
    "data/Mosenfelder_et_al_2006_ol_fluid.dat",
    dtype=[
        (np.unicode_, 16),
        float,
        float,
        float,
        (np.unicode_, 16),
        (np.unicode_, 16),
        (np.unicode_, 16),
        float,
    ],
)
d = np.array([[d[1], d[2], d[-1]] for d in Mdata if d[-2] == "yes" and d[1] == 8]).T
plt.errorbar(
    d[2] / 10000, d[1], xerr=d[2] / 10000 / 10, linestyle="None", color="green"
)
plt.scatter(
    d[2] / 10000, d[1], label="Olivine (Mosenfelder et al., 2006)", color="green"
)

Mdata = np.genfromtxt(
    "data/Smyth_et_al_2006_ol_melt.dat", dtype=[(np.unicode_, 16), float, float, float]
)
d = np.array([[d[1], d[2], d[3]] for d in Mdata]).T
plt.errorbar(
    d[2] / 10000, d[1], xerr=d[2] / 10000 / 10, linestyle="None", color="green"
)
plt.scatter(
    d[2] / 10000, d[1], label="Olivine (Smyth et al., 2006)", marker="P", color="green"
)

# Wadsleyite

plt.plot(
    wad_wtpctH2Os,
    temperatures - 273.15,
    label=f"model melt ({P2/1.e9} GPa)",
    color="orange",
    linestyle=":",
)
plt.plot(
    wad_wtpctH2Osb,
    temperatures - 273.15,
    label=f"model melt ({P2b/1.e9} GPa)",
    color="orange",
)

plt.plot(wadL_wtpctH2Os, temperatures - 273.15, color="orange", linestyle=":")
plt.plot(wadL_wtpctH2Osb, temperatures - 273.15, color="orange")

Ddata = np.genfromtxt(
    "data/Demouchy_et_al_2005_wad_melt.dat",
    dtype=[(np.unicode_, 16), float, float, float, float, float],
)
d = np.array([list(d)[1:] for d in Ddata]).T
plt.errorbar(d[3], d[1], xerr=d[4], linestyle="None", color="orange")
plt.scatter(
    d[3], d[1], label="Wadsleyite (Demouchy et al., 2005)", marker="P", color="orange"
)

Ldata = np.genfromtxt(
    "data/Litasov_et_al_2011_wad_melt.dat",
    dtype=[(np.unicode_, 16), float, float, float, float, float, float],
)
d = np.array([list(d)[1:] for d in Ldata]).T
plt.errorbar(d[2], d[1], xerr=d[3], linestyle="None", color="orange")
plt.scatter(d[2], d[1], label="Wadsleyite (Litasov et al., 2011)", color="orange")

# Ringwoodite
plt.plot(
    ring_wtpctH2Os,
    temperatures - 273.15,
    label=f"model melt ({P3/1.e9} GPa)",
    color="blue",
)
plt.plot(ringL_wtpctH2Os, temperatures - 273.15, color="blue")

Odata = np.genfromtxt(
    "data/Ohtani_et_al_2000_rw_melt.dat",
    dtype=[(np.unicode_, 16), float, float, float, float],
)
d = np.array([[d[1], d[2], d[3], d[4]] for d in Odata]).T
plt.errorbar(d[2], d[1], xerr=d[3], linestyle="None", color="blue")
plt.scatter(d[2], d[1], label="Ringwoodite (Ohtani et al., 2000)", color="blue")

plt.xlabel("Water concentration (wt %)")
plt.ylabel("Temperature (C)")
plt.xlim(0.0, 10.0)
plt.ylim(0.0, 2100.0)
plt.legend()

plt.savefig("output_figures/water_contents_fo_wad_rw.pdf")
plt.show()
