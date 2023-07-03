import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import os.path
import sys

sys.path.insert(1, os.path.abspath("../.."))

import burnman
from burnman import Mineral
from burnman.minerals import SLB_2011
from burnman.minerals.Pitzer_Sterner_1994 import H2O_Pitzer_Sterner
from burnman.tools.chemistry import dictionarize_formula, formula_mass


P_ref = 17.5e9


def fitting_function(m_name, formula, fn_type, V_args=None, G_args=None):
    molar_mass = formula_mass(formula)
    n = sum(formula.values())

    def make_mineral(V_params, G_params):
        V_0, K_0, Kp_0, Kdp_0, a_0, TE_0 = V_params
        H_0, S_0, Cpa_0, Cpb_0, Cpc_0, Cpd_0 = G_params

        m = burnman.Mineral(
            {
                "name": m_name,
                "equation_of_state": "mod_hp_tmt",
                "T_0": 298.15,
                "Pref": P_ref,
                "V_0": V_0,
                "K_0": K_0,
                "Kprime_0": Kp_0,
                "Kdprime_0": Kdp_0,
                "a_0": a_0,
                "T_einstein": TE_0,
                "H_Pref": H_0,
                "S_Pref": S_0,
                "Cp_Pref": np.array([Cpa_0, Cpb_0, Cpc_0, Cpd_0]),
                "molar_mass": molar_mass,
                "n": n,
                "formula": formula,
            }
        )
        return m

    def fit_V(PT_data, V_0, K_0, Kp_0, Kdp_0, a_0, TE_0):
        V_params = [V_0, K_0, Kp_0, Kdp_0, a_0, TE_0]
        G_params = [
            -1442310.0,
            62.6,
            1.61546581e02,
            -3.31714290e-03,
            -3.57533814e06,
            -1.11254791e03,
        ]
        m = make_mineral(V_params, G_params)

        pressures = PT_data.T[0]
        temperatures = PT_data.T[1]

        volumes = m.evaluate(["V"], pressures, temperatures)[0]
        return volumes

    def fit_Gref(T_data, H_0, S_0, Cpa_0, Cpb_0, Cpc_0, Cpd_0):
        V_params = V_args
        G_params = [H_0, S_0, Cpa_0, Cpb_0, Cpc_0, Cpd_0]
        m = make_mineral(V_params, G_params)

        temperatures = T_data
        pressures = T_data * 0.0 + P_ref

        volumes = m.evaluate(["gibbs"], pressures, temperatures)[0]
        return volumes

    if fn_type == "V":
        return fit_V
    elif fn_type == "Gref":
        return fit_Gref
    elif fn_type == "mineral":
        return make_mineral(V_args, G_args)
    else:
        raise Exception("Fitting type not recognised")


pressures0 = np.linspace(5.0e9, 30.0e9, 101)
temperatures0 = 1600.0 + 0.0 * pressures0

temperatures1 = np.linspace(1000.0, 2200.0, 101)
pressures1 = P_ref + 0.0 * temperatures1

Pstack = np.hstack((pressures0, pressures1))
Tstack = np.hstack((temperatures0, temperatures1))

PTstack = np.vstack((Pstack, Tstack)).T

V00 = 1.5e-5
K00 = 100.0e9  # 24.e9 better starting guess for H2O
Kp00 = 4.0
Kdp00 = -Kp00 / K00
a00 = 1.0e-5
Tein00 = 200.0

for m, Pshift, Vshift in [  # [SLB_2011.mg_wadsleyite(), 0., 0.],
    # [SLB_2011.fe_wadsleyite(), 0., 0.],
    [
        burnman.CombinedMineral([SLB_2011.mg_majorite()], [0.25], name="Mg_majorite"),
        0.0,
        0.0,
    ],
    [H2O_Pitzer_Sterner(), -1.5e9, -2.5e-6],
]:
    Vstack = m.evaluate(["V"], Pstack - Pshift, Tstack)[0] + Vshift

    Vfn = fitting_function(m.name, m.formula, "V")
    poptV, pcovV = curve_fit(Vfn, PTstack, Vstack, [V00, K00, Kp00, Kdp00, a00, Tein00])

    gibbs1, S1 = m.evaluate(["gibbs", "S"], pressures1, temperatures1)
    Gfn = fitting_function(m.name, m.formula, "Gref", poptV)
    poptG, pcovG = curve_fit(Gfn, temperatures1, gibbs1, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    """
    fig = plt.figure()
    ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
    ax[0].plot(temperatures1, -np.gradient(gibbs1, temperatures1, edge_order=2))
    ax[0].plot(temperatures1, S1)
    ax[0].plot(temperatures1, -np.gradient(Gfn(temperatures1, *poptG), temperatures1, edge_order=2))


    ax[1].plot(pressures0, Vstack[0:len(pressures0)])
    ax[1].plot(pressures0, Vfn(PTstack[0:len(pressures0)], *poptV))
    plt.show()
    """

    m_new = fitting_function(m.name, m.formula, "mineral", poptV, poptG)
    print(m_new.params)
