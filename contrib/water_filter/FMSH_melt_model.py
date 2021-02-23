import numpy as np
from model_parameters import R, ol, wad, ring, lm, melt
from model_parameters import olwad, wadring, ringlm
from model_parameters import liq_sp
from scipy.special import expi
import matplotlib.pyplot as plt


def _li(x):
    """
    The logarithmic integral
    """
    return expi(np.log(x))


def melting_temperature(solid, P):

    f = melt['b']*P + melt['c']

    return (((melt['E'] - solid['E']) + P * (melt['V'] - solid['V'])
             + (melt['a']/melt['b'])*(_li(f) - _li(melt['c'])))
            / (melt['S'] - solid['S']))


def ax_melt(T, Tm, S_fusion):
    """
    Activity-composition model for a hydrous silicate melt
    The following expressions are taken from Silver and Stolper (1985),
    and assume an ideal solution of oxygens (the silicate network is ignored),
    OH and H2O molecules.

    The function takes the temperature and the melting temperature and
    entropy of fusion of the anhydrous silicate (on an "r"-oxygen basis),
    and returns the activity and proportion of water in the melt
    (where the proportion of silicate in the melt is (1 - X_H2O), considered
    on the same "r"-oxygen basis).
    """
    K1 = np.exp((liq_sp['H_poly'] - liq_sp['S_poly']*T)/(R*T))

    X_O = np.exp(S_fusion*(T - Tm)/(liq_sp['r']*R*T))  # equation 13
    X_B = (1. - X_O
           + 1./4.*(K1 * X_O - np.sqrt((K1 * X_O)**2.
                                       + 4.*K1*X_O
                                       - 4.*K1*X_O**2.)))  # equation 14
    X_H2O = X_B - ((0.5 - np.sqrt(0.25 - ((K1 - 4.)/K1)*(X_B - X_B**2.)))
                   / ((K1 - 4.)/K1))  # equation 5.2

    X_OH = 2.*(X_B - X_H2O)
    S_conf = -R*((X_OH + X_H2O) * np.log(X_H2O)
                 + (X_OH + X_O) * np.log(X_O))

    a_H2O = X_H2O

    # X_B and S_conf (the bulk liquid H2O content and configurational entropy)
    # are defined on the basis of 1 oxygen,
    # so a conversion back to an "r" oxygen basis is necessary
    p_H2O = X_B/(X_B + (1. - X_B)/liq_sp['r'])
    n_O = p_H2O + (1. - p_H2O)*liq_sp['r']
    S_conf = S_conf * n_O

    return (a_H2O, p_H2O, S_conf)


def solve_quadratic(a, b, c, sgn):
    """
    solves the quadratic equation
    sgn should be either +1 or -1, depending on the root
    the user wishes to return
    """
    return (-b + sgn*np.sqrt(b*b - 4.*a*c))/(2.*a)


def partition(p_Fe_bulk, f_liq, KD):
    """
    p_Fe_bulk is the molar ratio Fe/(Mg+Fe) in the bulk
    f_liq is the fraction of combined Mg and Fe in the liq
    (relative to the bulk)
    KD is the partition coefficient (p_Feliq*p_Mgsol)/(p_Fesol*p_Mgliq)
    """
    a = -f_liq*(KD - 1.)
    b = (KD - 1.)*p_Fe_bulk - (1. - f_liq)*KD - f_liq
    c = p_Fe_bulk

    if a != 0.:
        p_Fe_liq = solve_quadratic(a, b, c, -1.)
    else:
        p_Fe_liq = -c / b
    p_Fe_sol = (p_Fe_bulk - f_liq*p_Fe_liq)/(1. - f_liq)

    return (p_Fe_liq, p_Fe_sol)


def melt_excess_volume(P, T, X_Mg2SiO4, X_Fe2SiO4, X_H2O):
    p_H2OL = X_H2O / (X_Mg2SiO4 + X_Fe2SiO4 + X_H2O)
    p_Mg2SiO4L = 1. - p_H2OL
    f = melt['b']*T + melt['c']
    V_xs_melt = p_Mg2SiO4L * (melt['V'] + melt['a']/np.log(f))
    return V_xs_melt


def solid_excess_volume(P, T, X_ol_comps, X_MgSiO3, X_H2O, phases, f_tr):
    x_fo0 = (1. - f_tr)*(X_H2O + X_ol_comps)
    x_fo1 = f_tr*(X_H2O + X_ol_comps)
    # x_maj = X_MgSiO3 - X_H2O

    p_H2MgSiO4fo0 = np.exp(-(phases[0]['hyE']
                             - T*phases[0]['hyS']
                             + P*phases[0]['hyV']) / (R*T))

    p_H2MgSiO4fo1 = np.exp(-(phases[1]['hyE']
                             - T*phases[1]['hyS']
                             + P*phases[1]['hyV']) / (R*T))

    r = p_H2MgSiO4fo1 / p_H2MgSiO4fo0
    p_H2MgSiO4fo0 = X_H2O / (x_fo0 + x_fo1*r)
    p_H2MgSiO4fo1 = X_H2O / (x_fo1 + x_fo0/r)
    p_Mg2SiO4fo0 = 1. - p_H2MgSiO4fo0
    p_Mg2SiO4fo1 = 1. - p_H2MgSiO4fo1

    V_xs_solid = ((1. - f_tr) * x_fo0 * (p_Mg2SiO4fo0 * phases[0]['V']
                                         + 0.5 * p_H2MgSiO4fo0 * phases[0]['hyV'])
                  + f_tr * x_fo1 * (p_Mg2SiO4fo1 * phases[1]['V']
                                    + 0.5 * p_H2MgSiO4fo1 * phases[1]['hyV']))
    return V_xs_solid


def one_phase_eqm(P, T, X_Mg2SiO4, X_Fe2SiO4, X_MgSiO3, X_H2O, phase):
    p_Fe_bulk = 2. * X_Fe2SiO4 / (2.*(X_Mg2SiO4 + X_Fe2SiO4) + X_MgSiO3)

    # Calculate the melting temperature of the olivine polymorph
    Tm = melting_temperature(phase, P)

    # Calculate the proportions of the melt and olivine polymorph endmembers
    # in their respective phases.
    S_fusion = melt['S'] - phase['S']
    a_H2OL, p_H2OL, S_conf_melt = ax_melt(T, Tm, S_fusion)

    p_H2MgSiO4fo = a_H2OL*np.exp(-(phase['hyE']
                                   - T*phase['hyS']
                                   + P*phase['hyV']) / (R*T))
    # Dependent fractions
    p_Mg2SiO4L = 1. - p_H2OL
    p_Mg2SiO4fo = 1. - p_H2MgSiO4fo

    # Calculate the amounts of the phases
    X_ol_comps = X_Mg2SiO4 + X_Fe2SiO4
    x_fo = (X_H2O*p_Mg2SiO4L - X_ol_comps*p_H2OL)/(p_H2MgSiO4fo - p_H2OL)
    x_maj = X_MgSiO3 - p_H2MgSiO4fo*x_fo
    x_L = (-X_H2O*p_Mg2SiO4fo + X_ol_comps*p_H2MgSiO4fo)/(p_H2MgSiO4fo - p_H2OL)

    if x_L <= 0.:
        # Composition is completely solid
        x_fo = X_H2O + X_ol_comps
        x_maj = X_MgSiO3 - X_H2O
        x_L = 0.

        p_H2MgSiO4fo = X_H2O / x_fo
        p_Mg2SiO4fo = 1. - p_H2MgSiO4fo

    elif x_fo <= 0.:
        # The olivine polymorph has been exhausted
        x_fo = 0.
        x_maj = X_MgSiO3
        x_L = X_ol_comps + X_H2O

        p_H2OL = X_H2O / x_L
        p_Mg2SiO4L = 1. - p_H2OL

    # checks:
    assert np.abs(X_ol_comps - x_fo*p_Mg2SiO4fo - x_L * p_Mg2SiO4L) < 1.e-10
    assert np.abs(X_MgSiO3 - x_fo*p_H2MgSiO4fo - x_maj) < 1.e-10
    assert np.abs(X_H2O - x_fo*p_H2MgSiO4fo - x_L * p_H2OL) < 1.e-10

    # Calculate the iron partitioning between the melt and the solid
    # a) Calculate the total amount of
    # divalent cation (Mg+Fe) in the solid and melt
    x_D_melt = 2. * x_L * p_Mg2SiO4L
    x_D_solid = x_fo * (2. * p_Mg2SiO4fo + p_H2MgSiO4fo) + x_maj

    # b) Calculate the fraction of divalent cation in the liquid
    f_divalent_liq = x_D_melt / (x_D_melt + x_D_solid)

    # c) Calculate the KD for the phase of interest
    KD = np.exp(-(phase['feE']
                  - T*phase['feS']
                  + P*phase['feV']) / (R*T))

    # d) calculate the partitioning
    p_Fe_liq, p_Fe_sol = partition(p_Fe_bulk, f_divalent_liq, KD)

    # Return the proportion of melt,
    # the composition of the solid and the melt,
    # and the excess energy, entropy, volume, and d2V/dP2 (beta_T*V)
    X_H2O_melt = p_H2OL
    X_MgSiO3_melt = 0.
    X_Fe2SiO4_melt = p_Fe_liq * p_Mg2SiO4L
    X_Mg2SiO4_melt = (1. - p_Fe_liq) * p_Mg2SiO4L

    X_total_solid = x_maj + x_fo * (2.*p_H2MgSiO4fo + p_Mg2SiO4fo)
    X_H2O_solid = (x_fo * p_H2MgSiO4fo) / X_total_solid
    X_MgSiO3_solid = (x_fo * p_H2MgSiO4fo + x_maj) / X_total_solid
    X_Fe2SiO4_solid = ((x_D_solid * p_Fe_sol)/2.) / X_total_solid
    X_Mg2SiO4_solid = ((x_fo * p_Mg2SiO4fo) / X_total_solid) - X_Fe2SiO4_solid

    f_melt = x_L/(x_L + X_total_solid)
    assert np.abs(f_melt - (X_H2O - X_H2O_solid)
                  / (X_H2O_melt - X_H2O_solid)) < 1.e-5

    f = melt['b']*T + melt['c']

    S_xs_melt = p_Mg2SiO4L * melt['S'] + S_conf_melt
    V_xs_melt = p_Mg2SiO4L * (melt['V'] + melt['a']/np.log(f))
    dVdP_xs_melt = -p_Mg2SiO4L * ((melt['a'] * melt['b'])/(f * np.log(f)**2.))

    S_xs_solid = x_fo * (p_Mg2SiO4fo * phase['S']
                         + 0.5 * p_H2MgSiO4fo * phase['hyS']) / X_total_solid
    V_xs_solid = x_fo * (p_Mg2SiO4fo * phase['V']
                         + 0.5 * p_H2MgSiO4fo * phase['hyV']) / X_total_solid
    dVdP_xs_solid = 0.

    return {'molar_fraction_melt': f_melt,
            'X_H2O_melt': X_H2O_melt,
            'X_MgSiO3_melt': X_MgSiO3_melt,
            'X_Fe2SiO4_melt': X_Fe2SiO4_melt,
            'X_Mg2SiO4_melt': X_Mg2SiO4_melt,
            'X_H2O_solid': X_H2O_solid,
            'X_MgSiO3_solid': X_MgSiO3_solid,
            'X_Fe2SiO4_solid': X_Fe2SiO4_solid,
            'X_Mg2SiO4_solid': X_Mg2SiO4_solid,
            'S_xs_melt': S_xs_melt,
            'V_xs_melt': V_xs_melt,
            'dVdP_xs_melt': dVdP_xs_melt,
            'S_xs_solid': S_xs_solid,
            'V_xs_solid': V_xs_solid,
            'dVdP_xs_solid': dVdP_xs_solid}


def two_phase_eqm(P, T, X_Mg2SiO4, X_Fe2SiO4, X_MgSiO3, X_H2O, phases, f_tr):
    p_Fe_bulk = 2. * X_Fe2SiO4 / (2.*(X_Mg2SiO4 + X_Fe2SiO4) + X_MgSiO3)

    # Calculate the melting temperature of the olivine polymorph
    Tm = ((1. - f_tr)*melting_temperature(phases[0], P)
          + f_tr*melting_temperature(phases[1], P))

    # Calculate the proportions of the melt and olivine polymorph endmembers
    # in their respective phases.
    S_fusion = melt['S'] - (1. - f_tr)*phases[0]['S'] - f_tr*phases[1]['S']
    a_H2OL, p_H2OL, S_conf_melt = ax_melt(T, Tm, S_fusion)

    p_H2MgSiO4fo0 = a_H2OL*np.exp(-(phases[0]['hyE']
                                    - T*phases[0]['hyS']
                                    + P*phases[0]['hyV']) / (R*T))

    p_H2MgSiO4fo1 = a_H2OL*np.exp(-(phases[1]['hyE']
                                    - T*phases[1]['hyS']
                                    + P*phases[1]['hyV']) / (R*T))

    # Dependent fractions
    p_Mg2SiO4L = 1. - p_H2OL
    p_Mg2SiO4fo0 = 1. - p_H2MgSiO4fo0
    p_Mg2SiO4fo1 = 1. - p_H2MgSiO4fo1

    # Calculate the amounts of the phases
    X_ol_comps = X_Mg2SiO4 + X_Fe2SiO4
    p_H2MgSiO4fo_eff = (1. - f_tr)*p_H2MgSiO4fo0 + f_tr*p_H2MgSiO4fo1

    x_fo0 = ((1. - f_tr)*(X_H2O*p_Mg2SiO4L - X_ol_comps*p_H2OL)
             / (p_H2MgSiO4fo_eff - p_H2OL))
    x_fo1 = (f_tr*(X_H2O*p_Mg2SiO4L - X_ol_comps*p_H2OL)
             / (p_H2MgSiO4fo_eff - p_H2OL))
    x_maj = X_MgSiO3 - p_H2MgSiO4fo0*x_fo0 - p_H2MgSiO4fo1*x_fo1
    x_L = ((-X_H2O*(1. - p_H2MgSiO4fo_eff) + X_ol_comps*p_H2MgSiO4fo_eff)
           / (p_H2MgSiO4fo_eff - p_H2OL))

    if x_L <= 0.:
        # Composition is completely solid
        x_fo0 = (1. - f_tr)*(X_H2O + X_ol_comps)
        x_fo1 = f_tr*(X_H2O + X_ol_comps)
        x_maj = X_MgSiO3 - X_H2O
        x_L = 0.

        r = p_H2MgSiO4fo1 / p_H2MgSiO4fo0
        p_H2MgSiO4fo0 = X_H2O / (x_fo0 + x_fo1*r)
        p_H2MgSiO4fo1 = X_H2O / (x_fo1 + x_fo0/r)
        p_Mg2SiO4fo0 = 1. - p_H2MgSiO4fo0
        p_Mg2SiO4fo1 = 1. - p_H2MgSiO4fo1

    elif x_fo0 <= 0.:
        # The olivine polymorphs have been exhausted
        x_fo0 = 0.
        x_fo1 = 0.
        x_maj = X_MgSiO3
        x_L = X_ol_comps + X_H2O

        p_H2OL = X_H2O / x_L
        p_Mg2SiO4L = 1. - p_H2OL

    # checks:
    assert np.abs(X_ol_comps - x_fo0*p_Mg2SiO4fo0 - x_fo1*p_Mg2SiO4fo1 - x_L * p_Mg2SiO4L) < 1.e-10
    assert np.abs(X_MgSiO3 - x_fo0*p_H2MgSiO4fo0 - x_fo1*p_H2MgSiO4fo1 - x_maj) < 1.e-10
    assert np.abs(X_H2O - x_fo0*p_H2MgSiO4fo0 - x_fo1*p_H2MgSiO4fo1 - x_L * p_H2OL) < 1.e-10

    # Calculate the iron partitioning between the melt and the solid
    # a) Calculate the total amount of
    # divalent cation (Mg+Fe) in the solid and melt
    x_D_melt = 2. * x_L * p_Mg2SiO4L
    x_D_solid = (x_fo0 * (2. * p_Mg2SiO4fo0 + p_H2MgSiO4fo0)
                 + x_fo1 * (2. * p_Mg2SiO4fo1 + p_H2MgSiO4fo1)
                 + x_maj)

    # b) Calculate the fraction of divalent cation in the liquid
    f_divalent_liq = x_D_melt / (x_D_melt + x_D_solid)

    # c) Calculate the KD for the phases of interest
    KD0 = np.exp(-(phases[0]['feE']
                   - T*phases[0]['feS']
                   + P*phases[0]['feV']) / (R*T))

    KD1 = np.exp(-(phases[1]['feE']
                   - T*phases[1]['feS']
                   + P*phases[1]['feV']) / (R*T))

    # d) Linearly average the KDs
    # (correct in the limit that the solids do not strongly partition iron)
    KD = (1. - f_tr) * KD0 + f_tr * KD1

    # e) calculate the partitioning
    p_Fe_liq, p_Fe_sol = partition(p_Fe_bulk, f_divalent_liq, KD)

    # Return the proportion of melt,
    # the composition of the solid and the melt,
    # and the excess energy, entropy, volume, and d2V/dP2 (beta_T*V)
    X_H2O_melt = p_H2OL
    X_MgSiO3_melt = 0.
    X_Fe2SiO4_melt = p_Fe_liq * p_Mg2SiO4L
    X_Mg2SiO4_melt = (1. - p_Fe_liq) * p_Mg2SiO4L

    X_total_solid = x_maj + (x_fo0 * (2.*p_H2MgSiO4fo0 + p_Mg2SiO4fo0)
                             + x_fo1 * (2.*p_H2MgSiO4fo1 + p_Mg2SiO4fo1))

    X_H2O_solid = (x_fo0 * p_H2MgSiO4fo0
                   + x_fo1 * p_H2MgSiO4fo1) / X_total_solid
    X_MgSiO3_solid = (x_fo0 * p_H2MgSiO4fo0
                      + x_fo1 * p_H2MgSiO4fo1 + x_maj) / X_total_solid
    X_Fe2SiO4_solid = ((x_D_solid * p_Fe_sol)/2.) / X_total_solid
    X_Mg2SiO4_solid = (((x_fo0 * p_Mg2SiO4fo0
                         + x_fo1 * p_Mg2SiO4fo1) / X_total_solid)
                       - X_Fe2SiO4_solid)

    f_melt = x_L/(x_L + X_total_solid)

    # Check that the bulk composition is conserved
    assert np.abs(f_melt*X_Mg2SiO4_melt + (1. - f_melt)*X_Mg2SiO4_solid - X_Mg2SiO4) < 1.e-10
    assert np.abs(f_melt*X_Fe2SiO4_melt + (1. - f_melt)*X_Fe2SiO4_solid - X_Fe2SiO4) < 1.e-10
    assert np.abs(f_melt*X_MgSiO3_melt + (1. - f_melt)*X_MgSiO3_solid - X_MgSiO3) < 1.e-10
    assert np.abs(f_melt*X_H2O_melt + (1. - f_melt)*X_H2O_solid - X_H2O) < 1.e-10

    S_conf_fo0 = -R*(p_Mg2SiO4fo0 * np.log(p_Mg2SiO4fo0)
                     + p_H2MgSiO4fo0 * np.log(p_H2MgSiO4fo0))
    S_conf_fo1 = -R*(p_Mg2SiO4fo1 * np.log(p_Mg2SiO4fo1)
                     + p_H2MgSiO4fo1 * np.log(p_H2MgSiO4fo1))

    f = melt['b']*T + melt['c']

    S_xs_melt = p_Mg2SiO4L * melt['S'] + S_conf_melt
    V_xs_melt = p_Mg2SiO4L * (melt['V'] + melt['a']/np.log(f))
    dVdP_xs_melt = -p_Mg2SiO4L * ((melt['a'] * melt['b'])/(f * np.log(f)**2.))

    S_xs_solid = (x_fo0 * (p_Mg2SiO4fo0 * phases[0]['S']
                           + 0.5 * p_H2MgSiO4fo0 * phases[0]['hyS']
                           + S_conf_fo0)
                  + x_fo1 * (p_Mg2SiO4fo1 * phases[1]['S']
                             + 0.5 * p_H2MgSiO4fo1 * phases[1]['hyS']
                             + S_conf_fo1)) / X_total_solid
    V_xs_solid = (x_fo0 * (p_Mg2SiO4fo0 * phases[0]['V']
                           + 0.5 * p_H2MgSiO4fo0 * phases[0]['hyV'])
                  + x_fo1 * (p_Mg2SiO4fo1 * phases[1]['V']
                             + 0.5 * p_H2MgSiO4fo1 * phases[1]['hyV'])) / X_total_solid
    dVdP_xs_solid = 0.

    return {'molar_fraction_melt': f_melt,
            'X_Mg2SiO4_melt': X_Mg2SiO4_melt,
            'X_Fe2SiO4_melt': X_Fe2SiO4_melt,
            'X_MgSiO3_melt': X_MgSiO3_melt,
            'X_H2O_melt': X_H2O_melt,
            'X_Mg2SiO4_solid': X_Mg2SiO4_solid,
            'X_Fe2SiO4_solid': X_Fe2SiO4_solid,
            'X_MgSiO3_solid': X_MgSiO3_solid,
            'X_H2O_solid': X_H2O_solid,
            'S_xs_melt': S_xs_melt,
            'V_xs_melt': V_xs_melt,
            'dVdP_xs_melt': dVdP_xs_melt,
            'S_xs_solid': S_xs_solid,
            'V_xs_solid': V_xs_solid,
            'dVdP_xs_solid': dVdP_xs_solid}


def stable_phases(P, T):
    # Determine from the pressure and temperature
    # which assemblage(s) is/are stable at a given point.
    P_olwad = -(olwad['Delta_E'] - T*olwad['Delta_S']) / olwad['Delta_V']
    P_wadring = -(wadring['Delta_E'] - T*wadring['Delta_S']) / wadring['Delta_V']
    P_ringlm = -(ringlm['Delta_E'] - T*ringlm['Delta_S']) / ringlm['Delta_V']

    half_P_interval_olwad = olwad['halfPint0'] + T*olwad['dhalfPintdT']
    half_P_interval_wadring = wadring['halfPint0'] + T*wadring['dhalfPintdT']
    half_P_interval_ringlm = ringlm['halfPint0'] + T*ringlm['dhalfPintdT']

    if P < (P_olwad - half_P_interval_olwad):
        return [[ol], 0.]

    elif P < (P_olwad + half_P_interval_olwad):
        f_tr = ((P - (P_olwad - half_P_interval_olwad))
                / (2. * half_P_interval_olwad))
        return [[ol, wad], f_tr]

    elif P < (P_wadring - half_P_interval_wadring):
        return [[wad], 0.]

    elif P < (P_wadring + half_P_interval_wadring):
        f_tr = ((P - (P_wadring - half_P_interval_wadring))
                / (2. * half_P_interval_wadring))
        return [[wad, ring], f_tr]

    elif P < (P_ringlm - half_P_interval_ringlm):
        return [[ring], 0.]

    elif P < (P_ringlm + half_P_interval_ringlm):
        f_tr = ((P - (P_ringlm - half_P_interval_ringlm))
                / (2. * half_P_interval_ringlm))
        return [[ring, lm], f_tr]

    else:
        return [[lm], 0.]


def equilibrate(P, T, X_Mg2SiO4, X_Fe2SiO4, X_MgSiO3, X_H2O):
    assert (X_Mg2SiO4 + X_Fe2SiO4 + X_MgSiO3 + X_H2O - 1.) < 1.e-12

    phases, f_tr = stable_phases(P, T)

    if len(phases) == 1:
        props = one_phase_eqm(P, T, X_Mg2SiO4, X_Fe2SiO4, X_MgSiO3, X_H2O,
                              phases[0])
    else:
        props = two_phase_eqm(P, T, X_Mg2SiO4, X_Fe2SiO4, X_MgSiO3, X_H2O,
                              phases, f_tr)

    return props


if __name__ == '__main__':
    #                [f_melt,
    #                 X_H2O_melt, X_MgSiO3_melt,
    #                 X_Fe2SiO4_melt, X_Mg2SiO4_melt,
    #                 X_H2O_solid, X_MgSiO3_solid,
    #                 X_Fe2SiO4_solid, X_Mg2SiO4_solid,
    #                 S_xs_melt, V_xs_melt, dVdP_xs_melt,
    #                 S_xs_solid, V_xs_solid, dVdP_xs_solid])
    pressures = np.linspace(6.e9, 25.e9, 1001)
    f_melts = np.empty_like(pressures)
    X_H2O_melts = np.empty_like(pressures)
    X_H2O_solids = np.empty_like(pressures)
    KD = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        T = 1600.
        X_Mg2SiO4, X_Fe2SiO4, X_MgSiO3, X_H2O = [0.65, 0.05, 0.1, 0.2]
        eqm = equilibrate(P, T, X_Mg2SiO4, X_Fe2SiO4, X_MgSiO3, X_H2O)
        f_melts[i] = eqm['molar_fraction_melt']
        X_H2O_melts[i] = eqm['X_H2O_melt']
        X_H2O_solids[i] = eqm['X_H2O_solid']
        KD[i] = (2.*eqm['X_Fe2SiO4_solid']*(2. * eqm['X_Mg2SiO4_melt'] + eqm['X_MgSiO3_melt'])
                 / (2.*eqm['X_Fe2SiO4_melt']*(2. * eqm['X_Mg2SiO4_solid'] + eqm['X_MgSiO3_solid'])))

    plt.plot(pressures/1.e9, f_melts, label='molar_fraction melt')
    plt.plot(pressures/1.e9, X_H2O_solids, label='X_H2O solid')
    plt.plot(pressures/1.e9, X_H2O_melts, label='X_H2O melt')
    plt.plot(pressures/1.e9, KD, label='K_D')
    plt.legend()
    plt.show()
