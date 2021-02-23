import numpy as np
from endmember_eos import thermodynamic_properties
from model_parameters import Mg2SiO4_params, Fe2SiO4_params, MgSiO3_params, H2O_params
from FMSH_melt_model import melt_excess_volume, solid_excess_volume, stable_phases
from FMSH_melt_model import one_phase_eqm, two_phase_eqm


def phase_properties(P, T, X_Mg2SiO4, X_Fe2SiO4, X_MgSiO3, X_H2O, XXXX_excesses):

    fractions = [X_Mg2SiO4, X_Fe2SiO4, X_MgSiO3, X_H2O]

    properties = [thermodynamic_properties(P, T, Mg2SiO4_params),
                  thermodynamic_properties(P, T, Fe2SiO4_params),
                  thermodynamic_properties(P, T, MgSiO3_params),
                  thermodynamic_properties(P, T, H2O_params)]

    molar_masses = [Mg2SiO4_params['molar_mass'],
                    Fe2SiO4_params['molar_mass'],
                    MgSiO3_params['molar_mass'],
                    H2O_params['molar_mass']]

    alphas = [prp['alpha'] for prp in properties]
    volumes = [prp['V'] for prp in properties]
    beta_Ts = [prp['beta_T'] for prp in properties]
    C_ps = [prp['molar_C_p'] for prp in properties]

    V_molar = np.sum([fractions[i]*volumes[i] for i in range(4)])
    M_molar = np.sum([fractions[i]*molar_masses[i] for i in range(4)])
    return {'molar mass': M_molar,
            'V': V_molar,
            'alpha': 1./V_molar*np.sum([fractions[i]*alphas[i]*volumes[i] for i in range(4)]),
            'rho': M_molar/V_molar,
            'beta_T': 1./V_molar*np.sum([fractions[i]*beta_Ts[i]*volumes[i] for i in range(4)]),
            'molar_C_p': np.sum([fractions[i]*C_ps[i] for i in range(4)]),
            'C_p_per_kilogram': np.sum([fractions[i]*C_ps[i] for i in range(2)])/M_molar}


def evaluate(P, T,
             X_Mg2SiO4_solid, X_Fe2SiO4_solid, X_MgSiO3_solid, X_H2O_solid,
             X_Mg2SiO4_melt, X_Fe2SiO4_melt, X_H2O_melt,
             porosity):

    # 1) Calculate reference component properties
    c_prps = [thermodynamic_properties(P, T, Mg2SiO4_params),
              thermodynamic_properties(P, T, Fe2SiO4_params),
              thermodynamic_properties(P, T, MgSiO3_params),
              thermodynamic_properties(P, T, H2O_params)]

    # 2) Calculate which phases are stable
    phases, f_tr = stable_phases(P, T)

    # 3) Calculate molar fraction of melt from volume fraction (porosity)
    # The solid is allowed to react with itself to form another
    # (metastable) solid, but solid is not allowed to melt,
    # melt is not allowed to solidify and melt and solid are not allowed to
    # react with each other until after this step.
    X_ol_solid = X_Mg2SiO4_solid + X_Fe2SiO4_solid
    V_xs_solid = solid_excess_volume(P, T, X_ol_solid, X_MgSiO3_solid, X_H2O_solid, phases, f_tr)
    V_xs_melt = melt_excess_volume(P, T, X_Mg2SiO4_melt, X_Fe2SiO4_melt, X_H2O_melt)

    moles_solid = ((X_Mg2SiO4_solid * c_prps[0]['V']
                   + X_Fe2SiO4_solid * c_prps[1]['V']
                   + X_MgSiO3_solid * c_prps[2]['V']
                   + X_H2O_solid * c_prps[3]['V']) + V_xs_solid) / (1. - porosity)

    moles_melt = ((X_Mg2SiO4_melt * c_prps[0]['V']
                   + X_Fe2SiO4_melt * c_prps[1]['V']
                   + X_H2O_melt * c_prps[3]['V']) + V_xs_melt) / porosity

    f_melt = moles_melt / (moles_melt + moles_solid)

    # 4) Calculate the bulk composition
    X_Mg2SiO4_bulk = X_Mg2SiO4_solid * (1. - f_melt) + X_Mg2SiO4_melt * f_melt
    X_Fe2SiO4_bulk = X_Fe2SiO4_solid * (1. - f_melt) + X_Fe2SiO4_melt * f_melt
    X_MgSiO3_bulk = X_MgSiO3_solid * (1. - f_melt)
    X_H2O_bulk = X_H2O_solid * (1. - f_melt) + X_H2O_melt * f_melt

    # 5) Calculate the equilibrium phase amounts and compositions
    # 6) Calculate the new solid and melt compositions
    # 7) Calculate the new excess properties
    assert (X_Mg2SiO4_bulk + X_Fe2SiO4_bulk
            + X_MgSiO3_bulk + X_H2O_bulk - 1.) < 1.e-12

    if len(phases) == 1:
        prps = one_phase_eqm(P, T,
                             X_Mg2SiO4_bulk, X_Fe2SiO4_bulk,
                             X_MgSiO3_bulk, X_H2O_bulk,
                             phases[0])
    else:
        prps = two_phase_eqm(P, T,
                             X_Mg2SiO4_bulk, X_Fe2SiO4_bulk,
                             X_MgSiO3_bulk, X_H2O_bulk,
                             phases, f_tr)

    # 8) Calculate the new porosity
    V_solid = (((prps['X_Mg2SiO4_solid'] * c_prps[0]['V']
                + prps['X_Fe2SiO4_solid'] * c_prps[1]['V']
                + prps['X_MgSiO3_solid'] * c_prps[2]['V']
                + prps['X_H2O_solid'] * c_prps[3]['V']) + prps['V_xs_solid'])) * (1. - prps['molar_fraction_melt'])

    V_melt_molar = (prps['X_Mg2SiO4_melt'] * c_prps[0]['V']
                    + prps['X_Fe2SiO4_melt'] * c_prps[1]['V']
                    + prps['X_H2O_melt'] * c_prps[3]['V']) + prps['V_xs_melt']

    V_melt = V_melt_molar * prps['molar_fraction_melt']

    porosity = V_melt / (V_melt + V_solid)

    # 9) Calculate the melt density and pressure gradient of the density
    M_melt_molar = (prps['X_Mg2SiO4_melt'] * c_prps[0]['molar_mass']
                    + prps['X_Fe2SiO4_melt'] * c_prps[1]['molar_mass']
                    + prps['X_H2O_melt'] * c_prps[3]['molar_mass'])

    dVdP_melt_molar = (prps['X_Mg2SiO4_melt'] * c_prps[0]['dVdP']
                         + prps['X_Fe2SiO4_melt'] * c_prps[1]['dVdP']
                         + prps['X_H2O_melt'] * c_prps[3]['dVdP']) + prps['dVdP_xs_melt']

    melt_density = M_melt_molar / V_melt_molar

    # To compute the density gradient we will ignore thermal gradients
    # and compositional gradients along the gravity vector
    melt_drhodP = - M_melt_molar / (V_melt_molar**2) * dVdP_melt_molar

    # 10) Calculate the bulk properties
    # i.e. density, alpha, beta_T, c_p

    # 11) Calculate the ?entropy? ?enthalpy?
