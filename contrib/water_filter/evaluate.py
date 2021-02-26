import numpy as np
import matplotlib.pyplot as plt
from FMSH_melt_model import equilibrate

from endmember_eos import thermodynamic_properties
from model_parameters import Mg2SiO4_params, Fe2SiO4_params
from model_parameters import MgSiO3_params, H2O_params
from FMSH_melt_model import stable_phases
from FMSH_melt_model import melt_excess_volume, solid_excess_volume
from FMSH_melt_model import one_phase_eqm, two_phase_eqm


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
    phases, f_tr, df_trdP, df_trdT = stable_phases(P, T)

    # 3) Calculate molar fraction of melt from volume fraction (porosity)
    # The solid is allowed to react with itself to form another
    # (metastable) solid, but solid is not allowed to melt,
    # melt is not allowed to solidify and melt and solid are not allowed to
    # react with each other until after this step.
    X_ol_solid = X_Mg2SiO4_solid + X_Fe2SiO4_solid
    V_xs_solid = solid_excess_volume(P, T,
                                     X_ol_solid, X_MgSiO3_solid,
                                     X_H2O_solid, phases, f_tr)
    V_xs_melt = melt_excess_volume(P, T,
                                   X_Mg2SiO4_melt, X_Fe2SiO4_melt, X_H2O_melt)

    moles_solid = ((1. - porosity)
                   / ((X_Mg2SiO4_solid * c_prps[0]['V']
                       + X_Fe2SiO4_solid * c_prps[1]['V']
                       + X_MgSiO3_solid * c_prps[2]['V']
                       + X_H2O_solid * c_prps[3]['V'])
                       + V_xs_solid))

    moles_melt = (porosity
                  / ((X_Mg2SiO4_melt * c_prps[0]['V']
                      + X_Fe2SiO4_melt * c_prps[1]['V']
                      + X_H2O_melt * c_prps[3]['V']) + V_xs_melt))

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

    dT = 1.
    dP = 10.
    # Not correctly centered, but this won't make a big difference.
    if len(phases) == 1:
        prps = one_phase_eqm(P, T,
                             X_Mg2SiO4_bulk, X_Fe2SiO4_bulk,
                             X_MgSiO3_bulk, X_H2O_bulk,
                             phases[0])
        prpsdP = one_phase_eqm(P + dP, T,
                               X_Mg2SiO4_bulk, X_Fe2SiO4_bulk,
                               X_MgSiO3_bulk, X_H2O_bulk,
                               phases[0])
        prpsdT = one_phase_eqm(P, T + dT,
                               X_Mg2SiO4_bulk, X_Fe2SiO4_bulk,
                               X_MgSiO3_bulk, X_H2O_bulk,
                               phases[0])
    else:
        prps = two_phase_eqm(P, T,
                             X_Mg2SiO4_bulk, X_Fe2SiO4_bulk,
                             X_MgSiO3_bulk, X_H2O_bulk,
                             phases, f_tr)
        prpsdP = two_phase_eqm(P+dP, T,
                               X_Mg2SiO4_bulk, X_Fe2SiO4_bulk,
                               X_MgSiO3_bulk, X_H2O_bulk,
                               phases, f_tr + df_trdP*dP)
        prpsdT = two_phase_eqm(P, T+dT,
                               X_Mg2SiO4_bulk, X_Fe2SiO4_bulk,
                               X_MgSiO3_bulk, X_H2O_bulk,
                               phases, f_tr + df_trdT*dT)

    # 8) Calculate the new porosity
    V_solid_molar = (((prps['X_Mg2SiO4_solid'] * c_prps[0]['V']
                       + prps['X_Fe2SiO4_solid'] * c_prps[1]['V']
                       + prps['X_MgSiO3_solid'] * c_prps[2]['V']
                       + prps['X_H2O_solid'] * c_prps[3]['V'])
                      + prps['V_xs_solid']))

    V_solid = V_solid_molar * (1. - prps['molar_fraction_melt'])

    V_melt_molar = (prps['X_Mg2SiO4_melt'] * c_prps[0]['V']
                    + prps['X_Fe2SiO4_melt'] * c_prps[1]['V']
                    + prps['X_H2O_melt'] * c_prps[3]['V']) + prps['V_xs_melt']

    V_melt = V_melt_molar * prps['molar_fraction_melt']

    new_porosity = V_melt / (V_melt + V_solid)

    # 9) Calculate the solid and melt density
    # and pressure gradient of the density
    M_solid_molar = (prps['X_Mg2SiO4_solid'] * c_prps[0]['molar_mass']
                     + prps['X_Fe2SiO4_solid'] * c_prps[1]['molar_mass']
                     + prps['X_MgSiO3_solid'] * c_prps[2]['molar_mass']
                     + prps['X_H2O_solid'] * c_prps[3]['molar_mass'])

    M_melt_molar = (prps['X_Mg2SiO4_melt'] * c_prps[0]['molar_mass']
                    + prps['X_Fe2SiO4_melt'] * c_prps[1]['molar_mass']
                    + prps['X_H2O_melt'] * c_prps[3]['molar_mass'])

    dVdP_melt_molar = ((prps['X_Mg2SiO4_melt'] * c_prps[0]['dVdP']
                        + prps['X_Fe2SiO4_melt'] * c_prps[1]['dVdP']
                        + prps['X_H2O_melt'] * c_prps[3]['dVdP'])
                       + (prpsdP['V_xs_melt'] - prps['V_xs_melt'])/dP)

    dVdP_solid_molar = ((prps['X_Mg2SiO4_solid'] * c_prps[0]['dVdP']
                         + prps['X_Fe2SiO4_solid'] * c_prps[1]['dVdP']
                         + prps['X_MgSiO3_solid'] * c_prps[2]['dVdP']
                         + prps['X_H2O_solid'] * c_prps[3]['dVdP'])
                        + (prpsdP['V_xs_solid'] - prps['V_xs_solid'])/dP)

    solid_density = M_solid_molar / V_solid_molar
    melt_density = M_melt_molar / V_melt_molar

    # To compute the density gradient we will ignore thermal gradients
    # and compositional gradients along the gravity vector
    solid_drhodP = - M_solid_molar / (V_solid_molar**2) * dVdP_solid_molar
    melt_drhodP = - M_melt_molar / (V_melt_molar**2) * dVdP_melt_molar

    # 10) Calculate the bulk properties
    # i.e. density, alpha, beta_T, c_p
    V_xs_bulk = (prps['V_xs_melt'] * prps['molar_fraction_melt']
                 + prps['V_xs_solid'] * (1. - prps['molar_fraction_melt']))

    S_xs_bulk = (prps['S_xs_melt'] * prps['molar_fraction_melt']
                 + prps['S_xs_solid'] * (1. - prps['molar_fraction_melt']))

    dVdP_xs_bulk = ((prpsdP['V_xs_melt'] * prpsdP['molar_fraction_melt']
                    + prpsdP['V_xs_solid']
                    * (1. - prpsdP['molar_fraction_melt'])) - V_xs_bulk) / dP

    dVdT_xs_bulk = ((prpsdT['V_xs_melt'] * prpsdT['molar_fraction_melt']
                    + prpsdT['V_xs_solid']
                    * (1. - prpsdT['molar_fraction_melt'])) - V_xs_bulk) / dT

    dSdT_xs_bulk = ((prpsdT['S_xs_melt'] * prpsdT['molar_fraction_melt']
                    + prpsdT['S_xs_solid']
                    * (1. - prpsdT['molar_fraction_melt'])) - S_xs_bulk) / dT

    V_bulk = ((X_Mg2SiO4_bulk * c_prps[0]['V']
               + X_Fe2SiO4_bulk * c_prps[1]['V']
               + X_MgSiO3_bulk * c_prps[2]['V']
               + X_H2O_bulk * c_prps[3]['V']) + V_xs_bulk)

    S_bulk = ((X_Mg2SiO4_bulk * c_prps[0]['S']
               + X_Fe2SiO4_bulk * c_prps[1]['S']
               + X_MgSiO3_bulk * c_prps[2]['S']
               + X_H2O_bulk * c_prps[3]['S']) + S_xs_bulk)

    dVdP_bulk = ((X_Mg2SiO4_bulk * c_prps[0]['dVdP']
                 + X_Fe2SiO4_bulk * c_prps[1]['dVdP']
                 + X_MgSiO3_bulk * c_prps[2]['dVdP']
                 + X_H2O_bulk * c_prps[3]['dVdP']) + dVdP_xs_bulk)

    dVdT_bulk = ((X_Mg2SiO4_bulk * c_prps[0]['dVdT']
                 + X_Fe2SiO4_bulk * c_prps[1]['dVdT']
                 + X_MgSiO3_bulk * c_prps[2]['dVdT']
                 + X_H2O_bulk * c_prps[3]['dVdT']) + dVdT_xs_bulk)

    dSdT_bulk = ((X_Mg2SiO4_bulk * c_prps[0]['dSdT']
                 + X_Fe2SiO4_bulk * c_prps[1]['dSdT']
                 + X_MgSiO3_bulk * c_prps[2]['dSdT']
                 + X_H2O_bulk * c_prps[3]['dSdT']) + dSdT_xs_bulk)

    M_bulk = (X_Mg2SiO4_bulk * c_prps[0]['molar_mass']
              + X_Fe2SiO4_bulk * c_prps[1]['molar_mass']
              + X_MgSiO3_bulk * c_prps[2]['molar_mass']
              + X_H2O_bulk * c_prps[3]['molar_mass'])

    # density = M / V
    density = M_bulk / V_bulk

    # alpha = 1/V dV/dT
    alpha = dVdT_bulk / V_bulk

    # beta_T = -1/V dV/dP
    beta_T = -dVdP_bulk / V_bulk

    # Specific C_p = T dS/dT / M
    specific_C_p = T * dSdT_bulk / M_bulk

    # Specific entropy
    specific_S = S_bulk / M_bulk

    return {'porosity': new_porosity,
            'solid_density': solid_density,
            'melt_density': melt_density,
            'solid_drhodP': solid_drhodP,
            'melt_drhodP': melt_drhodP,
            'X_Mg2SiO4_solid': prps['X_Mg2SiO4_solid'],
            'X_Fe2SiO4_solid': prps['X_Fe2SiO4_solid'],
            'X_MgSiO3_solid': prps['X_MgSiO3_solid'],
            'X_H2O_solid': prps['X_H2O_solid'],
            'X_Mg2SiO4_melt': prps['X_Mg2SiO4_melt'],
            'X_Fe2SiO4_melt': prps['X_Fe2SiO4_melt'],
            'X_MgSiO3_melt': prps['X_MgSiO3_melt'],
            'X_H2O_melt': prps['X_H2O_melt'],
            'bulk_density': density,
            'bulk_alpha': alpha,
            'bulk_beta_T': beta_T,
            'bulk_specific_C_p': specific_C_p,
            'bulk_specific_S': specific_S}


if __name__ == '__main__':
    #                [f_melt,
    #                 X_H2O_melt, X_MgSiO3_melt,
    #                 X_Fe2SiO4_melt, X_Mg2SiO4_melt,
    #                 X_H2O_solid, X_MgSiO3_solid,
    #                 X_Fe2SiO4_solid, X_Mg2SiO4_solid,
    #                 S_xs_melt, V_xs_melt, dVdP_xs_melt,
    #                 S_xs_solid, V_xs_solid, dVdP_xs_solid])
    pressures = np.linspace(6.e9, 30.e9, 101)
    porosity = np.empty_like(pressures)
    X_H2O_melts = np.empty_like(pressures)
    X_H2O_solids = np.empty_like(pressures)
    KD = np.empty_like(pressures)
    S_xs_melt = np.empty_like(pressures)
    S_xs_solid = np.empty_like(pressures)
    V_xs_melt = np.empty_like(pressures)
    V_xs_solid = np.empty_like(pressures)
    solid_density = np.empty_like(pressures)
    melt_density = np.empty_like(pressures)
    bulk_density = np.empty_like(pressures)
    bulk_entropy = np.empty_like(pressures)
    bulk_alpha = np.empty_like(pressures)
    bulk_beta = np.empty_like(pressures)
    bulk_Cp = np.empty_like(pressures)

    T = 1600.
    c = [0.65, 0.1, 0.2, 0.05]
    for i, P in enumerate(pressures):
        X_Mg2SiO4, X_Fe2SiO4, X_MgSiO3, X_H2O = c

        # Do the formward calculation for bulk properties
        eqm = equilibrate(P, T, X_Mg2SiO4, X_Fe2SiO4, X_MgSiO3, X_H2O)
        c_prps = [thermodynamic_properties(P, T, Mg2SiO4_params),
                  thermodynamic_properties(P, T, Fe2SiO4_params),
                  thermodynamic_properties(P, T, MgSiO3_params),
                  thermodynamic_properties(P, T, H2O_params)]
        V_solid_molar = (((eqm['X_Mg2SiO4_solid'] * c_prps[0]['V']
                           + eqm['X_Fe2SiO4_solid'] * c_prps[1]['V']
                           + eqm['X_MgSiO3_solid'] * c_prps[2]['V']
                           + eqm['X_H2O_solid'] * c_prps[3]['V'])
                          + eqm['V_xs_solid']))

        V_solid = V_solid_molar * (1. - eqm['molar_fraction_melt'])

        V_melt_molar = (eqm['X_Mg2SiO4_melt'] * c_prps[0]['V']
                        + eqm['X_Fe2SiO4_melt'] * c_prps[1]['V']
                        + eqm['X_H2O_melt'] * c_prps[3]['V']) + eqm['V_xs_melt']

        V_melt = V_melt_molar * eqm['molar_fraction_melt']

        old_porosity = V_melt / (V_melt + V_solid)

        # Now do the inverse calculation
        out = evaluate(P, T,
                       eqm['X_Mg2SiO4_solid'], eqm['X_Fe2SiO4_solid'],
                       eqm['X_MgSiO3_solid'], eqm['X_H2O_solid'],
                       eqm['X_Mg2SiO4_melt'], eqm['X_Fe2SiO4_melt'],
                       eqm['X_H2O_melt'],
                       old_porosity)

        assert (np.abs(out['porosity'] - old_porosity) < 1.e-10)
        assert (np.abs(out['X_Mg2SiO4_solid'] - eqm['X_Mg2SiO4_solid']) < 1.e-10)
        assert (np.abs(out['X_Fe2SiO4_solid'] - eqm['X_Fe2SiO4_solid']) < 1.e-10)
        assert (np.abs(out['X_MgSiO3_solid'] - eqm['X_MgSiO3_solid']) < 1.e-10)
        assert (np.abs(out['X_H2O_solid'] - eqm['X_H2O_solid']) < 1.e-10)
        assert (np.abs(out['X_H2O_melt'] - eqm['X_H2O_melt']) < 1.e-10)

        porosity[i] = out['porosity']
        solid_density[i] = out['solid_density']
        melt_density[i] = out['melt_density']
        bulk_density[i] = out['bulk_density']
        bulk_entropy[i] = out['bulk_specific_S']
        bulk_alpha[i] = out['bulk_alpha']
        bulk_beta[i] = out['bulk_beta_T']
        bulk_Cp[i] = out['bulk_specific_C_p']

        X_H2O_melts[i] = out['X_H2O_melt']
        X_H2O_solids[i] = out['X_H2O_solid']
        KD[i] = (2.*out['X_Fe2SiO4_solid']*(2. * out['X_Mg2SiO4_melt'] + out['X_MgSiO3_melt'])
                 / (2.*out['X_Fe2SiO4_melt']*(2. * out['X_Mg2SiO4_solid'] + out['X_MgSiO3_solid'])))

        S_xs_solid[i] = eqm['S_xs_solid']
        S_xs_melt[i] = eqm['S_xs_melt']
        V_xs_solid[i] = eqm['V_xs_solid']
        V_xs_melt[i] = eqm['V_xs_melt']

    fig = plt.figure(figsize=(12,6))
    ax = [fig.add_subplot(2, 3, i) for i in range(1, 7)]
    ax[0].plot(pressures/1.e9, porosity, label='porosity')
    ax[0].plot(pressures/1.e9, X_H2O_solids, label='X_H2O solid')
    ax[0].plot(pressures/1.e9, X_H2O_melts, label='X_H2O melt')
    ax[0].plot(pressures/1.e9, KD, label='K_D')
    #ax[1].plot(pressures/1.e9, S_xs_solid, label='S_xs_solid')
    #ax[1].plot(pressures/1.e9, S_xs_melt, label='S_xs_melt')
    #ax[2].plot(pressures/1.e9, V_xs_solid, label='V_xs_solid')
    #ax[2].plot(pressures/1.e9, V_xs_melt, label='V_xs_melt')

    ax[1].plot(pressures/1.e9, solid_density, label='solid')
    ax[1].plot(pressures/1.e9, melt_density, label='melt')
    ax[1].plot(pressures/1.e9, bulk_density, label='bulk')
    ax[2].plot(pressures/1.e9, bulk_entropy, label='bulk')

    ax[3].plot(pressures/1.e9, bulk_alpha, label='bulk alpha')
    ax[4].plot(pressures/1.e9, bulk_beta, label='bulk beta')
    ax[5].plot(pressures/1.e9, bulk_Cp, label='bulk Cp')

    for i in range(2):
        ax[i].legend()
    for i in range(6):
        ax[i].set_xlabel('Pressure (GPa)')
    ax[0].set_ylabel('composition')
    ax[1].set_ylabel('density (kg/m$^3$)')
    ax[2].set_ylabel('entropy (J/K/kg)')
    ax[3].set_ylabel('bulk alpha (/K)')
    ax[4].set_ylabel('bulk beta (/Pa)')
    ax[5].set_ylabel('bulk Cp (J/K/kg)')

    fig.tight_layout()
    fig.savefig(f'output_figures/model_properties_{c[0]}_{c[1]}_{c[2]}_{c[3]}_{T}_K.pdf')
    plt.show()
