# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2019 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
example_anisotropic_eos
----------------

This example script demonstrates the anisotropic equation of state
of Myhill (2019).

*Uses:*

* :doc:`mineral_database`


*Demonstrates:*

* creating a mineral with excess contributions
* calculating thermodynamic properties
"""
from __future__ import absolute_import

# Here we import standard python modules that are required for
# usage of BurnMan.  In particular, numpy is used for handling
# numerical arrays and mathematical operations on them, and
# matplotlib is used for generating plots of results of calculations
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))


# Here we import the relevant modules from BurnMan.  The burnman
# module imports several of the most important functionalities of
# the library, including the ability to make composites, and compute
# thermoelastic properties of them.  The minerals module includes
# the mineral physical parameters for the predefined minerals in
# BurnMan
from burnman import eos
from burnman import AnisotropicMineral
from burnman.constants import Avogadro
from burnman.minerals import SLB_2011
from burnman.minerals import HP_2011_ds62


def cubic_Cij_to_Sij(CN11, CN12, CN44):
    # Conversions from Cijs to Sijs for a cubic crystal
    SN11 = (CN11 + CN12)/(CN11**2 + CN11*CN12 - 2*CN12**2)
    SN12 = -CN12/(CN11**2 + CN11*CN12 - 2*CN12**2)
    SN44 = 1/CN44
    return [SN11, SN12, SN44]

if __name__ == "__main__":

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    per_HP = HP_2011_ds62.per()
    per_SLB = SLB_2011.periclase()

    Fan_data = np.loadtxt('data/Fan_et_al_2019_MgO_elastic_constants.dat')
    Isaak_data = np.loadtxt('data/Isaak_et_al_1989_MgO_elastic_constants.dat')
    DS1997_data = np.loadtxt('data/Dubrovinsky_Saxena_1997_periclase_volumes.dat')
    SL1968_data = np.loadtxt('data/Smith_Leider_1968_low_temperature_volume_per.dat')


    IT, ICN11, ICN12, ICN44 = Isaak_data.T
    ISN11, ISN12, ISN44 = cubic_Cij_to_Sij(ICN11, ICN12, ICN44)

    """
    plt.plot(IT, 0.6*np.power(ICN11 - ICN12 - 2*ICN44, 2.)/(3*ICN11 - 3.*ICN12 + 4.*ICN44))
    plt.show()
    """

    FP, FT, _, rho, _, VS100, _, VP100, _, VS110, _, VP110, _, CN11, _, CN12, _, CN44, _ = Fan_data.T

    SN11, SN12, SN44 = cubic_Cij_to_Sij(CN11, CN12, CN44)
    Fmask300 = [i for i in range(len(FT)) if np.abs(FT[i]-300.) < 1. ]
    Fmask900 = [i for i in range(len(FT)) if np.abs(FT[i]-900.) < 1. ]
    """
    plt.scatter(FP[mask], np.gradient(1./SN11[mask], P[mask]), label='$1/S^S_{11}$')
    plt.scatter(FP[mask], np.gradient(1./SN12[mask], P[mask]), label='$1/S^S_{12}$')
    plt.scatter(FP[mask], np.gradient(1./SN44[mask], P[mask]), label='$1/S^S_{44}$')
    plt.legend()
    plt.show()
    """
    IKN = (ICN11 + 2.*ICN12) / 3.
    FKN = (CN11 + 2.*CN12) / 3.


    IG_V = (3.*ICN11 + 9.*ICN44 - 3.*ICN12)/15.
    IG_R = 15./(12.*ISN11 + 9.*ISN44 - 12*ISN12)
    FG_V = (3.*CN11 + 9.*CN44 - 3.*CN12)/15.
    FG_R = 15./(12.*SN11 + 9.*SN44 - 12*SN12)
    """
    T, CN11, CN12, CN44 = Isaak_data.T
    SN11, SN12, SN44 = cubic_Cij_to_Sij(CN11, CN12, CN44)

    K_R = 1./(3.*SN11 + 6.*SN12)
    K_V = (3.*CN11 + 6.*CN12)/9.

    G_V = (3.*CN11 + 9.*CN44 - 3.*CN12)/15.
    G_R = 15./(12.*SN11 + 9.*SN44 - 12*SN12)

    plt.plot(T, K_R-K_V)
    plt.plot(T, G_R-G_V)
    plt.show()
    """

    beta_T = {'invS_0': 161.3836e9,
              'invSprime_0': 3.84045,
              'invSprime_inf': 2.7,
              'invSdprime_0': (2.7 - 3.7)*3.7/161.e9 * 8./10.,
              'grueneisen_0': 1.36127,
              'q_0': 1.7217}

    S_11 = {'invS_0': 251.3e9,
            'invSprime_0': 8.2,
            'invSprime_inf': 6.2,
            'invSdprime_0': -10./250.e9,
            'grueneisen_0': 1.35*1.36127,
            'q_0': 1.7217}

    S_44 = {'invS_0': 153.8e9,
            'invSprime_0': 0.93,  # 2., 2. required to end up with G_R-G_V~0 at high pressure
            'invSprime_inf': 0.0,
            'invSdprime_0': (0 - 0.93)*0.93/154.e9/10.,
            'grueneisen_0': 1.15*1.36127,
            'q_0': 1.7217}

    params = {'symmetry_type': 'cubic',
              'equation_of_state': 'aeos',
              'formula': {'Mg': 1.,
                          'O': 1.},
              'n': 2,
              'Z': 4.,
              'T_0': 300.,
              'P_0': 100.e9,
              'G_0': 0.,
              'theta_0': 767.0977*0.806,
              'molar_mass': 0.0403044,
              'unit_cell_vectors': [np.array([4.2127e-10, 0., 0.]),
                                    np.array([0., 4.2127e-10, 0.]),
                                    np.array([0., 0., 4.2127e-10])],
              'beta_T_reuss': beta_T,
              'S_T': {'11': S_11,
                      '44': S_44}}

    periclase = AnisotropicMineral(params)

    """
    Check thermal expansivity tensor
    """

    periclase.set_state(1.e5, 300.)
    print(periclase.thermal_expansivity_tensor)
    print(periclase.alpha/3.)

    """
    Check volume relations
    """

    pressures = np.linspace(0., 100.e9, 101)
    for T in [300., 1000., 1500.]:
        temperatures = T + 0.*pressures
        V, K_T = periclase.evaluate(['V', 'isothermal_bulk_modulus_reuss'], pressures, temperatures)
        plt.plot(pressures, K_T)
        plt.plot(pressures, -V*np.gradient(pressures, V, edge_order=2), linestyle=':')
    plt.show()

    P = 20.e9
    T = 1000.
    periclase.set_state(P, T)
    a, b, c, d, gr_0, q_0 = params['beta_T_reuss']['coeffs']

    temperatures = np.linspace(0., 2000., 101)
    Pths = np.empty_like(temperatures)
    for P in [0., 25.e9, 50.e9, 100.e9, 200.e9]:
        for i, T in enumerate(temperatures):
            Pths[i] = eos.aeos.thermal_properties_ijkl(P, T, a, b, c, d, gr_0, q_0, params)['P_th']
        plt.plot(temperatures, (Pths-Pths[0])/1.e9, label='{0}'.format(P/1.e9))
    plt.legend()
    plt.show()

    pressures = np.linspace(0.e9, 100.e9, 101)
    for T in [300., 1000., 2000.]:
        temperatures = T + 0.*pressures
        volumes_SLB = per_SLB.evaluate(['K_T'], pressures, temperatures)[0]
        volumes_HP = per_HP.evaluate(['K_T'], pressures, temperatures)[0]
        volumes = periclase.evaluate(['isothermal_bulk_modulus_reuss'], pressures, temperatures)[0]
        volumes_SLB = per_SLB.evaluate(['V'], pressures, temperatures)[0]
        volumes_HP = per_HP.evaluate(['V'], pressures, temperatures)[0]
        volumes = periclase.evaluate(['V'], pressures, temperatures)[0]
        plt.plot(pressures, volumes)
        plt.plot(pressures, volumes_SLB, linestyle=':')
        plt.plot(pressures, volumes_HP, linestyle='--')
    plt.show()

    fig = plt.figure(figsize=(18, 5))
    ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

    pressures = np.linspace(0., 60.e9, 101)
    S11 = np.empty_like(pressures)
    S12 = np.empty_like(pressures)
    S44 = np.empty_like(pressures)
    for T, linestyle in [(300., '-'), (500., ':'), (700., ':'), (900., ':')]:
        print(T)
        for i, P in enumerate(pressures):
            periclase.set_state(P, T)
            S = periclase.isentropic_elastic_compliance_tensor
            S11[i] = S[0][0]
            S12[i] = S[0][1]
            S44[i] = S[3][3]

        ax[0].plot(pressures/1.e9, 1.e-9/S11, linestyle=linestyle)
        ax[1].plot(pressures/1.e9, 1.e-9/S12, linestyle=linestyle)
        ax[2].plot(pressures/1.e9, 1.e-9/S44, linestyle=linestyle)

    IT, ICN11, ICN12, ICN44 = Isaak_data.T
    ISN11, ISN12, ISN44 = cubic_Cij_to_Sij(ICN11, ICN12, ICN44)

    mask = [i for i in range(len(IT)) if int(IT[i]) in [300, 500, 700, 900]]
    ax[0].scatter(0.+ 0.*IT[mask], 1./ISN11[mask])
    ax[1].scatter(0.+ 0.*IT[mask], 1./ISN12[mask])
    ax[2].scatter(0.+ 0.*IT[mask], 1./ISN44[mask])

    FP, FT, _, rho, _, VS100, _, VP100, _, VS110, _, VP110, _, CN11, _, CN12, _, CN44, _ = Fan_data.T
    SN11, SN12, SN44 = cubic_Cij_to_Sij(CN11, CN12, CN44)

    for temp in [300, 500, 700, 900]:
        mask = [i for i in range(len(FT)) if np.abs(FT[i]-temp) < 1.]
        ax[0].scatter(FP[mask], 1./SN11[mask])
        ax[1].scatter(FP[mask], 1./SN12[mask])
        ax[2].scatter(FP[mask], 1./SN44[mask])

    plt.show()


    fig = plt.figure(figsize=(12, 5))
    ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

    pressures = np.linspace(0., 20.e9, 101)
    for i, T in enumerate([300., 900., 1500., 1800.]):
        print(T)
        c = colors[i]
        temperatures = pressures*0. + T

        K_SLB, G_SLB = per_SLB.evaluate(['adiabatic_bulk_modulus', 'shear_modulus'], pressures, temperatures)
        K_HP = per_HP.evaluate(['adiabatic_bulk_modulus'], pressures, temperatures)[0]
        ax[0].plot(pressures/1.e9, K_SLB/1.e9, linestyle=':', color=c)
        ax[0].plot(pressures/1.e9, G_SLB/1.e9, linestyle=':', color=c)
        ax[0].plot(pressures/1.e9, K_HP/1.e9, linestyle='--', color=c)

        K_vrh, G_vrh = periclase.evaluate(['isentropic_bulk_modulus_vrh', 'shear_modulus_vrh'], pressures, temperatures)
        K_v, G_v = periclase.evaluate(['isentropic_bulk_modulus_voigt', 'shear_modulus_voigt'], pressures, temperatures)
        K_r, G_r = periclase.evaluate(['isentropic_bulk_modulus_reuss', 'shear_modulus_reuss'], pressures, temperatures)

        ax[0].fill_between(pressures/1.e9, K_r/1.e9, K_v/1.e9, color=c, alpha=0.3)
        ax[0].fill_between(pressures/1.e9, G_r/1.e9, G_v/1.e9, color=c, alpha=0.3)

        ax[0].plot(pressures/1.e9, K_vrh/1.e9, color=c, linewidth=1.0, label='$K^S$ ({0} K)'.format(int(T)))
        ax[0].plot(pressures/1.e9, G_vrh/1.e9, color=c, linewidth=1.0, linestyle='--', label='$G$ ({0} K)'.format(int(T)))
        ax[1].plot(pressures/1.e9, np.gradient(K_vrh, pressures, edge_order=2), color=c, linewidth=1., label='$K^S$ ({0} K)'.format(int(T)))
        ax[1].plot(pressures/1.e9, np.gradient(G_vrh, pressures, edge_order=2), color=c, linewidth=1., linestyle='--', label='$G$ ({0} K)'.format(int(T)))

        Imask = [i for i in range(len(IT)) if int(IT[i]) in [T]]
        ax[0].scatter(0.+ 0.*IT[Imask], IKN[Imask], color=c)
        ax[0].scatter(0.+ 0.*IT[Imask], IG_V[Imask], color=c)
        ax[0].scatter(0.+ 0.*IT[Imask], IG_R[Imask], color=c)

    for i in range(2):
        ax[i].set_xlim(0,20.)
        ax[i].set_xlabel('Pressure (GPa)')

    ax[1].legend()

    for i, Fmask in enumerate([Fmask300, Fmask900]):
        ax[0].scatter(FP[Fmask], FKN[Fmask], color=colors[i])
        ax[0].scatter(FP[Fmask], FG_V[Fmask], color=colors[i])
        ax[0].scatter(FP[Fmask], FG_R[Fmask], color=colors[i])


    ax[0].set_ylabel('$G$, $K^S$ (GPa)')
    ax[1].set_ylabel('$G\'$, $K^{S}\'$')
    fig.savefig('periclase_elastic_moduli')
    plt.show()
    exit()

    pressures = np.linspace(0., 3.e9, 101)
    for T in [300., 800., 1300.]:
        temperatures = pressures*0. + T
        vp_reuss, vp_voigt, vp_vrh = periclase.evaluate(['p_wave_velocity_reuss', 'p_wave_velocity_voigt', 'p_wave_velocity_vrh'], pressures, temperatures)
        vs_reuss, vs_voigt, vs_vrh = periclase.evaluate(['shear_wave_velocity_reuss', 'shear_wave_velocity_voigt', 'shear_wave_velocity_vrh'], pressures, temperatures)
        vphi_reuss, vphi_voigt, vphi_vrh = periclase.evaluate(['bulk_sound_velocity_reuss', 'bulk_sound_velocity_voigt', 'bulk_sound_velocity_vrh'], pressures, temperatures)
        vp_slb, vs_slb, vphi_slb = per_SLB.evaluate(['p_wave_velocity', 'shear_wave_velocity', 'bulk_sound_velocity'], pressures, temperatures)

        plt.fill_between(pressures/1.e9, vs_reuss, vs_voigt)
        plt.plot(pressures/1.e9, vs_vrh, color='black', linewidth=1.0)
        plt.fill_between(pressures/1.e9, vp_reuss, vp_voigt)
        plt.plot(pressures/1.e9, vp_vrh, color='black', linewidth=1.0)
        plt.fill_between(pressures/1.e9, vphi_reuss, vphi_voigt)
        plt.plot(pressures/1.e9, vphi_vrh, color='black', linewidth=1.0)
        plt.plot(pressures/1.e9, vp_slb, color='red', linestyle=':')
        plt.plot(pressures/1.e9, vs_slb, color='red', linestyle=':')
        plt.plot(pressures/1.e9, vphi_slb, color='red', linestyle=':')

    plt.show()

    fig = plt.figure(figsize=(12, 5))
    ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
    pressures = np.linspace(0., 150.e9, 101)

    for i, T in enumerate([300., 1000., 2000., 3000., 4000.]):
        temperatures = T + 0.*pressures
        G_R, G_V, G_VRH = periclase.evaluate(['shear_modulus_reuss', 'shear_modulus_voigt', 'shear_modulus_vrh'], pressures, temperatures)
        K_R, K_V, K_VRH = periclase.evaluate(['isentropic_bulk_modulus_reuss', 'isentropic_bulk_modulus_voigt', 'isentropic_bulk_modulus_vrh'], pressures, temperatures)

        c = colors[i]
        ax[0].fill_between(pressures/1.e9, K_R/1.e9, K_V/1.e9, color=c, alpha=0.3,
                           linewidth=0)
        ax[0].plot(pressures/1.e9, K_VRH/1.e9, color=c, linewidth=1,
                   label='{0} K'.format(T))

        ax[1].fill_between(pressures/1e9, G_R/1.e9, G_V/1.e9, color=c, alpha=0.3,
                           linewidth=0)
        ax[1].plot(pressures/1e9, G_VRH/1.e9, color=c, linewidth=1,
                   label='{0} K'.format(T))

    ax[0].set_ylim(0,)
    ax[1].set_ylim(0,)
    ax[0].set_xlabel('P (GPa)')
    ax[0].set_ylabel('$K^S$ (GPa)')
    ax[1].set_xlabel('P (GPa)')
    ax[1].set_ylabel('$G^S$ (GPa)')
    ax[0].legend()
    ax[1].legend()
    plt.show()


    temperatures = np.linspace(100., 2000., 101)
    S11 = np.empty_like(temperatures)
    S12 = np.empty_like(temperatures)
    S44 = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        periclase.set_state(0, T)
        S = periclase.isentropic_elastic_compliance_tensor
        S11[i] = S[0][0]
        S12[i] = S[0][1]
        S44[i] = S[3][3]
    plt.plot(temperatures, 1./S11)
    plt.plot(temperatures, 1./S12)
    plt.plot(temperatures, 1./S44)

    plt.scatter(IT, 1.e9/ISN11)
    plt.scatter(IT, 1.e9/ISN12)
    plt.scatter(IT, 1.e9/ISN44)

    plt.show()


    fig = plt.figure(figsize=(18, 5))
    ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

    pressures = np.linspace(-5.e9, 150.e9, 101)
    for T in [300., 1000., 2000.]:
        temperatures = T + 0.*pressures
        alphas, volumes, KT = periclase.evaluate(['alpha', 'V', 'isothermal_bulk_modulus_reuss'], pressures, temperatures)
        alphas_HP, volumes_HP, KT_HP = per.evaluate(['alpha', 'V', 'K_T'], pressures, temperatures)
        alphas_SLB, volumes_SLB, KT_SLB = per_SLB.evaluate(['alpha', 'V', 'K_T'], pressures, temperatures)

        ax[0].plot(pressures/1.e9, volumes*1.e6, label='{0} K'.format(T))
        ax[0].plot(pressures/1.e9, volumes_HP*1.e6, linestyle='--', label='{0} K (HP)'.format(T))
        ax[0].plot(pressures/1.e9, volumes_SLB*1.e6, linestyle=':', label='{0} K (SLB)'.format(T))

        ax[1].plot(pressures/1.e9, np.gradient(KT, pressures, edge_order=2), label='K_T @ {0} K'.format(T))
        ax[1].plot(pressures/1.e9, np.gradient(KT_HP, pressures, edge_order=2), linestyle='--', label='K_T @ {0} K (HP)'.format(T))
        ax[1].plot(pressures/1.e9, np.gradient(KT_SLB, pressures, edge_order=2), linestyle=':', label='K_T @ {0} K (SLB)'.format(T))

        ax[2].plot(pressures/1.e9, alphas*1.e6, label='a @ {0} K'.format(T))
        ax[2].plot(pressures/1.e9, alphas_HP*1.e6, linestyle='--', label='a @ {0} K (HP)'.format(T))
        ax[2].plot(pressures/1.e9, alphas_SLB*1.e6, linestyle=':', label='a @ {0} K (SLB)'.format(T))

    ax[1].set_xlim(0., 150.)
    ax[2].set_xlim(0., 150.)

    #ax[0].set_ylim(5., 14.)
    ax[1].set_ylim(2.5, 4.5)
    ax[2].set_ylim(5., 40.)

    ax[0].set_ylabel('V (cm$^3$/mol)')
    ax[1].set_ylabel('$K_T$ (GPa)')
    ax[2].set_ylabel('$\\alpha$ (cm$^3$/mol)')


    mask = [i for i in range(len(FT)) if np.abs(FT[i]-300) < 1.]
    ax[1].scatter(FP[mask], np.gradient(1./(3.*SN11[mask] + 6.*SN12[mask]), FP[mask], edge_order=2))

    print(per_SLB.params)
    for i in range(3):
        ax[i].set_xlabel('P (GPa)')
        ax[i].set_xlim(0., 150.)
        ax[i].legend()
    plt.show()
