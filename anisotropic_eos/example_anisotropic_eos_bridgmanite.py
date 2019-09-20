# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
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
from burnman.minerals import HGP_2018_ds633
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz


if __name__ == "__main__":
    bdg_HP = HGP_2018_ds633.mpv()
    bdg_SLB = SLB_2011.mg_bridgmanite()

    data = np.loadtxt('data/Zhang_et_al_2013_LDA.dat')
    S_sets = []
    labels = ['$\\beta_R$', '$E_2$', '$E_3$',
              '$S_{11}$', '$S_{22}$', '$S_{33}$',
              '$S_{44}$', '$S_{55}$', '$S_{66}$']

    K_R = []
    K_V = []
    for datum in data:
        T, V, rho, P, C11T, C11S, C22T, C22S, C33T, C33S = datum[:10]
        C12T, C12S, C13T, C13S, C23T, C23S, C44, C55, C66 = datum[10:]
        C = np.array([[C11T, C12T, C13T, 0, 0, 0],
                      [C12T, C22T, C23T, 0, 0, 0],
                      [C13T, C23T, C33T, 0, 0, 0],
                      [0, 0, 0, C44, 0, 0],
                      [0, 0, 0, 0, C55, 0],
                      [0, 0, 0, 0, 0, C66]])

        S = np.linalg.pinv(C)

        E_1 = sum(S[0, :3])
        E_2 = sum(S[1, :3])
        E_3 = sum(S[2, :3])
        beta_R = E_1 + E_2 + E_3
        S_sets.append([beta_R, E_2, E_3,
                       S[0, 0], S[1, 1], S[2, 2],
                       S[3, 3], S[4, 4], S[5, 5]])

        K_R.append(1./beta_R)
        K_V.append(np.sum(np.sum(C[:3, :3])) / 9.)

    S_sets = np.array(S_sets).T*1.e-9

    dct = {'invS_0': 2.505264e+11,
           'invSprime_0': 4.14,
           'invSprime_inf': 2.7,
           'invSdprime_0': -4.14/250.e9,
           'f': 1.}

    params = {'theta_0': 905.9412*0.806,
              'T_0': 300.,
              'V_0': 2.4445e-05,
              'grueneisen_0': 1.56508,
              'q_0': 1.10945,
              'n': 5.,
              'beta_T_reuss': dct}

    params['beta_T_reuss']['coeffs'] = eos.aeos.Sijkl_derivs_to_coeffs(dct)

    def fn_invcompliance(PT, invS0, invSprime, invSprime_inf, invSdprime, fac):

        dct = {'invS_0': invS0,
               'invSprime_0': invSprime,
               'invSprime_inf': invSprime_inf,
               'invSdprime_0': invSdprime,
               'f': fac}
        a, b, c, d, f = eos.aeos.Sijkl_derivs_to_coeffs(dct)

        P = PT[:, 0]
        T = PT[:, 1]
        Tmod = np.array([t if t > 10. else 10. for t in T])
        Ss = []
        for i in range(len(P)):
            Ss.append(eos.aeos.thermal_properties_ijkl(P[i], Tmod[i],
                                                       a, b, c, d, f,
                                                       params)['S'])
        return 1./np.array(Ss)

    PT = np.array([data[:, 3]*1.e9, data[:, 0]]).T
    print(PT)
    S_prms = []
    for Sij in S_sets:

        p = curve_fit(fn_invcompliance, PT, 1./Sij,
                      [100.e9, 8., 4., -0.3/100.e9, 1.],
                      bounds = ((0, 0, 0, -0.02e-9, -1.),
                                (1000e9, 15, 15, 0.00, 3.)))[0]
        dct = {'invS_0': p[0],
               'invSprime_0': p[1],
               'invSprime_inf': p[2],
               'invSdprime_0': p[3],
               'f': p[4]}
        a, b, c, d, f = eos.aeos.Sijkl_derivs_to_coeffs(dct)

        if (2*(p[1] - p[2])/p[3] > -p[0]/p[1]):
            # Heuristic
            f = 10. # >> 1
            p[3] = (p[2] - p[1])*p[1]/p[0]/f
        S_prms.append(p)
        print(p)
    S_prms = np.array(S_prms)

    pressures = np.linspace(-40e9, 150e9, 1001)
    temperatures = pressures*0. + 300.
    PT = np.array([pressures, temperatures]).T
    for i in range(9):
        plt.plot(pressures, fn_invcompliance(PT, *S_prms[i, :5]),
                 label=labels[i])

    for i in range(9):
        plt.scatter(data[:, 3]*1.e9, 1./S_sets[i])
    plt.legend()
    plt.show()

    S_dict = [{'invS_0': 2.505264e+11,
               'invSprime_0': 4.14,
               'invSprime_inf': 3.0,
               'invSdprime_0': -10./250.e9,
               'f': 1.}]
    S_dict.extend([{'invS_0': p[0],
                    'invSprime_0': p[1],
                    'invSprime_inf': p[2],
                    'invSdprime_0': p[3],
                    'f': p[4]}
                   for p in S_prms[1:]])

    params = {'symmetry_type': 'orthorhombic',
              'equation_of_state': 'aeos',
              'formula': {'Mg': 1.,
                          'Si': 1.,
                          'O': 3.},
              'n': 5.,
              'Z': 4.,
              'T_0': 300.,
              'P_0': 200.e9,
              'G_0': 0.,
              'theta_0': 905.9412*0.806,
              'grueneisen_0': 1.56508,
              'q_0': 1.10945,
              'molar_mass': 0.1003887,  # for MgSiO3
              'unit_cell_vectors': [np.array([4.7759e-10, 0., 0.]),
                                    np.array([0., 4.9284e-10, 0.]),
                                    np.array([0., 0., 6.8972e-10])],  # Tange et al., 2009 for MgSiO3
              'beta_T_reuss': S_dict[0],
              'E_2': S_dict[1],
              'E_3': S_dict[2],
              'S_T': {'11': S_dict[3],
                      '22': S_dict[4],
                      '33': S_dict[5],
                      '44': S_dict[6],
                      '55': S_dict[7],
                      '66': S_dict[8]}}

    bridgmanite = AnisotropicMineral(params)
    print(bridgmanite.params['beta_T_reuss'])
    pressures = np.linspace(0.e9, 150.e9, 101)
    for T in [300., 1000., 2000.]:
        temperatures = T + pressures*0.
        volumes = bridgmanite.evaluate(['V'], pressures, temperatures)[0]
        plt.plot(pressures/1.e9, volumes*1.e6)
        volumes = bdg_SLB.evaluate(['V'], pressures, temperatures)[0]
        plt.plot(pressures/1.e9, volumes*1.e6, linestyle=':')
    plt.show()
    """
    MODIFY TO CORRECT FOR OVER/UNDERBINDING!!!
    """
    from scipy.optimize import brentq

    def _delta_V(pressure, volume, temperature):
        bridgmanite.set_state(pressure, temperature)
        return volume - bridgmanite.V

    bridgmanite.set_state(1.e5, 10.)
    VEXP0 = bridgmanite.V # 24.12e-6
    VLDA0 = 23.97e-6
    KEXP0 = 262.0e9
    KLDA0 = 259.3e9

    PT = []
    S_sets = []
    for datum in data:
        V = datum[1]*1.e-30*Avogadro/4.
        Vmod = V*VLDA0/VEXP0
        T = max((datum[0], 10.))

        bridgmanite.set_state(datum[3]*1.e9, T)
        print(bridgmanite.V, V)

        P_DFT = brentq(_delta_V, 0.e9, 200.e9, args=(Vmod, T))
        delta_P_DFT = (brentq(_delta_V, 0.e9, 200.e9, args=(V, T)) -
                       brentq(_delta_V, 0.e9, 200.e9, args=(V, 10.)))
        P_mod = KEXP0/KLDA0*P_DFT #+ delta_P_DFT
        #print(P_mod/1.e9, datum[3])

        bridgmanite.set_state(P_DFT, T)
        c = bridgmanite.isothermal_elastic_stiffness_tensor
        bridgmanite.set_state(P_DFT, 10.)
        c0 = bridgmanite.isothermal_elastic_stiffness_tensor
        C_mod = c*KEXP0/KLDA0  #+ (c - c0)

        S = np.linalg.pinv(C_mod)*1.e9

        E_1 = sum(S[0,:3])
        E_2 = sum(S[1,:3])
        E_3 = sum(S[2,:3])
        beta_R = E_1 + E_2 + E_3

        PT.append([P_mod/1.e9, T])
        S_sets.append([beta_R, E_2, E_3,
                       S[0,0], S[1,1], S[2,2],
                       S[3,3], S[4,4], S[5,5]])

    PT = np.array(PT)
    S_sets = np.array(S_sets).T

    S_prms = []
    for Sij in S_sets:
        p = curve_fit(fn_invcompliance, PT, 1./Sij, [100., 8., 4., -0.3/100., 0.],
                      bounds = ((0, 0, 0, -0.02, -2.), (1000, 15, 15, 0.00, 0.1)))[0]
        a, b, c, d, _, _, _ = eos.aeos.Sijkl_derivs_to_coeffs([p[0], p[1], p[2], p[3], 10., 800., 300.])


        if (-2*(p[1] - p[2])/p[3]*p[1]/p[0] < 1.):
            # Heuristic
            f = 10. # >> 1
            print('yoyoyo')
            p[3] = (p[2] - p[1])*p[1]/p[0]/f
        S_prms.append(p)
        print(p)
    S_prms = np.array(S_prms)

    pressures = np.linspace(-40, 150, 1001)
    temperatures = pressures*0. + 300.
    PT1 = np.array([pressures, temperatures]).T
    for i in range(9):
        plt.plot(pressures, fn_invcompliance(PT1, *S_prms[i,:5]), label=labels[i])

    for i in range(9):
        plt.scatter(PT[:,0], 1./S_sets[i])
    plt.legend()
    plt.show()

    """
    Modifying pressures and temperatures
    """
    S_dict = [{'invS_0': p[0]*1.e9,
               'invSprime_0': p[1],
               'invSprime_inf': p[2],
               'invSdprime_0': p[3]/1.e9,
               'dinvSdT_0': p[4]*1.e9, #
               'Theta': 905.*0.806}
              for p in S_prms]

    params = {'symmetry_type': 'orthorhombic',
              'equation_of_state': 'aeos',
              'formula': {'Mg': 1.,
                          'Si': 1.,
                          'O': 3.},
              'n': 5.,
              'Z': 4.,
              'T_0': 300.,
              'P_0': 200.e9,
              'G_0': 0.,
              'Cv': [1300.*0.806, 3.*5.*8.31446, 0.],
              'molar_mass': 0.1003887, # for MgSiO3
              'unit_cell_vectors': [np.array([4.7759e-10, 0., 0.]),
                                    np.array([0., 4.9284e-10, 0.]),
                                    np.array([0., 0., 6.8972e-10])], # Tange et al., 2009 for MgSiO3
              'beta_T_reuss': S_dict[0],
              'E_2': S_dict[1],
              'E_3': S_dict[2],
              'S_T': {'11': S_dict[3],
                      '22': S_dict[4],
                      '33': S_dict[5],
                      '44': S_dict[6],
                      '55': S_dict[7],
                      '66': S_dict[8]}}

    bridgmanite = AnisotropicMineral(params)

    pressures = np.linspace(0., 350.e9, 101)
    for T in [300., 1500., 3000.]:
        temperatures = pressures*0. + T
        vp_reuss, vp_voigt, vp_vrh = bridgmanite.evaluate(['p_wave_velocity_reuss', 'p_wave_velocity_voigt', 'p_wave_velocity_vrh'], pressures, temperatures)
        vs_reuss, vs_voigt, vs_vrh = bridgmanite.evaluate(['shear_wave_velocity_reuss', 'shear_wave_velocity_voigt', 'shear_wave_velocity_vrh'], pressures, temperatures)
        vphi_reuss, vphi_voigt, vphi_vrh = bridgmanite.evaluate(['bulk_sound_velocity_reuss', 'bulk_sound_velocity_voigt', 'bulk_sound_velocity_vrh'], pressures, temperatures)
        vp_slb, vs_slb, vphi_slb = bdg_SLB.evaluate(['p_wave_velocity', 'shear_wave_velocity', 'bulk_sound_velocity'], pressures, temperatures)

        plt.fill_between(pressures/1.e9, vs_reuss, vs_voigt)
        plt.plot(pressures/1.e9, vs_vrh, color='black', linewidth=0.5)
        plt.fill_between(pressures/1.e9, vp_reuss, vp_voigt)
        plt.plot(pressures/1.e9, vp_vrh, color='black', linewidth=0.5)
        plt.fill_between(pressures/1.e9, vphi_reuss, vphi_voigt)
        plt.plot(pressures/1.e9, vphi_vrh, color='black', linewidth=0.5)
        plt.plot(pressures/1.e9, vp_slb, color='red', linestyle=':')
        plt.plot(pressures/1.e9, vs_slb, color='red', linestyle=':')
        plt.plot(pressures/1.e9, vphi_slb, color='red', linestyle=':')

    plt.show()


    temperatures = np.linspace(10., 3000., 101)
    for P in [1.e9, 10.e9, 100.e9, 200.e9]:
        pressures = temperatures*0. + P
        gr = bridgmanite.evaluate(['universal_elastic_anisotropy'], pressures, temperatures)[0]
        plt.plot(temperatures, gr, label='{0}'.format(P/1.e9))
        #gr = bdg_HP.evaluate(['grueneisen_parameter'], pressures, temperatures)[0]
        #plt.plot(temperatures, gr, linestyle=':')
    plt.legend()
    plt.show()
    """
    END MODIFY
    """
    temperatures = np.linspace(10., 3000., 101)
    for P in [1.e9, 10.e9, 100.e9, 200.e9]:
        pressures = temperatures*0. + P
        K_T, alpha = bridgmanite.evaluate(['isothermal_bulk_modulus_reuss', 'alpha'],
                                          pressures, temperatures)
        plt.plot(temperatures, cumtrapz(K_T*alpha, temperatures, initial=0))
    plt.show()

    pressures = data[:,0]*1.e9
    #pressures = np.linspace(-40.e9, 150.e9, 101)
    temperatures = 300. + 0.*pressures
    SN_model = []
    K_T, V, STs = bridgmanite.evaluate(['isothermal_bulk_modulus_reuss', 'V', 'isentropic_elastic_compliance_tensor'], pressures, temperatures)
    beta_V = np.array([sum(sum(ST[0:3,0:3])) for ST in STs])
    E_2 = np.array([sum(ST[1,:3]) for ST in STs])
    E_3 = np.array([sum(ST[2,:3]) for ST in STs])
    ST_ii = np.array([np.diag(ST) for ST in STs])
    #plt.plot(pressures/1.e9, 1.e-9/beta_V, label='1/{0}'.format(labels[0]))
    #plt.plot(pressures/1.e9, 1.e-9/E_2, label='1/{0}'.format(labels[1]))
    #plt.plot(pressures/1.e9, 1.e-9/E_3, label='1/{0}'.format(labels[2]))
    #for i in range(6):
    #    plt.plot(pressures/1.e9, 1.e-9/ST_ii[:,i], label='1/{0}'.format(labels[3+i]))

    S_sets_model = np.array([beta_V, E_2, E_3,
                             ST_ii[:,0], ST_ii[:,1], ST_ii[:,2],
                             ST_ii[:,3], ST_ii[:,4], ST_ii[:,5]])*1.e9

    for i in range(9):
        plt.scatter(data[:,0], 1./S_sets[i] - 1./S_sets_model[i], label='1/{0}'.format(labels[i])) #
    plt.legend()
    plt.show()

    pressures = np.linspace(0., 150.e9, 101)
    temperatures = pressures*0. + 300.
    vs = bridgmanite.evaluate(['unit_cell_vectors'], pressures, temperatures)[0]
    plt.plot(pressures/1.e9, vs[:,0,0]*1e10, label='a')
    plt.plot(pressures/1.e9, vs[:,1,1]*1e10, label='b')
    plt.plot(pressures/1.e9, vs[:,2,2]*1e10, label='c')


    data = np.loadtxt('data/Tange_2012_MgSiO3_bridgmanite_EoS.dat')
    mask = [i for i, T in enumerate(data[:,1]) if np.abs(T - 300) < 1.e-10]
    plt.scatter(data[:,13][mask], data[:,3][mask])
    plt.scatter(data[:,13][mask], data[:,5][mask])
    plt.scatter(data[:,13][mask], data[:,7][mask])
    plt.xlabel('Pressure (GPa)')
    plt.ylabel('Length ($\\mathrm{\\AA}$)')
    plt.legend()
    plt.show()

    pressures = np.linspace(0., 150.e9, 101)
    for T in [10., 300., 1500., 2500., 3500.]:
        temperatures = pressures*0. + T
        rho, alpha = bridgmanite.evaluate(['rho', 'alpha'], pressures, temperatures)
        plt.plot(pressures, alpha, label='{0} K'.format(int(T)))
    plt.legend()
    plt.show()


    P = 38.e9
    temperatures = np.linspace(10., 4000., 101)
    SN_model = []
    V, SNs = bridgmanite.evaluate(['V', 'isentropic_elastic_compliance_tensor'], P + 0.*temperatures, temperatures)
    beta_V = np.array([sum(sum(SN[0:3,0:3])) for SN in SNs])
    E_2 = np.array([sum(SN[1,:3]) for SN in SNs])
    E_3 = np.array([sum(SN[2,:3]) for SN in SNs])
    SN_ii = np.array([np.diag(SN) for SN in SNs])
    plt.plot(temperatures, 1.e-9/beta_V)
    plt.plot(temperatures, 1.e-9/E_2)
    plt.plot(temperatures, 1.e-9/E_3)
    plt.plot(temperatures, 1.e-9/SN_ii)

    # High temperature params
    data = np.loadtxt('data/Oganov_et_al_2001.dat')
    SN_sets = []
    for datum in data[0:3]:
        P, T, V, a, b, c, C11, _, C22, _, C33, _, C12, _, C13, _, C23, _, C44, C55, C66, KT, KS, G, _, _ = datum
        C = np.array([[C11, C12, C13, 0, 0, 0],
                      [C12, C22, C23, 0, 0, 0],
                      [C13, C23, C33, 0, 0, 0],
                      [0, 0, 0, C44, 0, 0],
                      [0, 0, 0, 0, C55, 0],
                      [0, 0, 0, 0, 0, C66]])

        SN = np.linalg.pinv(C)
        EN_1 = sum(SN[0,:3])
        EN_2 = sum(SN[1,:3])
        EN_3 = sum(SN[2,:3])
        betaN_R = EN_1 + EN_2 + EN_3
        SN_sets.append([betaN_R, EN_2, EN_3,
                        SN[0,0], SN[1,1], SN[2,2],
                        SN[3,3], SN[4,4], SN[5,5]])

    SN_sets = np.array(SN_sets).T

    for i in range(9):
        plt.scatter(data[0:3,1], 1./SN_sets[i], label='1/{0}'.format(labels[i]))
    plt.legend()
    plt.show()


    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig = plt.figure(figsize=(12, 5))
    ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
    pressures = np.linspace(0., 150.e9, 101)
    for i, T in enumerate([300., 1000., 2000., 3000., 4000.]):
        temperatures = T + 0.*pressures
        G_R, G_V, G_VRH = bridgmanite.evaluate(['shear_modulus_reuss', 'shear_modulus_voigt', 'shear_modulus_vrh'], pressures, temperatures)
        K_R, K_V, K_VRH = bridgmanite.evaluate(['isentropic_bulk_modulus_reuss', 'isentropic_bulk_modulus_voigt', 'isentropic_bulk_modulus_vrh'], pressures, temperatures)

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

    temperatures = np.linspace(10., 4000., 1001)
    for P in [1.e5, 20.e9, 50.e9, 100.e9, 200.e9]: #, 4000., 5000.]:
        pressures = temperatures*0. + P
        S1s, Cp1s, Cv1s, volumes, gibbs = bridgmanite.evaluate(['S', 'C_p', 'C_v', 'V', 'gibbs'], pressures, temperatures)

        S2s, Cp2s, Cv2s, volumes2, gibbs2 = bdg_SLB.evaluate(['S', 'C_p', 'C_v', 'V', 'gibbs'], pressures, temperatures)

        plt.plot(temperatures, Cp1s, label='{0} GPa'.format(P/1.e9))
        plt.plot(temperatures, Cp2s, linestyle=':')

    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(18, 5))
    ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

    pressures = np.linspace(-5.e9, 150.e9, 101)
    for T in [300., 1000., 2000.]:
        temperatures = T + 0.*pressures
        alphas, volumes, KT = bridgmanite.evaluate(['alpha', 'V', 'isothermal_bulk_modulus_reuss'], pressures, temperatures)
        alphas_HP, volumes_HP, KT_HP = bdg_HP.evaluate(['alpha', 'V', 'K_T'], pressures, temperatures)
        alphas_SLB, volumes_SLB, KT_SLB = bdg_SLB.evaluate(['alpha', 'V', 'K_T'], pressures, temperatures)

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

    for i in range(3):
        ax[i].set_xlabel('P (GPa)')
        ax[i].set_xlim(0., 150.)
        ax[i].legend()
    plt.show()
