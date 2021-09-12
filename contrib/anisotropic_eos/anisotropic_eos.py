# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""

anisotropic_eos
---------------

"""
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt

import burnman_path  # adds the local burnman directory to the path

import burnman
from burnman.minerals.SLB_2011 import forsterite, periclase

assert burnman_path  # silence pyflakes warning



from anisotropicmineral import AnisotropicMineral


run_isotropic = False
if run_isotropic:
    if False:
        per = burnman.minerals.SLB_2011.periclase()
        per.set_state(1.e5, 300.)

        C44_over_K0 = per.shear_modulus/per.isothermal_bulk_modulus
        S44_over_beta0 = 1./C44_over_K0
        V0 = per.V

        pressures = np.linspace(1.e5, 400.e9, 401)

        from scipy.optimize import brentq

        def delta_Pth(T, P, P_th):
            per.set_state(P, T)
            P0 = per.method.pressure(per.params['T_0'], per.V, per.params)
            return (P - P0) - P_th

        def delta_V(T, V, P_th):
            P = per.method.pressure(T, V, per.params)
            P0 = per.method.pressure(per.params['T_0'], V, per.params)
            return (P - P0) - P_th

        fig = plt.figure(figsize=(8, 4))
        ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

        # constant Pth
        S44_over_betas = []
        Vs = []
        Ts = []
        P_ths = np.array([0., 5.e9, 10.e9])
        for P_th in P_ths:
            temperatures = np.empty_like(pressures)

            for i, P in enumerate(pressures):
                temperatures[i] = brentq(delta_Pth, 200., 3000., args=(P, P_th))

            K, C44, V = per.evaluate(['isothermal_bulk_modulus', 'shear_modulus', 'V'],
                                     pressures, temperatures)
            Ts.append(temperatures)
            S44_over_betas.append(K/C44)
            Vs.append(V)

        Vs = np.array(Vs)
        S44_over_betas = np.array(S44_over_betas)

        for i in range(len(Vs)):
            ax[0].plot(np.log(Vs[i]/V0), S44_over_betas[i]/(S44_over_beta0), label=f'$P_{{th}}$ = {P_ths[i]/1.e9} GPa')

        for i in range(len(Vs[0])):
            if i==0:
                ax[0].plot(np.log(Vs[:,i]/V0), S44_over_betas[:,i]/(S44_over_beta0), linestyle=':', color='k', label='$P = P_c$ (20 GPa spacing)')
            if i%20 == 0 and i != 0:
                ax[0].plot(np.log(Vs[:,i]/V0), S44_over_betas[:,i]/(S44_over_beta0), linestyle=':', color='k')


        # constant V
        P_ths2 = np.linspace(0., 10.e9, 11)

        fs = np.linspace(-0.6, 0., 4)
        for f in fs:
            volume = per.params['V_0'] * np.exp(f)
            temperatures = np.empty_like(P_ths2)
            pressures = np.empty_like(P_ths2)

            for i, P_th in enumerate(P_ths2):
                temperatures[i] = brentq(delta_V, 200., 3000., args=(volume, P_th))
                pressures[i] = per.method.pressure(temperatures[i], volume, per.params)

            K, C44, V = per.evaluate(['isothermal_bulk_modulus', 'shear_modulus', 'V'],
                                     pressures, temperatures)
            S44_over_beta = K/C44
            ax[1].plot(P_ths2/1.e9, S44_over_beta/(S44_over_beta0), label=f'$f$ = {f}')

        for i in range(len(Vs[0])):
            if i == 0:
                ax[1].plot(P_ths/1.e9, S44_over_betas[:,i]/(S44_over_beta0), linestyle=':', color='k', label='$P = P_c$ (20 GPa spacing)')

            if i%20 == 0 and i != 0:
                ax[1].plot(P_ths/1.e9, S44_over_betas[:,i]/(S44_over_beta0), linestyle=':', color='k')


        ax[0].plot([0, -1], [1, 2.18], linestyle='--')
        ax[0].set_xlim(0.1, -0.6)
        ax[0].set_ylim(0.9, 2)

        ax[0].set_xlabel('f = ln(V/V0)')
        ax[0].set_ylabel('$S_{44}/\\beta_{RT}$')
        ax[0].legend()

        ax[1].set_xlim(0., 10.)
        ax[1].set_ylim(0.9, 2)

        ax[1].set_xlabel('$P_{th}$ (GPa)')
        ax[1].set_ylabel('$S_{44}/\\beta_{RT}$')
        ax[1].legend()

        fig.set_tight_layout(True)
        plt.show()

    per = burnman.minerals.SLB_2011.periclase()
    per.set_state(1.e5, 300.)
    c4410 = per.isothermal_bulk_modulus / per.shear_modulus


    a = np.cbrt(per.params['V_0'])
    cell_parameters = np.array([a, a, a, 90, 90, 90])

    f_order = 3
    Pth_order = 1
    constants = np.zeros((6, 6, f_order+1, Pth_order+1))

    constants[5,5,1,0] = 1.23230955
    constants[5,5,2,0] = -1.42976418
    constants[5,5,3,0] = 0.14282993

    constants[5,5,0,1] = 0.05614165e-11
    constants[5,5,1,1] = 1.70739084e-11

    constants[:3,:3,:,:] = -constants[5,5,:,:]/6.
    constants[0,0,:,:] = constants[5,5,:,:]/3.
    constants[1,1,:,:] = constants[5,5,:,:]/3.
    constants[2,2,:,:] = constants[5,5,:,:]/3.
    constants[3,3,:,:] = constants[5,5,:,:]
    constants[4,4,:,:] = constants[5,5,:,:]

    constants[:3,:3,1,0] += 1./9.


    m = AnisotropicMineral(per, cell_parameters, constants)


    fig = plt.figure(figsize=(8, 4))
    ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

    pressures = np.linspace(1.e5, 100.e9, 101)
    G_iso = np.empty_like(pressures)
    G_aniso = np.empty_like(pressures)
    C11 = np.empty_like(pressures)
    C12 = np.empty_like(pressures)
    C44 = np.empty_like(pressures)
    C44_slb = np.empty_like(pressures)

    f = np.empty_like(pressures)
    dXdf = np.empty_like(pressures)

    temperatures = [300., 1000., 2000.]
    for T in temperatures:
        for i, P in enumerate(pressures):

            per.set_state(P, T)
            m.set_state(P, T)
            G_iso[i] = per.isothermal_bulk_modulus / per.shear_modulus
            G_aniso[i] = 1./np.sum(m.isothermal_compliance_tensor[:3,:3]) / m.isothermal_stiffness_tensor[5,5]

        l = ax[0].plot(pressures/1.e9, 1./G_iso, label=f'SLB2011 {T} K', linestyle='--')
        ax[0].plot(pressures/1.e9, 1./G_aniso, label=f'anisotropic {T} K', color=l[0].get_color(), linestyle='-')

    temperatures = np.linspace(10., 2000., 101)
    P = 1.e5
    for i, T in enumerate(temperatures):
        m.set_state(P, T)
        per.set_state(P, T)
        C11[i] = m.isothermal_stiffness_tensor[0,0]
        C12[i] = m.isothermal_stiffness_tensor[0,1]
        C44[i] = m.isothermal_stiffness_tensor[3,3]

        C44_slb[i] = per.shear_modulus
    ax[1].plot(temperatures, C11/1.e9, label='C11')
    ax[1].plot(temperatures, C12/1.e9, label='C12')
    l = ax[1].plot(temperatures, C44/1.e9, label='C44')
    ax[1].plot(temperatures, C44_slb/1.e9, label='C44 (SLB)', color=l[0].get_color(), linestyle='--')

    ax[0].set_xlabel('Pressure (GPa)')
    ax[0].set_ylabel('$C_{44} / K_{RT}$')
    ax[0].legend()

    ax[1].set_xlabel('Temperature (K)')
    ax[1].set_ylabel('Elastic modulus (GPa)')
    ax[1].legend()
    fig.set_tight_layout(True)
    plt.show()

    run_fitting = False
    if run_fitting:
        def isotropic_shear_misfit(x):
            f_order = 3
            Pth_order = 1
            constants = np.zeros((6, 6, f_order+1, Pth_order+1))

            constants[5,5,1,0] = x[0]
            constants[5,5,2,0] = x[1]
            constants[5,5,3,0] = x[2]

            constants[5,5,0,1] = x[3]*1e-11
            constants[5,5,1,1] = x[4]*1e-11

            constants[:3,:3,:,:] = -constants[5,5,:,:]/6.
            constants[0,0,:,:] = constants[5,5,:,:]/3.
            constants[1,1,:,:] = constants[5,5,:,:]/3.
            constants[2,2,:,:] = constants[5,5,:,:]/3.
            constants[3,3,:,:] = constants[5,5,:,:]
            constants[4,4,:,:] = constants[5,5,:,:]

            constants[:3,:3,1,0] += 1./9.

            m = AnisotropicMineral(per, cell_parameters, constants)

            Ps = np.linspace(1.e5, 100.e9, 21)
            pressures = list(Ps)
            temperatures = list(300. + 0.*Ps)

            Ps = np.linspace(10.e9, 100.e9, 21)
            pressures.extend(list(Ps))
            temperatures.extend(list(1000. + 0.*Ps))

            Ps = np.linspace(20.e9, 100.e9, 21)
            pressures.extend(list(Ps))
            temperatures.extend(list(2000. + 0.*Ps))

            chisqr = 0.
            for i in range(len(pressures)):
                per.set_state(pressures[i], temperatures[i])
                m.set_state(pressures[i], temperatures[i])
                G_iso = per.shear_modulus
                G_aniso = m.isothermal_stiffness_tensor[5,5]

                chisqr += np.power((G_iso / G_aniso) - 1., 2.)

            return chisqr

        from scipy.optimize import minimize

        sol = minimize(isotropic_shear_misfit, [c4410, -1.44, 0.15, 0.12, 1.6])
        print(sol)

run_cubic = False
if run_cubic:

    per = burnman.minerals.SLB_2011.periclase()
    per.set_state(1.e5, 300.)
    a = np.cbrt(per.params['V_0'])
    cell_parameters = np.array([a, a, a, 90, 90, 90])

    beta_RT = per.beta_T

    per_data = np.loadtxt('data/isentropic_stiffness_tensor_periclase.dat')

    def make_cubic_mineral_from_parameters(x):
        f_order = 3
        Pth_order = 1
        constants = np.zeros((6, 6, f_order+1, Pth_order+1))

        S11_0 = x[0]
        dS11df = x[1]
        d2S11df2 = x[2]
        dS11dPth = x[3] * 1.e-11
        d2S11dfdPth = x[4] * 1.e-11


        S44_0 = x[5]
        dS44df = x[6]
        d2S44df2 = x[7]
        dS44dPth = x[8] * 1.e-11
        d2S44dfdPth = x[9] * 1.e-11


        S12_0 = (1. - 3.*S11_0)/6.
        dS12df = -dS11df/2.
        d2S12df2 = -d2S11df2/2.
        dS12dPth = -dS11dPth/2.
        d2S12dfdPth = -d2S11dfdPth/2.

        constants[:3,:3,1,0] = S12_0
        constants[:3,:3,2,0] = dS12df
        constants[:3,:3,3,0] = d2S12df2
        constants[:3,:3,0,1] = dS12dPth
        constants[:3,:3,1,1] = d2S12dfdPth
        for i in range(3):
            constants[i,i,1,0] = S11_0
            constants[i,i,2,0] = dS11df
            constants[i,i,3,0] = d2S11df2

            constants[i,i,0,1] = dS11dPth
            constants[i,i,1,1] = d2S11dfdPth

        for i in range(3, 6):
            constants[i,i,1,0] = S44_0
            constants[i,i,2,0] = dS44df
            constants[i,i,3,0] = d2S44df2

            constants[i,i,0,1] = dS44dPth
            constants[i,i,1,1] = d2S44dfdPth


        m = AnisotropicMineral(per, cell_parameters, constants)
        return m

    run_fitting = False
    if run_fitting:

        def cubic_misfit(x):
            m = make_cubic_mineral_from_parameters(x)

            chisqr = 0.
            for d in per_data:
                T, PGPa, Perr, rho, rhoerr, C11S, C11Serr, C12S, C12Serr, C44S, C44Serr = d

                P = PGPa * 1.e9

                m.set_state(P, T)
                chisqr += np.power((m.isentropic_stiffness_tensor[0,0]/1.e9 - C11S)/C11Serr, 2.)
                chisqr += np.power((m.isentropic_stiffness_tensor[0,1]/1.e9 - C12S)/C12Serr, 2.)
                chisqr += np.power((m.isentropic_stiffness_tensor[3,3]/1.e9 - C44S)/C44Serr, 2.)

                #for xi in x:
                #    chisqr += xi*xi/10.


            print(x)
            print(chisqr)

            return chisqr

        from scipy.optimize import minimize

        """
        S11_0 = x[0]
        dS11df = x[1]
        d2S11df2 = x[2]
        dS11dPth = x[3] * 1.e-11
        d2S11dfdPth = x[4] * 1.e-11


        S44_0 = x[5]
        dS44df = x[6]
        d2S44df2 = x[7]
        dS44dPth = x[8] * 1.e-11
        d2S44dfdPth = x[9] * 1.e-11
        """

        sol = minimize(cubic_misfit, [1./3., 0., 0., 0., 0., 1., 0., 0., 0., 0.])

        print(sol)

        make_cubic_mineral_from_parameters(sol.x)

    m = make_cubic_mineral_from_parameters([0.64434719,  0.97982023,  2.28703418,  0.04069744,  0.83313498 , 1.02999379,
                                            -3.39390829, -2.02738898,  0.06480835,  0.52939447])


    fig = plt.figure(figsize=(8, 8))
    ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

    pressures = np.linspace(1.e5, 30.e9, 101)
    G_iso = np.empty_like(pressures)
    G_aniso = np.empty_like(pressures)
    C11 = np.empty_like(pressures)
    C12 = np.empty_like(pressures)
    C44 = np.empty_like(pressures)
    C44_slb = np.empty_like(pressures)

    f = np.empty_like(pressures)
    dXdf = np.empty_like(pressures)

    temperatures = [300., 500., 700., 900.]
    for T in temperatures:
        for i, P in enumerate(pressures):

            per.set_state(P, T)
            m.set_state(P, T)
            C11[i] = m.isentropic_stiffness_tensor[0,0]
            C12[i] = m.isentropic_stiffness_tensor[0,1]
            C44[i] = m.isentropic_stiffness_tensor[3,3]

            C44_slb[i] = per.shear_modulus

        ax[0].plot(pressures/1.e9, C11/1.e9, label=f'{T} K')
        ax[1].plot(pressures/1.e9, C12/1.e9, label=f'{T} K')
        l = ax[2].plot(pressures/1.e9, C44/1.e9, label=f'{T} K')

        ax[2].plot(pressures/1.e9, C44_slb/1.e9, label='C44 (SLB)', color=l[0].get_color(), linestyle='--')

        # T, PGPa, Perr, rho, rhoerr, C11S, C11Serr, C12S, C12Serr, C44S, C44Serr
        T_data = np.array([[d[1], d[5], d[7], d[9]] for d in per_data if np.abs(d[0] - T) < 1])

        ax[0].scatter(T_data[:,0], T_data[:,1])
        ax[1].scatter(T_data[:,0], T_data[:,2])
        ax[2].scatter(T_data[:,0], T_data[:,3])


    temperatures = np.linspace(10., 2000., 101)
    P = 1.e5
    for i, T in enumerate(temperatures):
        m.set_state(P, T)
        per.set_state(P, T)
        C11[i] = m.isentropic_stiffness_tensor[0,0]
        C12[i] = m.isentropic_stiffness_tensor[0,1]
        C44[i] = m.isentropic_stiffness_tensor[3,3]

        C44_slb[i] = per.shear_modulus
    ax[3].plot(temperatures, C11/1.e9, label='C11_N')
    ax[3].plot(temperatures, C12/1.e9, label='C12_N')
    l = ax[3].plot(temperatures, C44/1.e9, label='C44_N')
    ax[3].plot(temperatures, C44_slb/1.e9, label='C44 (SLB)', color=l[0].get_color(), linestyle='--')

    # T, PGPa, Perr, rho, rhoerr, C11S, C11Serr, C12S, C12Serr, C44S, C44Serr
    LP_data = np.array([[d[0], d[5], d[7], d[9]] for d in per_data if d[1] < 0.1])

    ax[3].scatter(LP_data[:,0], LP_data[:,1])
    ax[3].scatter(LP_data[:,0], LP_data[:,2])
    ax[3].scatter(LP_data[:,0], LP_data[:,3])

    for i in range(3):
        ax[0].set_xlabel('Pressure (GPa)')

    ax[0].set_ylabel('$C_{N 11}$')
    ax[1].set_ylabel('$C_{N 12}$')
    ax[2].set_ylabel('$C_{N 44}$')

    ax[3].set_xlabel('Temperature (K)')
    ax[3].set_ylabel('Elastic modulus (GPa)')

    for i in range(4):
        ax[i].legend()

    fig.set_tight_layout(True)
    plt.show()


run_orthorhombic = True
if run_orthorhombic:


    formula = 'Mg1.8Fe0.2SiO4'
    formula = burnman.processchemistry.dictionarize_formula(formula)

    # Define the unit cell lengths and unit cell volume.
    # These are taken from Abramson et al., 1997
    Z = 4.
    cell_lengths_angstrom = np.array([4.7646, 10.2296, 5.9942])
    cell_lengths = cell_lengths_angstrom*np.cbrt(burnman.constants.Avogadro/Z/1.e30)

    san_carlos_params = {'name': 'San Carlos olivine',
                         'formula': formula,
                         'equation_of_state': 'slb3',
                         'F_0': 0.0,
                         'V_0': np.prod(cell_lengths),
                         'K_0': 1.263e+11, # Abramson et al. 1997
                         'Kprime_0': 4.28, # Abramson et al. 1997
                         'Debye_0': 809.1703, # Fo in SLB2011
                         'grueneisen_0': 0.99282, # Fo in SLB2011
                         'q_0': 2.10672, # Fo in SLB2011
                         'G_0': 81.6e9,
                         'Gprime_0': 1.46257,
                         'eta_s_0': 2.29972,
                         'n': sum(formula.values()),
                         'molar_mass': burnman.processchemistry.formula_mass(formula)}

    san_carlos_property_modifiers = [['linear', {'delta_E': 0.0,
                                                 'delta_S': 26.76*0.1 - 2.*burnman.constants.gas_constant*(0.1*np.log(0.1) + 0.9*np.log(0.9)),
                                                 'delta_V': 0.0}]]


    ol = burnman.Mineral(params=san_carlos_params,
                         property_modifiers=san_carlos_property_modifiers)
    ol.set_state(1.e5, 300.)

    print(ol.params['V_0'])

    ol_cell_parameters = np.array([cell_lengths[0],
                                  cell_lengths[1],
                                  cell_lengths[2],
                                  90, 90, 90])


    ol_data = np.loadtxt('data/Mao_et_al_2015_ol.dat')

    def make_orthorhombic_mineral_from_parameters(x):
        f_order = 3
        Pth_order = 1
        constants = np.zeros((6, 6, f_order+1, Pth_order+1))

        # First, we want to fit some isotropic constants.
        # For the SLB EoS, these are:

        # volume, which we don't change

        ol.params['K_0'] = x[0]*1.263e+11 # Abramson et al. 1997
        ol.params['Kprime_0'] = x[1]*4.28 # Abramson et al. 1997
        ol.params['Debye_0'] = x[2]*809.1703 # Fo in SLB2011
        ol.params['grueneisen_0'] = x[3]*0.99282 # Fo in SLB2011
        ol.params['q_0'] = x[4]*2.10672 # Fo in SLB2011


        # Next, each of the eight independent elastic tensor component get their turn.
        # We arbitrarily choose S[2,3] as the ninth component, which is determined by the others.
        i = 5
        for (p, q) in ((1, 1),
                       (2, 2),
                       (3, 3),
                       (4, 4),
                       (5, 5),
                       (6, 6),
                       (1, 2),
                       (1, 3)):
            for (m, n) in ((1, 0),
                           (2, 0),
                           (3, 0)):
                constants[p-1, q-1, m, n] = x[i]
                constants[q-1, p-1, m, n] = x[i]
                i += 1

            for (m, n) in ((0, 1),
                           (1, 1)):
                constants[p-1, q-1, m, n] = x[i]*1.e-11
                constants[q-1, p-1, m, n] = x[i]*1.e-11
                i += 1

        assert i == 45 # 45 parameters

        # Fill the values for the dependent element c[2,3]
        constants[1,2,1,0] = (1. - np.sum(constants[:3,:3,1,0])) / 2.
        constants[1,2,2,0] = - np.sum(constants[:3,:3,2,0]) / 2.
        constants[1,2,3,0] = - np.sum(constants[:3,:3,3,0]) / 2.
        constants[1,2,0,1] = - np.sum(constants[:3,:3,0,1]) / 2.
        constants[1,2,1,1] = - np.sum(constants[:3,:3,1,1]) / 2.

        # And for c[3,2]
        constants[2,1,:,:] = constants[1,2,:,:]

        m = AnisotropicMineral(ol, ol_cell_parameters, constants)
        return m

    run_fitting = False
    if run_fitting:

        def orthorhombic_misfit(x):
            m = make_orthorhombic_mineral_from_parameters(x)

            chisqr = 0.
            for d in ol_data:

                TK, PGPa, rho, rhoerr = d[:4]
                C11, C11err = d[4:6]
                C22, C22err = d[6:8]
                C33, C33err = d[8:10]
                C44, C44err = d[10:12]
                C55, C55err = d[12:14]
                C66, C66err = d[14:16]
                C12, C12err = d[16:18]
                C13, C13err = d[18:20]
                C23, C23err = d[20:22]

                PPa = PGPa * 1.e9

                m.set_state(PPa, TK)


                chisqr += np.power((m.density/1000. - rho)/rhoerr, 2.)

                CN = m.isentropic_stiffness_tensor/1.e9

                chisqr += np.power((CN[0,0] - C11)/C11err, 2.)
                chisqr += np.power((CN[1,1] - C22)/C22err, 2.)
                chisqr += np.power((CN[2,2] - C33)/C33err, 2.)
                chisqr += np.power((CN[3,3] - C44)/C44err, 2.)
                chisqr += np.power((CN[4,4] - C55)/C55err, 2.)
                chisqr += np.power((CN[5,5] - C66)/C66err, 2.)
                chisqr += np.power((CN[0,1] - C12)/C12err, 2.)
                chisqr += np.power((CN[0,2] - C13)/C13err, 2.)
                chisqr += np.power((CN[1,2] - C23)/C23err, 2.)

            if np.isnan(chisqr):
                print(d)
                raise Exception("Noooo, there was a nan")

            print(x)
            print(chisqr)

            return chisqr

        from scipy.optimize import minimize

        guesses = [ 1.00258822e+00,  1.00005738e+00,  1.00627254e+00,  1.00792761e+00,
                                                       1.00313859e+00,  4.43157071e-01, -1.11135061e+00, -5.17744084e-01,
                                                       -5.11773354e-02,  6.56537659e-01,  7.86791564e-01, -8.85454484e-01,
                                                       5.88990648e+00,  4.82142252e-01, -2.34709601e-01,  6.48291161e-01,
                                                       -1.30313295e+00,  3.50033709e+00,  7.27762794e-02,  1.09585960e+00,
                                                       1.98105535e+00, -6.98453134e-01,  1.99817627e+01, -5.54380359e-01,
                                                       3.52039448e+00,  1.63599202e+00, -1.62546449e+00,  2.60778768e+01,
                                                       -8.22413923e-01,  3.57055209e+00,  1.56332611e+00, -1.11671956e+00,
                                                       1.66582410e+01, -6.47625372e-01,  7.45797943e+00, -1.24949320e-01,
                                                       5.98120393e-01, -1.50767709e-03, -7.89838698e-02, -3.82606391e-01,
                                                       -9.65961899e-02,  3.63504952e-01, -1.42204731e+00, -8.21265777e-02,
                                                       1.64681341e-01]

        sol = minimize(orthorhombic_misfit, guesses)

        print(sol)

        make_orthorhombic_mineral_from_parameters(sol.x)


    # Not final solution, but taken while improvement was slowing down.
    m = make_orthorhombic_mineral_from_parameters([ 1.00258822e+00,  1.00005738e+00,  1.00627254e+00,  1.00792761e+00,
                                                   1.00313859e+00,  4.43157071e-01, -1.11135061e+00, -5.17744084e-01,
                                                   -5.11773354e-02,  6.56537659e-01,  7.86791564e-01, -8.85454484e-01,
                                                   5.88990648e+00,  4.82142252e-01, -2.34709601e-01,  6.48291161e-01,
                                                   -1.30313295e+00,  3.50033709e+00,  7.27762794e-02,  1.09585960e+00,
                                                   1.98105535e+00, -6.98453134e-01,  1.99817627e+01, -5.54380359e-01,
                                                   3.52039448e+00,  1.63599202e+00, -1.62546449e+00,  2.60778768e+01,
                                                   -8.22413923e-01,  3.57055209e+00,  1.56332611e+00, -1.11671956e+00,
                                                   1.66582410e+01, -6.47625372e-01,  7.45797943e+00, -1.24949320e-01,
                                                   5.98120393e-01, -1.50767709e-03, -7.89838698e-02, -3.82606391e-01,
                                                   -9.65961899e-02,  3.63504952e-01, -1.42204731e+00, -8.21265777e-02,
                                                   1.64681341e-01])


    fig = plt.figure(figsize=(12, 12))
    ax = [fig.add_subplot(3, 3, i) for i in range(1, 10)]

    pressures = np.linspace(1.e5, 30.e9, 101)
    G_iso = np.empty_like(pressures)
    G_aniso = np.empty_like(pressures)
    C = np.empty((len(pressures), 6, 6))

    f = np.empty_like(pressures)
    dXdf = np.empty_like(pressures)

    i_pq = ((1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (1, 2),
            (1, 3),
            (2, 3))


    temperatures = [300., 500., 750., 900.]
    for T in temperatures:
        for i, P in enumerate(pressures):

            m.set_state(P, T)

            C[i] = m.isentropic_stiffness_tensor


        # TK, PGPa, rho, rhoerr = d[:4]
        #C11, C11err = d[4:6]
        #C22, C22err = d[6:8]
        #C33, C33err = d[8:10]
        #C44, C44err = d[10:12]
        #C55, C55err = d[12:14]
        #C66, C66err = d[14:16]
        #C12, C12err = d[16:18]
        #C13, C13err = d[18:20]
        #C23, C23err = d[20:22]
        T_data = np.array([[d[1],
                            d[4], d[6], d[8],
                            d[10], d[12], d[14],
                            d[16], d[18], d[20]]
                           for d in ol_data if np.abs(d[0] - T) < 1])

        for i, (p, q) in enumerate(i_pq):
            ax[i].plot(pressures/1.e9, C[:, p-1, q-1]/1.e9, label=f'{T} K')
            ax[i].scatter(T_data[:,0], T_data[:,1+i])



    for i, (p, q) in enumerate(i_pq):
        ax[i].set_xlabel('Pressure (GPa)')
        ax[i].set_ylabel(f'$C_{{N {p}{q}}}$')
        ax[i].legend()

    fig.set_tight_layout(True)
    plt.show()


"""
exit()

# Load forsterite data from Kumazawa and Anderson (1969)
fo_data = np.loadtxt('data/Kumazawa_Anderson_1969_fo.dat')

fo_data[:,2] *= 1.e-11 # Mbar^-1 = 10^-11 Pa^-1
fo_data[:,3] *= 1.e-15 # 10^-4 Mbar^-1 / K = 10^-15 Pa^-1 / K
fo_data[:,4] *= 1.e-22 # Mbar^-2 = 10^-22 Pa^-2

inds = tuple(fo_data[:,0:2].astype(int).T - 1)
indsT = (inds[1], inds[0])

S_N = np.zeros((6, 6))
S_N[inds] = fo_data[:,2]
S_N[indsT] = fo_data[:,2]

dSdT_N = np.zeros((6, 6))
dSdT_N[inds] = fo_data[:,3]
dSdT_N[indsT] = fo_data[:,3]

dSdP_N = np.zeros((6, 6))
dSdP_N[inds] = fo_data[:,4]
dSdP_N[indsT] = fo_data[:,4]


print(S_N)
print(dSdT_N)
print(dSdP_N)


# 3) Compute anisotropic properties
# c[i,k,m,n] corresponds to xth[i,k], f-coefficient, Pth-coefficient

f_order = 1
Pth_order = 0
constants = np.zeros((6, 6, f_order+1, Pth_order+1))

beta_RN = np.sum(S_N[:3,:3], axis=(0,1))


fo = forsterite()


S_N *= 1./(fo.params['K_0']*beta_RN)
beta_RN = np.sum(S_N[:3,:3], axis=(0,1))


constants[:,:,1,0] = S_N*fo.params['K_0'] # /beta_RN

cell_parameters = np.array([4.7540, 10.1971, 5.9806, 90., 90., 90.])
vecs = cell_parameters_to_vectors(*cell_parameters)
cell_parameters[:3] *= np.cbrt(fo.params['V_0']/np.linalg.det(vecs))

m = AnisotropicMineral(forsterite(), cell_parameters, constants)

pressure = 1.e5
temperature = 300.
m.set_state(pressure, temperature)

np.set_printoptions(precision=3)
print(m.deformation_gradient_tensor)
print(m.thermal_expansivity_tensor)

print('Compliance tensor')
print(m.isothermal_compliance_tensor)

print('Original compliance tensor')
print(S_N)


print('Mineral isotropic elastic properties:\n')
print('Bulk modulus bounds: {0:.3e} {1:.3e} {2:.3e}'.format(m.isentropic_bulk_modulus_reuss,
                                                            m.isentropic_bulk_modulus_vrh,
                                                            m.isentropic_bulk_modulus_voigt))
print('Shear modulus bounds: {0:.3e} {1:.3e} {2:.3e}'.format(m.shear_modulus_reuss,
                                                             m.shear_modulus_vrh,
                                                             m.shear_modulus_voigt))
print('Universal elastic anisotropy: {0:.4f}\n'
      'Isotropic poisson ratio: {1:.4f}\n'.format(m.isentropic_universal_elastic_anisotropy,
                                                  m.isentropic_isotropic_poisson_ratio))

T = 300.
pressures = np.linspace(1.e5, 10.e9, 101)
temperatures = T + 0.*pressures

a = np.empty((101, 3))
for i, P in enumerate(pressures):
    m.set_state(P, T)
    prms = m.cell_parameters
    Fs = m.deformation_gradient_tensor
    a[i] = np.diag(Fs) #prms[:3]

for i in range(3):
    plt.plot(pressures, a[:,i])
plt.show()
"""
