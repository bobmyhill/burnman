# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""

isotropic_fitting
-----------------

"""
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

        sol = minimize(isotropic_shear_misfit, [c4410, -1.44, 0.15, 0.12, 1.6],
                       method='COBYLA')
        print(sol)
