# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""

orthorhombic_fitting
--------------------

"""
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import burnman_path  # adds the local burnman directory to the path
import burnman

from anisotropicmineral import AnisotropicMineral

from tools import print_table_for_mineral_constants

assert burnman_path  # silence pyflakes warning

formula = 'Mg1.8Fe0.2SiO4'
formula = burnman.processchemistry.dictionarize_formula(formula)
formula_mass = burnman.processchemistry.formula_mass(formula)

# Define the unit cell lengths and unit cell volume.
# These are taken from Abramson et al., 1997
Z = 4.
cell_lengths_angstrom = np.array([4.7646, 10.2296, 5.9942])
cell_lengths_0_guess = cell_lengths_angstrom*np.cbrt(burnman.constants.Avogadro/Z/1.e30)
V_0_guess = np.prod(cell_lengths_0_guess)

ol_data = np.loadtxt('data/Mao_et_al_2015_ol.dat')
ol_1bar_lattice_data = np.loadtxt('data/Singh_Simmons_1976_1bar_lattice_parameters_ol.dat')
ol_1bar_lattice_data_Suzuki = np.loadtxt('data/Suzuki_1975_ol_Kenya_expansion.dat')

fo = burnman.minerals.SLB_2011.forsterite()
fa = burnman.minerals.SLB_2011.fayalite()

def make_orthorhombic_mineral_from_parameters(x):
    f_order = 3
    Pth_order = 2
    constants = np.zeros((6, 6, f_order+1, Pth_order+1))

    san_carlos_params = {'name': 'San Carlos olivine',
                         'formula': formula,
                         'equation_of_state': 'slb3',
                         'F_0': 0.0,
                         'V_0': V_0_guess, # we overwrite this in a second
                         'K_0': 1.263e+11, # Abramson et al. 1997
                         'Kprime_0': 4.28, # Abramson et al. 1997
                         'Debye_0': fo.params['Debye_0']*0.9 + fa.params['Debye_0']*0.1, #
                         'grueneisen_0': 0.99282, # Fo in SLB2011
                         'q_0': 2.10672, # Fo in SLB2011
                         'G_0': 81.6e9,
                         'Gprime_0': 1.46257,
                         'eta_s_0': 2.29972,
                         'n': 7.,
                         'molar_mass': formula_mass}

    san_carlos_property_modifiers = [['linear', {'delta_E': 0.0,
                                                 'delta_S': 26.76*0.1 - 2.*burnman.constants.gas_constant*(0.1*np.log(0.1) + 0.9*np.log(0.9)),
                                                 'delta_V': 0.0}]]

    ol = burnman.Mineral(params=san_carlos_params,
                         property_modifiers=san_carlos_property_modifiers)

    # Overwrite some properties
    ol.params['V_0'] = x[0]*V_0_guess # Abramson et al. 1997
    ol.params['K_0'] = x[1]*1.263e+11 # Abramson et al. 1997
    ol.params['Kprime_0'] = x[2]*4.28 # Abramson et al. 1997
    #ol.params['Debye_0'] = x[3]*809.1703 # Fo in SLB2011 strong tendency to 0
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

        #for (m, n) in ((0, 2),):
        #    constants[p-1, q-1, m, n] = x[i]*1.e-22
        #    constants[q-1, p-1, m, n] = x[i]*1.e-22
        #    i += 1

    assert i == 45 # 40 parameters

    # Fill the values for the dependent element c[2,3]
    constants[1,2,1,0] = (1. - np.sum(constants[:3,:3,1,0])) / 2.
    constants[1,2,2:,0] = - np.sum(constants[:3,:3,2:,0], axis=(0, 1)) / 2.
    constants[1,2,:,1:] = - np.sum(constants[:3,:3,:,1:], axis=(0, 1)) / 2.

    # And for c[3,2]
    constants[2,1,:,:] = constants[1,2,:,:]

    cell_lengths = cell_lengths_0_guess*np.cbrt(ol.params['V_0']/V_0_guess)
    ol_cell_parameters = np.array([cell_lengths[0],
                                   cell_lengths[1],
                                   cell_lengths[2],
                                   90, 90, 90])

    m = AnisotropicMineral(ol, ol_cell_parameters, constants)
    return m

run_fitting = False
sol = []
if run_fitting:

    def orthorhombic_misfit(x, index):
        m = make_orthorhombic_mineral_from_parameters(x)

        chisqr = 0.

        try:
            if index[0][0] % 100 == 1:
                print(repr(x))

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

                CN = m.isentropic_stiffness_tensor/1.e9

                chisqr += np.power((m.density/1000. - rho)/rhoerr, 2.)
                chisqr += np.power((CN[0,0] - C11)/C11err, 2.)
                chisqr += np.power((CN[1,1] - C22)/C22err, 2.)
                chisqr += np.power((CN[2,2] - C33)/C33err, 2.)
                chisqr += np.power((CN[3,3] - C44)/C44err, 2.)
                chisqr += np.power((CN[4,4] - C55)/C55err, 2.)
                chisqr += np.power((CN[5,5] - C66)/C66err, 2.)
                chisqr += np.power((CN[0,1] - C12)/C12err, 2.)
                chisqr += np.power((CN[0,2] - C13)/C13err, 2.)
                chisqr += np.power((CN[1,2] - C23)/C23err, 2.)

            """
            # Data from Singh and Simmons, 1976
            # first data point is at room temperature
            d0 = ol_1bar_lattice_data[0]
            for d in ol_1bar_lattice_data[1:]:
                m.set_state(1.e5, d[0] + 273.15) # T in C

                a = np.diag(m.cell_vectors) / cell_lengths
                a_expt = d[1:4] / d0[1:4]

                # typical error taken from Boufidh et al.
                # If the unit vector is smaller, the relative error is larger
                a_err = 0.001 / d0[1:4]
                for i in range(3):
                    chisqr += np.power((a[i] - a_expt[i])/a_err[i], 2.)
            """


            # Not San Carlos, fo92.3, not fo90.4
            for d in ol_1bar_lattice_data_Suzuki:
                m.set_state(1.e5, d[0] + 273.15) # T in C

                Y = ((np.diag(m.cell_vectors) / np.diag(m.cell_vectors_0)) - 1.)*1.e4
                Y_expt = d[1:4]
                Y_err = 0.05*Y_expt + 1.
                for i in range(3):
                    chisqr += np.power((Y_expt[i] - Y[i])/Y_err[i], 2.)


            #if chisqr < 1500.:
            #    print(chisqr)
            #m.set_state(1.e5, 300)
            #print(np.diag(m.thermal_expansivity_tensor))

            if np.isnan(chisqr):
                print(d, "Noooo, there was a nan")
                chisqr = 1.e7

        except:
            print('There was an exception')
            chisqr = 1.e7
        index[0][0] += 1
        print(index[0][0], chisqr)
        return chisqr


    #guesses = [ 1.00263119e+00,  1.00503329e+00,  7.99519521e-01,  1.40122330e+00, 1.10634879e+00,
    #guesses = [ 1, 1, 1, 1, 1,
    guesses = np.array([ 1.00251449e+00,  9.88647516e-01,  1.00050976e+00,  1.15348177e+00,
        1.47769682e-02,  4.47786840e-01, -9.28130118e-01,  4.73175719e-01,
        2.55952386e-01,  1.20129983e-01,  7.63438290e-01, -1.06450597e+00,
        5.80632935e+00,  5.18769305e-01,  1.89075061e+00,  6.35870834e-01,
       -5.54918950e-01,  1.15219383e+01,  1.70727271e+00,  4.01087083e+00,
        2.01058369e+00, -3.12695375e-01,  1.45938964e+01,  3.71999568e-01,
        2.10897454e+00,  1.62360737e+00, -1.98177749e+00,  2.04428746e+01,
       -1.08862822e+00,  2.11245573e+00,  1.54555199e+00, -1.94706523e+00,
        1.14066000e+01,  3.54333966e+00,  4.81936376e+00, -1.22611088e-01,
        4.26151874e-01, -1.32871927e+00,  4.73979931e-01,  2.07631306e-01,
       -9.80828642e-02,  2.56158987e-01, -2.61268766e+00, -6.88787067e-01,
       -8.02639561e-01])

    i = 0
    sol = minimize(orthorhombic_misfit, guesses, method='COBYLA', args=[[i]])
    print(sol)

do_plotting = True
if do_plotting:
    if run_fitting:
        m = make_orthorhombic_mineral_from_parameters(sol.x)
    else:
        # Not final solution, but taken while improvement was slowing down.
        m = make_orthorhombic_mineral_from_parameters([ 1.00251449e+00,  9.88647516e-01,  1.00050976e+00,  1.15348177e+00,
        1.47769682e-02,  4.47786840e-01, -9.28130118e-01,  4.73175719e-01,
        2.55952386e-01,  1.20129983e-01,  7.63438290e-01, -1.06450597e+00,
        5.80632935e+00,  5.18769305e-01,  1.89075061e+00,  6.35870834e-01,
       -5.54918950e-01,  1.15219383e+01,  1.70727271e+00,  4.01087083e+00,
        2.01058369e+00, -3.12695375e-01,  1.45938964e+01,  3.71999568e-01,
        2.10897454e+00,  1.62360737e+00, -1.98177749e+00,  2.04428746e+01,
       -1.08862822e+00,  2.11245573e+00,  1.54555199e+00, -1.94706523e+00,
        1.14066000e+01,  3.54333966e+00,  4.81936376e+00, -1.22611088e-01,
        4.26151874e-01, -1.32871927e+00,  4.73979931e-01,  2.07631306e-01,
       -9.80828642e-02,  2.56158987e-01, -2.61268766e+00, -6.88787067e-01,
       -8.02639561e-01])


    print_table_for_mineral_constants(m, [(1, 1), (2, 2), (3, 3),
                                          (4, 4), (5, 5), (6, 6),
                                          (1, 2), (1, 3), (2, 3)])

    fig = plt.figure(figsize=(8, 4))
    ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

    temperatures = np.linspace(10., 1300., 101)
    alphas = np.empty((101,3))
    extensions = np.empty((101,3))
    vectors = np.empty((101,3))
    for i, T in enumerate(temperatures):
        m.set_state(1.e5, T)
        alphas[i] = np.diag(m.thermal_expansivity_tensor)*1.e5
        extensions[i] = ((np.diag(m.cell_vectors) / np.diag(m.cell_vectors_0)) - 1.)*1.e4
        vectors[i] = np.diag(m.cell_vectors)

    for i in range(3):
        ax[0].plot(temperatures, alphas[:,i])
        l = ax[1].plot(temperatures, extensions[:,i])
        ax[1].scatter(ol_1bar_lattice_data_Suzuki[:,0]+273.15,
                      ol_1bar_lattice_data_Suzuki[:,1+i],
                      color=l[0].get_color())
        #ax[1].scatter(ol_1bar_lattice_data[:,0]+273.15,
        #              ((ol_1bar_lattice_data[:,1+i]
        #               / ol_1bar_lattice_data[0,1+i])
        #              - 1.)*1.e4)

    ax[0].set_xlim(0.,)
    ax[0].set_ylim(0.,)
    plt.show()


    fig = plt.figure(figsize=(12, 12))
    ax = [fig.add_subplot(3, 3, i) for i in range(1, 10)]

    pressures = np.linspace(1.e7, 30.e9, 101)
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

    m.set_state(1.e5, 300.)
    print(m.thermal_expansivity_tensor)
    print(m.alpha)
    fo = burnman.minerals.SLB_2011.forsterite()
    fo = burnman.minerals.HP_2011_ds62.fo()
    fo.set_state(1.e5, 300.)
    print(fo.alpha)

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
# These parameters are from before adding in the thermal expansivity constraints
m = make_orthorhombic_mineral_from_parameters([ 1.00332023,  0.99306332,  0.26844583,  1.33251424,  1.03601286,
    0.45788811, -0.80718029,  1.3205917 ,  0.31268015,  0.12705779,
    0.81875369, -0.98232261,  5.09371197,  1.21339423,  1.02020599,
    0.6744214 , -0.82763648,  9.61277771,  0.18059782,  2.63862882,
    1.98732745, -1.26043057, 12.77807779, -1.74008056,  1.3066505 ,
    1.71020558, -1.83503072, 18.16393737, -3.14111273,  0.81273168,
    1.57182711, -1.3640314 , 13.2638437 ,  3.34240458, -0.86396854,
   -0.11468532,  0.47858299, -0.20309022,  0.62263923,  0.86583006,
   -0.11372523,  0.20516553, -3.30323097, -0.52180018, -0.92712402])
"""
