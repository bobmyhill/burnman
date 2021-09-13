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

import time
import emcee
import os
from multiprocessing import cpu_count
from multiprocessing import Pool

#os.environ["OMP_NUM_THREADS"] = "1"  # important to kill numpy multiprocessing
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['MKL_NUM_THREADS'] = '1'

assert burnman_path  # silence pyflakes warning

formula = 'Mg1.8Fe0.2SiO4'
formula = burnman.processchemistry.dictionarize_formula(formula)
formula_mass = burnman.processchemistry.formula_mass(formula)

# Define the unit cell lengths and unit cell volume.
# These are taken from Abramson et al., 1997
Z = 4.
cell_lengths_angstrom = np.array([4.7646, 10.2296, 5.9942])
cell_lengths = cell_lengths_angstrom*np.cbrt(burnman.constants.Avogadro/Z/1.e30)
ol_cell_parameters = np.array([cell_lengths[0],
                              cell_lengths[1],
                              cell_lengths[2],
                              90, 90, 90])
V_0 = np.prod(cell_lengths)

ol_data = np.loadtxt('data/Mao_et_al_2015_ol.dat')
ol_1bar_lattice_data = np.loadtxt('data/Singh_Simmons_1976_1bar_lattice_parameters_ol.dat')
ol_1bar_lattice_data_Suzuki = np.loadtxt('data/Suzuki_1975_ol_Kenya_expansion.dat')

def make_orthorhombic_mineral_from_parameters(x):
    f_order = 3
    Pth_order = 2
    constants = np.zeros((6, 6, f_order+1, Pth_order+1))

    san_carlos_params = {'name': 'San Carlos olivine',
                         'formula': formula,
                         'equation_of_state': 'slb3',
                         'F_0': 0.0,
                         'V_0': V_0,
                         'K_0': 1.263e+11, # Abramson et al. 1997
                         'Kprime_0': 4.28, # Abramson et al. 1997
                         'Debye_0': 809.1703, # Fo in SLB2011
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
                       (1, 1),
                       (2, 1),
                       (3, 1)):
            constants[p-1, q-1, m, n] = x[i]*1.e-11
            constants[q-1, p-1, m, n] = x[i]*1.e-11
            i += 1

        #for (m, n) in ((0, 2),):
        #    constants[p-1, q-1, m, n] = x[i]*1.e-22
        #    constants[q-1, p-1, m, n] = x[i]*1.e-22
        #    i += 1

    assert i == 61 # 53 parameters

    # Fill the values for the dependent element c[2,3]
    constants[1,2,1,0] = (1. - np.sum(constants[:3,:3,1,0])) / 2.
    constants[1,2,2:,0] = - np.sum(constants[:3,:3,2:,0], axis=(0, 1)) / 2.
    constants[1,2,:,1:] = - np.sum(constants[:3,:3,:,1:], axis=(0, 1)) / 2.

    # And for c[3,2]
    constants[2,1,:,:] = constants[1,2,:,:]

    m = AnisotropicMineral(ol, ol_cell_parameters, constants)
    return m

run_fitting = True
if run_fitting:

    def orthorhombic_misfit(x):
        m = make_orthorhombic_mineral_from_parameters(x)

        chisqr = 0.

        try:
            #print(repr(x))

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

                Y = ((np.diag(m.cell_vectors) / cell_lengths) - 1.)*1.e4
                Y_expt = d[1:4]
                Y_err = 0.05*Y_expt + 1.
                for i in range(3):
                    chisqr += np.power((Y_expt[i] - Y[i])/Y_err[i], 2.)
            """

            #if chisqr < 1500.:
            #    print(chisqr)
            #m.set_state(1.e5, 300)
            #print(np.diag(m.thermal_expansivity_tensor))

            if np.isnan(chisqr):
                print(d, "Noooo, there was a nan")
                chisqr = 1.e7

        except:
            chisqr = 1.e7

        return chisqr

    def log_prob(x):
        return -0.5*orthorhombic_misfit(x)

    #guesses = [ 1.00263119e+00,  1.00503329e+00,  7.99519521e-01,  1.40122330e+00, 1.10634879e+00,
    #guesses = [ 1, 1, 1, 1, 1,
    guesses = np.array([ 1.02165865e+00,  9.41850472e-01,  3.70980059e-01,  1.14943551e+00,
    1.07650340e+00,  4.52702745e-01, -8.55675236e-01,  2.32148936e-01,
   -8.68166882e-03,  2.94685367e-01, -4.00576983e-01,  8.24046295e-01,
    7.99980945e-01, -1.06760121e+00,  1.20954100e+00,  1.17467078e-01,
   -1.29005047e-01,  2.77543310e+00, -3.12030005e-01,  6.58579965e-01,
   -1.07079030e+00,  3.91599083e+00,  1.55351554e-02,  5.30910650e-01,
    6.00282732e-01, -5.50954728e-01,  2.04179943e+00, -5.28829010e-01,
    1.18260352e+01,  6.12673002e-02, -3.85766462e-01, -1.92956253e+00,
    5.06235209e-01,  1.73819755e+00, -1.77217717e+00,  1.41788995e+01,
   -5.92204213e-01, -1.74241030e+00, -1.81925460e+00,  3.43491447e+00,
    1.60222254e+00, -1.09958035e+00,  1.07352203e+01, -3.95635116e-01,
    4.02201452e+00, -8.97434197e-01,  2.81003943e+00, -1.38769436e-01,
    3.77915416e-01,  2.48398232e-01, -4.96315825e-02, -1.18407630e-01,
    4.61637500e-01,  1.00388138e+00, -9.54628178e-02,  3.18543470e-01,
   -2.28433767e+00,  1.47642448e-01, -4.03397688e-01, -6.78582327e-01,
    7.89748310e-01])


    """
    PARAMETERS FOR RUNNING SIMULATION!!
    """
    new_inversion = True
    nsteps = 100000

    ncpu = cpu_count()
    ncpus_for_mp = 4 # ncpu - 30

    print(f"{ncpu} CPUs total on machine, running with {ncpus_for_mp} CPUs")
    print(f"Running for {nsteps} steps.")

    ndim, nwalkers = len(guesses), len(guesses)*4

    filename = "samples.h5"
    print('Loading backend')
    backend = emcee.backends.HDFBackend(filename)

    eps = 0.1
    np.random.seed(42)
    deltas = eps*np.random.rand(nwalkers, ndim)
    initial = guesses + deltas

    with Pool(processes=ncpus_for_mp) as pool:
        # Only reset for new files
        if new_inversion:
            print("Starting new inversion")
            backend.reset(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
                                            pool=pool, backend=backend)
        else:
            print("Continuing from previous state")
            print("Initial size: {0}".format(backend.iteration))
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
                                            pool=pool, backend=backend)
            initial = sampler.get_chain()[-1]

        start = time.time()
        sampler.run_mcmc(initial, nsteps, progress=True)
        end = time.time()
        multi_data_global_time = end - start
        print(f"Multiprocessing took {multi_data_global_time:.1f} seconds")


do_plotting = False
if do_plotting:

    # Not final solution, but taken while improvement was slowing down.
    #m = make_orthorhombic_mineral_from_parameters([1.00263119e+00,  1.00503329e+00,  7.99519521e-01,  1.40122330e+00,  1.10634879e+00,
    m = make_orthorhombic_mineral_from_parameters([ 1.02165865e+00,  9.41850472e-01,  3.70980059e-01,  1.14943551e+00,
        1.07650340e+00,  4.52702745e-01, -8.55675236e-01,  2.32148936e-01,
       -8.68166882e-03,  2.94685367e-01, -4.00576983e-01,  8.24046295e-01,
        7.99980945e-01, -1.06760121e+00,  1.20954100e+00,  1.17467078e-01,
       -1.29005047e-01,  2.77543310e+00, -3.12030005e-01,  6.58579965e-01,
       -1.07079030e+00,  3.91599083e+00,  1.55351554e-02,  5.30910650e-01,
        6.00282732e-01, -5.50954728e-01,  2.04179943e+00, -5.28829010e-01,
        1.18260352e+01,  6.12673002e-02, -3.85766462e-01, -1.92956253e+00,
        5.06235209e-01,  1.73819755e+00, -1.77217717e+00,  1.41788995e+01,
       -5.92204213e-01, -1.74241030e+00, -1.81925460e+00,  3.43491447e+00,
        1.60222254e+00, -1.09958035e+00,  1.07352203e+01, -3.95635116e-01,
        4.02201452e+00, -8.97434197e-01,  2.81003943e+00, -1.38769436e-01,
        3.77915416e-01,  2.48398232e-01, -4.96315825e-02, -1.18407630e-01,
        4.61637500e-01,  1.00388138e+00, -9.54628178e-02,  3.18543470e-01,
       -2.28433767e+00,  1.47642448e-01, -4.03397688e-01, -6.78582327e-01,
        7.89748310e-01])


    fig = plt.figure(figsize=(8, 4))
    ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

    temperatures = np.linspace(10., 1300., 101)
    alphas = np.empty((101,3))
    extensions = np.empty((101,3))
    vectors = np.empty((101,3))
    for i, T in enumerate(temperatures):
        m.set_state(1.e5, T)
        alphas[i] = np.diag(m.thermal_expansivity_tensor)*1.e5
        extensions[i] = ((np.diag(m.cell_vectors) / cell_lengths) - 1.)*1.e4
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
