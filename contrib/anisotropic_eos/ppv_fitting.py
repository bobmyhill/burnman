# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""

ppv_fitting
-----------

"""
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import burnman_path  # adds the local burnman directory to the path
import burnman
from burnman.eos_fitting import fit_PTV_data

from anisotropicmineral import AnisotropicMineral

from tools import print_table_for_mineral_constants
from tools import plot_projected_elastic_properties

assert burnman_path  # silence pyflakes warning

formula = 'MgSiO3'
formula = burnman.processchemistry.dictionarize_formula(formula)
formula_mass = burnman.processchemistry.formula_mass(formula)

# Define the unit cell lengths and unit cell volume.
Z = 4. #XXXX
cell_lengths_angstrom = np.array([2.662, 8.91, 7.5])
cell_lengths_0_guess = cell_lengths_angstrom*np.cbrt(burnman.constants.Avogadro/Z/1.e30)

length_scale = 1./np.cbrt(burnman.constants.Avogadro/Z/1.e30)

Sakai_lattice_data = np.loadtxt('data/Sakai_et_al_2016_ppv_lattice_params.dat')
Stackhouse_elastic_data = np.loadtxt('data/Stackhouse_2005_ppv_isothermal.dat')
Stackhouse_lattice_data = np.loadtxt('data/Stackhouse_2005_ppv_lattice_params.dat')


i300 = Sakai_lattice_data.T[8] < 301
PGPa, PGPaerr = Sakai_lattice_data[:,:2].T
TK, TKerr = Sakai_lattice_data[:,8:10].T
a, aerr, b, berr, c, cerr = Sakai_lattice_data[:,10:16].T
V, Verr = Sakai_lattice_data[:,16:18].T


# Step 1: Fit volume data
ppv = burnman.minerals.SLB_2011.mg_post_perovskite()
data = np.array([PGPa*1.e9, TK, V*burnman.constants.Avogadro/Z/1.e30]).T
data_covariances = np.power(np.array([np.diag(d)
                                      for d in np.array([PGPaerr*1.e9, TKerr,
                                                         Verr*burnman.constants.Avogadro/Z/1.e30]).T]),
                            2.)


model = fit_PTV_data(ppv, ['V_0', 'K_0', 'Kprime_0', 'grueneisen_0'], data, data_covariances, verbose=False)


# Plot data


PGPa_Stackhouse, TK_Stackhouse, C11, C22, C33, C12, C13, C23, C44, C55, C66, K, G = Stackhouse_elastic_data.T
PPa_Stackhouse = PGPa_Stackhouse * 1.e9

zrs = np.zeros_like(C11)

C_T = np.array([[C11, C12, C13, zrs, zrs, zrs],
                [C12, C22, C23, zrs, zrs, zrs],
                [C13, C23, C33, zrs, zrs, zrs],
                [zrs, zrs, zrs, C44, zrs, zrs],
                [zrs, zrs, zrs, zrs, C55, zrs],
                [zrs, zrs, zrs, zrs, zrs, C66]])

C_T = np.moveaxis(C_T, -1, 0)

S = np.linalg.inv(C_T)
betas_Stackhouse = np.sum(S, axis=2)[:,:3]
betaV_Stackhouse = np.sum(betas_Stackhouse, axis=1)

pressures = np.linspace(1.e9, 300.e9, 101)
temperatures = 300. + 0.*pressures
#plt.plot(pressures/1.e9, ppv.evaluate(['K_T'], pressures, temperatures)[0]/1.e9)
#plt.scatter(PGPa_Stackhouse, 1./betaV_Stackhouse)

# C11, C22, C33, C12, C13, C23, C44, C55, C66
S_T_Stackhouse = np.array([S[:,0,0], S[:,1,1], S[:,2,2],
                           S[:,0,1], S[:,0,2], S[:,1,2],
                           S[:,3,3], S[:,4,4], S[:,5,5]]).T
C_T_Stackhouse = np.array([C_T[:,0,0], C_T[:,1,1], C_T[:,2,2],
                           C_T[:,0,1], C_T[:,0,2], C_T[:,1,2],
                           C_T[:,3,3], C_T[:,4,4], C_T[:,5,5]]).T

dchidf = (S_T_Stackhouse.T/betaV_Stackhouse).T

print(dchidf)
print(np.sum(dchidf[:,:6], axis=1) + np.sum(dchidf[:,3:6], axis=1))




oneoverST_Stackhouse = 1./S_T_Stackhouse


f_order = 3
Pth_order = 1
constants = np.zeros((6, 6, f_order+1, Pth_order+1))

ppv_cell_parameters = np.array([cell_lengths_0_guess[0],
                                cell_lengths_0_guess[1],
                                ppv.params['V_0'] / (cell_lengths_0_guess[0]*cell_lengths_0_guess[1]),
                                90, 90, 90])

constants[0, 0, 1, 0] = 0.31
constants[1, 1, 1, 0] = 0.39
constants[2, 2, 1, 0] = 0.3
constants[3, 3, 1, 0] = 1.
constants[4, 4, 1, 0] = 1.
constants[5, 5, 1, 0] = 1.
m = AnisotropicMineral(ppv, ppv_cell_parameters, constants)



fig = plt.figure(figsize=(12, 4))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

abc = np.empty((101, 3))

for i, P in enumerate(pressures):
    m.set_state(P, 300.)
    abc[i] = np.diag(m.cell_vectors)

abc_data = [a, b, c]
for i in range(3):
    ax[i].plot(pressures/1.e9, abc[:,i]*length_scale)
    ax[i].scatter(PGPa[i300], abc_data[i][i300])


plt.show()



# Step 2: Fit anisotropic data



def make_orthorhombic_mineral_from_parameters(x):
    f_order = 3
    Pth_order = 1
    constants = np.zeros((6, 6, f_order+1, Pth_order+1))

    ppv_cell_parameters = np.array([x[0]*cell_lengths_0_guess[0],
                                    x[1]*cell_lengths_0_guess[1],
                                    ppv.params['V_0'] / (x[0]*cell_lengths_0_guess[0]*x[1]*cell_lengths_0_guess[1]),
                                    90, 90, 90])

    # Each of the eight independent elastic tensor component get their turn.
    # We arbitrarily choose S[2,3] as the ninth component, which is determined by the others.
    i = 2
    for (p, q) in ((1, 1),
                   (2, 2),
                   (3, 3),
                   (4, 4),
                   (5, 5),
                   (6, 6),
                   (1, 2),
                   (1, 3)):
        for (m, n) in ((1, 0),
                       (2, 0)):
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

    assert i == 34 # 32 parameters, last one is pressure shift
    assert i+1 == len(x), f'Param length {len(x)}, should be {i+1}'
    # Fill the values for the dependent element c[2,3]
    constants[1,2,1,0] = (1. - np.sum(constants[:3,:3,1,0])) / 2.
    constants[1,2,2:,0] = - np.sum(constants[:3,:3,2:,0], axis=(0, 1)) / 2.
    constants[1,2,:,1:] = - np.sum(constants[:3,:3,:,1:], axis=(0, 1)) / 2.

    # And for c[3,2]
    constants[2,1,:,:] = constants[1,2,:,:]

    m = AnisotropicMineral(ppv, ppv_cell_parameters, constants)
    return m

run_fitting = False
sol = []
if run_fitting:

    def orthorhombic_misfit(x, imin):
        m = make_orthorhombic_mineral_from_parameters(x)

        chisqr = 0.
        #try:
        for d in Sakai_lattice_data:
            PGPa, _, _, _, _, _, _, _ = d[:8]
            TK, Terr = d[8:10]
            a, aerr, b, berr, c, cerr = d[10:16]

            PPa = PGPa * 1.e9
            m.set_state(PPa, TK)

            a_model, b_model, c_model = np.diag(m.cell_vectors)*length_scale

            chisqr += np.power((a_model - a)/aerr, 2.)
            chisqr += np.power((b_model - b)/berr, 2.)
            chisqr += np.power((c_model - c)/cerr, 2.)


        #K, G = Stackhouse_elastic_data[0,11:13]
        PPa_Stackhouse_shift = (PPa_Stackhouse + x[-1] * 1.e9)
        m.set_state(PPa_Stackhouse_shift[0], TK_Stackhouse[0])
        S = m.isothermal_compliance_tensor
        ST = np.array([S[0,0], S[1,1], S[2,2],
                       S[0,1], S[0,2], S[1,2],
                       S[3,3], S[4,4], S[5,5]])
        oneoverST0 = 1.e-9/ST
        Delta_oneoverST0 = oneoverST0 - oneoverST_Stackhouse[0]

        # Very weak no-change constraints
        chisqr += np.sum(np.power(Delta_oneoverST0/100., 2.))

        # Stronger difference constraints
        for i in range(1, len(oneoverST_Stackhouse)):
            m.set_state(PPa_Stackhouse_shift[i], TK_Stackhouse[i])

            S = m.isothermal_compliance_tensor
            ST = np.array([S[0,0], S[1,1], S[2,2],
                           S[0,1], S[0,2], S[1,2],
                           S[3,3], S[4,4], S[5,5]])
            oneoverST = 1.e-9/ST
            Delta_oneoverST = oneoverST - oneoverST_Stackhouse[i]

            chisqr += np.sum(np.power((Delta_oneoverST - Delta_oneoverST0)/10., 2.))

        """
        for d in Stackhouse_lattice_data:
            PGPa, TK, a, b, c, V, rho = d

            PPa = (PGPa + x[-1]) * 1.e9
            m.set_state(PPa, TK)

            a_model, b_model, c_model = np.diag(m.cell_vectors)*length_scale

            chisqr += np.power((a_model - a)/aerr, 2.)
            chisqr += np.power((b_model - b)/berr, 2.)
            chisqr += np.power((c_model - c)/cerr, 2.)
        """

        #if chisqr < 1500.:
        #    print(chisqr)
        #m.set_state(1.e5, 300)
        #print(np.diag(m.thermal_expansivity_tensor))

        if np.isnan(chisqr):
            print(d, "Noooo, there was a nan")
            chisqr = 1.e7

        #except Exception as e:
        #    print('There was an exception')
        #    print(e)
        #    chisqr = 1.e7
        imin[0][0] += 1
        if chisqr < imin[0][1]:
            imin[0][1] = chisqr
            print(imin[0])
            print(repr(x))
        return chisqr

    guesses = np.array([1., 1.,
                        0.609, 0., 0., 0.,
                        0.951, 0., 0., 0.,
                        0.662, 0., 0., 0.,
                        2.34, 0., 0., 0.,
                        2.48, 0., 0., 0.,
                        1.59, 0., 0., 0.,
                        -0.233, 0., 0., 0.,
                        -0.072, 0., 0., 0.,
                        0.])

    guesses = np.array([ 1.00042533e+00,  1.00194019e+00,  6.09129222e-01, -5.14208437e-02,
       -1.12344538e-02,  3.67313552e-02,  9.52602229e-01,  2.48623416e-01,
        8.66955296e-03,  3.27942549e-02,  6.58050855e-01, -3.99247615e-03,
        9.39673036e-03,  1.94344780e-02,  2.52286769e+00,  8.88482828e-02,
       -2.58275218e-01,  2.09452256e-01,  2.56757450e+00,  8.52278807e-01,
       -3.26952176e-01,  7.73687817e-01,  1.71734306e+00,  6.47034255e-01,
       -7.93215917e-02,  3.97100320e-01, -2.30849522e-01, -4.28062389e-03,
        6.03259368e-03, -3.15951972e-02, -7.52641312e-02,  1.45167497e-02,
        1.00845187e-03, -3.38227528e-02,  4.09452346e-01])

    """
    guesses = np.array([ 0.81172636,  0.84136231,  0.79462468,  0.80258777,  0.88711058,
        0.58820838,  2.18703617,  0.31562519,  0.13020317,  0.03525421,
        0.03593945,  0.70407015,  0.13043439,  0.08613014,  0.25200444,
        0.3894672 ,  0.26463096,  0.0267337 , -0.11455039,  2.15634471,
       -0.50319819,  0.32969081, -0.49071236,  2.14701372,  0.18119596,
        0.57519206,  0.0919673 ,  1.64670193,  0.30957439,  0.44313979,
       -0.07818676, -0.08666978, -0.03624815, -0.03367018, -0.14104753,
        0.09822089, -0.05170461, -0.01115777,  0.05015828,  0.54532194])

    guesses = np.array([ 8.32525622e-01,  8.14940794e-01,  7.84918198e-01,  7.65019837e-01,
        1.02810174e+00,  5.72898790e-01,  1.96514096e+00,  4.10442374e-01,
       -8.39037592e-02,  2.17812177e-02,  4.71414552e-02,  1.00235743e-01,
        5.30956488e-01, -2.29477070e-01,  1.56040132e-03,  2.65048135e-01,
        7.27067856e-02,  6.05058150e-01, -1.24508782e-01,  1.66995414e-02,
        2.90023312e-01,  3.12621398e-02,  2.04124644e+00, -4.33507164e-01,
       -8.97885478e-03, -3.55836500e-01, -2.46766852e-02,  2.15752721e+00,
       -5.22507778e-01,  4.02912263e-02, -4.11396031e-02,  7.91969846e-01,
        1.48575779e+00, -1.55482269e-02, -6.23894154e-02, -1.21602627e-01,
        3.92122211e-01,  1.22733366e-01,  2.76807291e-01, -1.12834835e-03,
       -4.36759190e-02, -1.61006351e-01, -3.70070012e-02,  3.12092920e-01,
        3.26014227e-03,  1.00027395e-01,  3.36677067e-01, -1.62225880e-01,
        9.29439226e-02,  7.78370241e-03, -4.13622081e-02, -1.34409308e-01,
       -2.14114874e-01])
    """
    i = 0
    min = 1.e10
    sol = minimize(orthorhombic_misfit, guesses, method='COBYLA', args=[[i, min]], options={'rhobeg': 0.2, 'maxiter': 100000})
    print(sol)

do_plotting = True
if do_plotting:
    if run_fitting:
        m = make_orthorhombic_mineral_from_parameters(sol.x)
    else:
        # Not final solution, but taken while improvement was slowing down.
        m = make_orthorhombic_mineral_from_parameters([ 1.00042533e+00,  1.00194019e+00,  6.09129222e-01, -5.14208437e-02,
       -1.12344538e-02,  3.67313552e-02,  9.52602229e-01,  2.48623416e-01,
        8.66955296e-03,  3.27942549e-02,  6.58050855e-01, -3.99247615e-03,
        9.39673036e-03,  1.94344780e-02,  2.52286769e+00,  8.88482828e-02,
       -2.58275218e-01,  2.09452256e-01,  2.56757450e+00,  8.52278807e-01,
       -3.26952176e-01,  7.73687817e-01,  1.71734306e+00,  6.47034255e-01,
       -7.93215917e-02,  3.97100320e-01, -2.30849522e-01, -4.28062389e-03,
        6.03259368e-03, -3.15951972e-02, -7.52641312e-02,  1.45167497e-02,
        1.00845187e-03, -3.38227528e-02,  4.09452346e-01])

    print('The following parameters were used for the volumetric part of '
          f'the isotropic model: $V_0$: {m.params["V_0"]*1.e6:.5f} cm$^3$/mol, '
          f'$K_0$: {m.params["K_0"]/1.e9:.5f} GPa, '
          f'$K\'_0$: {m.params["Kprime_0"]:.5f}, '
          f'$\Theta_0$: {m.params["Debye_0"]:.5f} K, '
          f'$\gamma_0$: {m.params["grueneisen_0"]:.5f}, '
          f'and $q_0$: {m.params["q_0"]:.5f}.')

    print_table_for_mineral_constants(m, [(1, 1), (2, 2), (3, 3),
                                          (4, 4), (5, 5), (6, 6),
                                          (1, 2), (1, 3), (2, 3)])
    """
    m.set_state(100.e9, 2000.)
    np.set_printoptions(precision=3)
    print(np.linalg.inv(m.isentropic_compliance_tensor))
    print(m.isentropic_stiffness_tensor)
    print(np.linalg.inv(m.isothermal_compliance_tensor))
    print(m.isothermal_stiffness_tensor)
    exit()
    """

    # Plot thermal expansion figure
    fig = plt.figure(figsize=(12, 4))
    ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

    pressures = np.linspace(80.e9, 300.e9, 101)
    vectors = np.empty((101,4))

    labels = ['a', 'b', 'c', 'V']

    for T in [300., 1000., 2000., 3000., 4000.]:
        for i, P in enumerate(pressures):
            m.set_state(P, T)

            vectors[i,:3] = np.diag(m.cell_vectors)*length_scale
            vectors[i,3] = m.V*np.power(length_scale, 3.)


        #i300 = Sakai_lattice_data.T[8] < 301
        a, aerr, b, berr, c, cerr = Sakai_lattice_data[:,10:16].T
        lattice_params = [a, b, c]
        lattice_param_errors = [aerr, berr, cerr]
        for i in range(3):
            ln = ax[i].plot(pressures/1.e9, vectors[:,i], label=f'{T} K',
                            color='black')

            if T < 400:
                ax[i].errorbar(Sakai_lattice_data[:,0], lattice_params[i],
                               yerr=lattice_param_errors[i],
                               linestyle='None', color=ln[0].get_color())
                pts = ax[i].scatter(Sakai_lattice_data[:,0], lattice_params[i],
                              c=Sakai_lattice_data.T[8])
                cbar = fig.colorbar(pts, ax=ax[i])
                cbar.set_label('Temperature (K)')

    for i in range(2):
        ax[i].set_xlabel('Pressure (GPa)')
        ax[i].legend()

    #ax[0].set_ylabel('Thermal expansivity (10$^{-5}$/K)')
    #ax[1].set_ylabel('Relative length change ($10^{4} (x/x_0 - 1)$)')

    fig.set_tight_layout(True)
    fig.savefig('ppv_lattice_parameters.pdf')
    plt.show()

    # Start plotting Cij figure
    fig = plt.figure(figsize=(12, 12))
    ax = [fig.add_subplot(3, 3, i) for i in range(1, 10)]

    temperatures = np.linspace(10., 4000., 101)
    SN = np.empty((len(temperatures), 6, 6))
    ST = np.empty((len(temperatures), 6, 6))
    CN = np.empty((len(temperatures), 6, 6))
    CT = np.empty((len(temperatures), 6, 6))

    i_pq = ((1, 1),
            (2, 2),
            (3, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (4, 4),
            (5, 5),
            (6, 6))

    pressures = [120.e9, 135.e9]
    for P in pressures:
        for i, T in enumerate(temperatures):
            m.set_state(P, T)
            SN[i] = m.isentropic_compliance_tensor
            ST[i] = m.isothermal_compliance_tensor
            CN[i] = m.isentropic_stiffness_tensor
            CT[i] = m.isothermal_stiffness_tensor

        """
        for i, (p, q) in enumerate(i_pq):
            ln = ax[i].plot(temperatures, 1.e-9/SN[:, p-1, q-1], label=f'{P/1.e9} GPa')
            ax[i].plot(temperatures, 1.e-9/ST[:, p-1, q-1],
                       linestyle='--', label=f'{P/1.e9} GPa', color=ln[0].get_color())
            # Stackhouse data format
            # PGPa, TK, C11, C22, C33, C12, C13, C23, C44, C55, C66, K, G = d

            ax[i].scatter(TK_Stackhouse, oneoverST_Stackhouse[:,i], color=ln[0].get_color())
        """

        for i, (p, q) in enumerate(i_pq):
            ln = ax[i].plot(temperatures, CN[:, p-1, q-1]/1.e9,
                            linestyle='--', label=f'$C_N$ {P/1.e9} GPa')
            ax[i].plot(temperatures, CT[:, p-1, q-1]/1.e9,
                       label=f'$C_T$ {P/1.e9} GPa', color=ln[0].get_color())
            # Stackhouse data format
            # PGPa, TK, C11, C22, C33, C12, C13, C23, C44, C55, C66, K, G = d
            mask = np.abs(PGPa_Stackhouse - P/1.e9) < 4.
            ax[i].scatter(TK_Stackhouse[mask], C_T_Stackhouse[mask,i], color=ln[0].get_color())

    for i, (p, q) in enumerate(i_pq):
        ax[i].set_xlabel('Temperature (K)')
        ax[i].set_ylabel(f'$1/S_{{X {p}{q}}}$ (GPa)')
        ax[i].legend()

    fig.set_tight_layout(True)
    fig.savefig('ppv_CNijs.pdf')
    plt.show()
    exit()

    fig = plt.figure(figsize=(12, 7))
    ax = [fig.add_subplot(2, 3, i, projection='polar') for i in range(1, 7)]

    P = 3.e9
    T = 1600.
    m.set_state(P, T)
    plot_types = ['vp', 'vs1', 'vp/vs1',
                  's anisotropy', 'linear compressibility', 'youngs modulus']

    contour_sets, ticks, lines = plot_projected_elastic_properties(m,
                                                                   plot_types,
                                                                   ax)
    for i in range(len(contour_sets)):
        cbar = fig.colorbar(contour_sets[i], ax=ax[i],
                            ticks=ticks[i], pad = 0.1)
        cbar.add_lines(lines[i])

    fig.set_tight_layout(True)
    fig.savefig(f'olivine_seismic_properties_{P/1.e9:.2f}_GPa_{int(T)}_K.pdf')
    plt.show()
