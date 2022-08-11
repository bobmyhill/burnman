from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import burnman

from burnman import AnisotropicMineral

from burnman.tools.plot import plot_projected_elastic_properties


run_fitting = False
plot_SLB = False

formula = 'Mg1.8Fe0.2SiO4'
formula = burnman.tools.chemistry.dictionarize_formula(formula)
formula_mass = burnman.tools.chemistry.formula_mass(formula)

# Define the unit cell lengths and unit cell volume.
# These are taken from Abramson et al., 1997
Z = 4.
cell_lengths_angstrom = np.array([4.7646, 10.2296, 5.9942])
cell_lengths_0_guess = cell_lengths_angstrom * \
    np.cbrt(burnman.constants.Avogadro/Z/1.e30)
V_0_guess = np.prod(cell_lengths_0_guess)


fo = burnman.minerals.SLB_2011.forsterite()
fa = burnman.minerals.SLB_2011.fayalite()



def psi_func(f, Pth, params):
    dPsidf = (params['a'] + params['b_1']*params['c_1']
              * np.exp(params['c_1']*f) + params['b_2']
              * params['c_2']*np.exp(params['c_2']*f))
    Psi = (0. + params['a']*f
           + params['b_1'] * np.exp(params['c_1']*f)
           + params['b_2']*np.exp(params['c_2']*f)
           + params['d'] * Pth/1.e9)
    dPsidPth = params['d']/1.e9
    return (Psi, dPsidf, dPsidPth)


def make_orthorhombic_mineral_from_parameters(x):
    # First, make the scalar model
    san_carlos_params = {'name': 'San Carlos olivine',
                         'formula': formula,
                         'equation_of_state': 'slb3',
                         'F_0': 0.0,
                         'V_0': V_0_guess,  # we overwrite this in a second
                         'K_0': 1.263e+11,  # Abramson et al. 1997
                         'Kprime_0': 4.28,  # Abramson et al. 1997
                         'Debye_0': 760.,  # Robie, forsterite
                         'grueneisen_0': 0.99282,  # Fo in SLB2011
                         'q_0': 2.10672,  # Fo in SLB2011
                         'G_0': 81.6e9,
                         'Gprime_0': 1.46257,
                         'eta_s_0': 2.29972,
                         'n': 7.,
                         'molar_mass': formula_mass}

    R = burnman.constants.gas_constant
    san_carlos_property_modifiers = [['linear', {'delta_E': 0.0,
                                                 'delta_S': 26.76*0.1 - 2.*R*(0.1*np.log(0.1) + 0.9*np.log(0.9)),
                                                 'delta_V': 0.0}]]

    ol = burnman.Mineral(params=san_carlos_params,
                         property_modifiers=san_carlos_property_modifiers)

    # Overwrite some properties
    y = x

    ol.params['V_0'] = y[0]*V_0_guess  # Abramson et al. 1997
    ol.params['K_0'] = y[1]*1.263e+11  # Abramson et al. 1997
    ol.params['Kprime_0'] = y[2]*4.28  # Abramson et al. 1997
    ol.params['grueneisen_0'] = y[3]*0.99282  # Fo in SLB2011
    ol.params['q_0'] = y[4]*2.10672  # Fo in SLB2011
    # ol.params['Debye_0'] = x[5]*809.1703 # Fo in SLB2011 strong tendency to 0

    # Next, each of the eight independent elastic tensor component get
    # their turn.
    # We arbitrarily choose S[2,3] as the ninth component,
    # which is determined by the others.
    i = 5
    anisotropic_parameters = {'a': np.zeros((6, 6)),
                              'b_1': np.zeros((6, 6)),
                              'c_1': np.ones((6, 6)),
                              'd': np.zeros((6, 6)),
                              'b_2': np.zeros((6, 6)),
                              'c_2': np.ones((6, 6))}

    for (p, q) in ((1, 1),
                   (2, 2),
                   (3, 3),
                   (4, 4),
                   (5, 5),
                   (6, 6),
                   (1, 2),
                   (1, 3)):
        anisotropic_parameters['a'][p-1, q-1] = x[i]
        anisotropic_parameters['a'][q-1, p-1] = x[i]
        i = i + 1
        anisotropic_parameters['b_1'][p-1, q-1] = x[i]
        anisotropic_parameters['b_1'][q-1, p-1] = x[i]
        i = i + 1
        anisotropic_parameters['d'][p-1, q-1] = x[i]
        anisotropic_parameters['d'][q-1, p-1] = x[i]
        i = i + 1

    anisotropic_parameters['c_1'][:3, :3] = x[i]
    i = i + 1
    for j in range(3):
        anisotropic_parameters['c_1'][3+j, 3+j] = x[i]
        i = i + 1

    anisotropic_parameters['b_2'][3, 3] = x[i]
    i = i + 1
    anisotropic_parameters['b_2'][4, 4] = x[i]
    i = i + 1
    anisotropic_parameters['b_2'][5, 5] = x[i]
    i = i + 1
    anisotropic_parameters['c_2'][3, 3] = x[i]
    i = i + 1
    anisotropic_parameters['c_2'][4, 4] = x[i]
    i = i + 1
    anisotropic_parameters['c_2'][5, 5] = x[i]
    i = i + 1

    assert len(x) == i

    # Fill the values for the dependent element c[2,3]
    anisotropic_parameters['a'][1, 2] = (
        1. - np.sum(anisotropic_parameters['a'][:3, :3]))/2.
    anisotropic_parameters['b_1'][1, 2] = (
        0. - np.sum(anisotropic_parameters['b_1'][:3, :3]))/2.
    anisotropic_parameters['d'][1, 2] = (
        0. - np.sum(anisotropic_parameters['d'][:3, :3]))/2.

    anisotropic_parameters['a'][2, 1] = anisotropic_parameters['a'][1, 2]
    anisotropic_parameters['b_1'][2, 1] = anisotropic_parameters['b_1'][1, 2]
    anisotropic_parameters['d'][2, 1] = anisotropic_parameters['d'][1, 2]

    cell_lengths = cell_lengths_0_guess*np.cbrt(ol.params['V_0']/V_0_guess)
    ol_cell_parameters = np.array([cell_lengths[0],
                                   cell_lengths[1],
                                   cell_lengths[2],
                                   90, 90, 90])

    m = AnisotropicMineral(ol, ol_cell_parameters,
                           anisotropic_parameters, psi_func, orthotropic=True)
    return m

def forsterite():
    x = np.array([1.00263718e+00,  9.92582060e-01,  9.94959405e-01,  1.11986468e+00,
                    3.26968155e-01,  1.36482754e+00, -9.12627751e-01,  1.42227141e-02,
                    2.38325937e+00, -1.60675570e+00,  1.97961677e-02,  2.40863946e+00,
                    -1.76297463e+00,  1.58628282e-02,  2.77600544e+01, -8.63488144e+01,
                    -2.01661072e-01,  1.16595606e+01, -9.09004579e+00, -1.89410920e-01,
                    2.35013807e+01, -5.64225909e+01, -4.62157096e-01, -5.14209400e-01,
                    3.78113506e-01, -9.03837805e-03, -6.29669950e-01,  5.34427058e-01,
                    -4.77601300e-03,  1.00090172e+00,  3.13842027e-01,  1.48687484e+00,
                    4.16041813e-01,  2.19213171e-01,  6.40447373e-01,  2.53813715e-01,
                    6.08105337e+00,  5.47261046e+00,  6.12582927e+00])
    m = make_orthorhombic_mineral_from_parameters(x)
    return m
