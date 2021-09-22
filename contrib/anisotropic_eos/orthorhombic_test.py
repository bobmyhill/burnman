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
from tools import plot_projected_elastic_properties

from test_consistency import check_anisotropic_eos_consistency

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
                       (1, 1),
                       (2, 1),
                       (3, 1)):
            constants[p-1, q-1, m, n] = x[i]*1.e-11
            constants[q-1, p-1, m, n] = x[i]*1.e-11
            i += 1

        for (m, n) in ((0, 2),):
            constants[p-1, q-1, m, n] = x[i]*1.e-22
            constants[q-1, p-1, m, n] = x[i]*1.e-22
            i += 1

    assert i == 69 # 40 parameters

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


m = make_orthorhombic_mineral_from_parameters([ 1.00261177e+00,  9.91759509e-01,  1.00180767e+00,  1.12629568e+00,
3.13913957e-01,  4.43835171e-01, -9.38192626e-01,  8.57450038e-01,
2.63521201e-01,  3.10992538e-01, -5.84207311e+00,  1.22205974e+01,
5.11362234e-01,  7.76039201e-01, -1.00640533e+00,  5.66780847e+00,
5.12401782e-01,  1.59529634e+00,  1.23345902e+01, -7.60264507e+00,
3.06123818e-01,  6.62862573e-01, -6.29539285e-01,  9.07101981e+00,
1.70501045e+00,  1.90725482e+00,  6.48576298e+00,  2.99733967e+00,
3.62644594e-01,  1.96838589e+00, -4.97224163e-01,  2.08768703e+01,
-2.66242709e+00,  2.32579910e+00, -6.26342959e+00,  1.10758805e+01,
-4.99496737e+00,  1.61144010e+00, -1.85034515e+00,  2.32110973e+01,
-3.15692901e+00,  2.65209318e+00,  4.39232410e-01,  4.71069329e+00,
-6.24379333e+00,  1.55360338e+00, -1.42688476e+00,  1.26449796e+01,
-3.69943280e-01,  5.71780041e+00,  6.49141249e+00, -3.81945412e+00,
-1.25012075e+00, -1.20402033e-01,  4.38934297e-01, -1.17987749e+00,
4.61289178e-01, -2.21403680e-01,  7.81563940e+00,  8.17777878e+00,
-1.34030384e-02, -1.01671929e-01,  2.70232982e-01, -2.68143106e+00,
-6.93075277e-01, -4.04634113e-01, -3.49178491e+00,  1.09213501e+01,
4.91098948e-02])
np.set_printoptions(precision=2)
print(np.array_repr(m.c[:,:,1,0]))
exit()

check_anisotropic_eos_consistency(m, verbose=True)
