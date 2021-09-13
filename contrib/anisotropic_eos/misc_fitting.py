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
from scipy.optimize import minimize

import burnman_path  # adds the local burnman directory to the path

import burnman
from burnman.minerals.SLB_2011 import forsterite, periclase

assert burnman_path  # silence pyflakes warning



from anisotropicmineral import AnisotropicMineral



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
