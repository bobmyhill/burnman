# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
cubic_fitting
-------------

This script creates an AnisotropicMineral object corresponding to
periclase (a cubic mineral). If run_fitting is set to True, the script uses
experimental data to find the optimal anisotropic parameters.
If set to False, it uses pre-optimized parameters.
The data is used only to optimize the anisotropic parameters;
the isotropic equation of state is taken from
Stixrude and Lithgow-Bertelloni (2011).

The script ends by making three plots; one with elastic moduli
at high pressure, one with the corresponding shear moduli,
and one with the elastic moduli at 1 bar.
"""

from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from tools import print_table_for_mineral_constants

import burnman
from burnman import AnisotropicMineral
from lmfit import Model

def fn_quartic(x, a, b, c, d, e):
    return a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4

per = burnman.minerals.SLB_2011.periclase()

run_fitting = True

V0 = 1.131744e-05

per_data = np.loadtxt('data/Karki_2000_periclase_CSijs.dat')



def make_voigt_matrix_inverse(Cs):
    T, P, C11, C12, C44, KT, V, betaTminusbetaS = Cs
    CS = np.zeros((6, 6))
    CS[:3, :3] = C12
    for i in range(3):
        CS[i, i] = C11
    for i in range(3, 6):
        CS[i, i] = C44
    # betaT = 1./KT

    SS = np.linalg.inv(CS)

    ST = np.linalg.inv(CS)
    ST[:3, :3] += betaTminusbetaS/9.

    betaT = np.sum(ST[:3, :3])

    print(f'{np.sum(ST[:3,:3]):.2e}, {betaT:.2e}')
    dpsidf = ST / betaT
    f = np.log(V / V0)
    f2 = 0.5*(np.power(V / V0, -2./3.) - 1.)
    print(T, V, f)
    return np.array([T, P, f, KT, dpsidf[0, 0], dpsidf[0, 1], dpsidf[3, 3]])


dpsidf_data = np.empty((per_data.shape[0], per_data.shape[1]-1))
for i in range(len(per_data)):
    dpsidf_data[i] = make_voigt_matrix_inverse(per_data[i])

fig = plt.figure(figsize=(12, 12))
ax = [fig.add_subplot(3, 3, i) for i in range(1, 10)]

pmodel = Model(fn_quartic)

dpsidf0 = dpsidf_data[0, 4:]


n_d = 31
idx = 0
P300_spline = interp1d(dpsidf_data[idx*n_d:(idx+1)*n_d, 2], dpsidf_data[idx*n_d:(idx+1)*n_d, 1], fill_value="extrapolate")



for idx in range(4):  # only plot 300 K data
    T = dpsidf_data[idx*n_d, 0]
    P = dpsidf_data[idx*n_d:(idx+1)*n_d, 1]
    f = dpsidf_data[idx*n_d:(idx+1)*n_d, 2]  # /2.6
    
    P300 = P300_spline(f)
    Pth = P - P300
    
    KT = dpsidf_data[idx*n_d:(idx+1)*n_d, 3]
    dpsidf = dpsidf_data[idx*n_d:(idx+1)*n_d, 4:].T
    psi = cumulative_trapezoid(dpsidf, f, initial=0)
    print(f)
    for j in range(3):
        print(dpsidf[j, 0])

        V = np.exp(f)
        gradpsi = np.gradient(psi[j], f, edge_order=2)
        splgrad = interp1d(f, gradpsi)

        spl = interp1d(f, psi[j] - dpsidf0[j]*f)
        v0 = spl(0.)
        grad0 = splgrad(0.)
        y = (psi[j]) #  - grad0*f - v0)
        #y = (psi[j] - v0)
        sply = interp1d(f, y)

        spl = interp1d(f, dpsidf[j])
        v0 = spl(0.)

        grady = np.gradient(y, f, edge_order=2)

        ln, = ax[j].plot(V, dpsidf[j], label=f'{T} K')
        
        ln, = ax[j+3].plot(f, dpsidf[j], label=f'{T} K')
    ln, = ax[j+6].plot(f, Pth, label=f'{T} K')


labels = ['11', '12', '44']
for i in range(3):
    ax[i].set_xlim(0., )
    ax[i].legend()
    ax[i].set_xlabel('$V$')
    ax[i].set_ylabel('$S/beta$')
    
    ax[i+3].legend()
    ax[i+3].set_xlabel('$f$')
    ax[i+3].set_ylabel('$S/beta$')

ax[0].set_ylim(0., )
ax[1].set_ylim(-0.4, 0.)

fig.set_tight_layout(True)
plt.show()
