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
from scipy.optimize import curve_fit

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from tools import print_table_for_mineral_constants

import burnman
from burnman import AnisotropicMineral
from lmfit import Model
import pyeq3
from pyeq3.Models_2D.UserDefinedFunction import UserDefinedFunction as UserDefined2DFunction
from pyeq3.Models_3D.UserDefinedFunction import UserDefinedFunction as UserDefined3DFunction
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


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

    # print(f'{np.sum(ST[:3,:3]):.2e}, {betaT:.2e}')
    dpsidf = ST / betaT
    f = np.log(V / V0)
    f2 = 0.5*(np.power(V / V0, -2./3.) - 1.)
    # print(T, V, f)
    return np.array([T, P, f, 1./betaT, dpsidf[0, 0], dpsidf[0, 1], dpsidf[3, 3]])


dpsidf_data = np.empty((per_data.shape[0], per_data.shape[1]-1))
for i in range(len(per_data)):
    dpsidf_data[i] = make_voigt_matrix_inverse(per_data[i])

fig = plt.figure(figsize=(6, 6))
ax = [fig.add_subplot(3, 3, i) for i in range(1, 10)]

pmodel = Model(fn_quartic)

dpsidf0 = dpsidf_data[0, 4:]


n_d = 31
idx = 0
P300_spline = interp1d(dpsidf_data[idx*n_d:(idx+1)*n_d, 2],
                       dpsidf_data[idx*n_d:(idx+1)*n_d, 1], fill_value="extrapolate")

data = [[], [], [], []]

for idx in range(4):  # only plot 300 K data
    T = dpsidf_data[idx*n_d, 0]
    P = dpsidf_data[idx*n_d:(idx+1)*n_d, 1]
    f = dpsidf_data[idx*n_d:(idx+1)*n_d, 2]  # /2.6

    P300 = P300_spline(f)
    Pth = P - P300

    KT = dpsidf_data[idx*n_d:(idx+1)*n_d, 3]
    dpsidf = dpsidf_data[idx*n_d:(idx+1)*n_d, 4:].T
    psi = cumulative_trapezoid(dpsidf, f, initial=0)

    data[0].extend(f)
    data[1].extend(Pth/1.e9)
    data[2].extend(dpsidf[0])
    data[3].extend(dpsidf[2])

    # print(f)
    for j in range(3):
        # print(dpsidf[j, 0])

        V = np.exp(f)
        gradpsi = np.gradient(psi[j], f, edge_order=2)
        splgrad = interp1d(f, gradpsi)

        spl = interp1d(f, psi[j] - dpsidf0[j]*f)
        v0 = spl(0.)
        grad0 = splgrad(0.)
        y = (psi[j])  # - grad0*f - v0)
        # y = (psi[j] - v0)
        sply = interp1d(f, y)

        spl = interp1d(f, dpsidf[j])
        v0 = spl(0.)

        grady = np.gradient(y, f, edge_order=2)

        ln, = ax[j].plot(V, dpsidf[j], label=f'{T} K')

        if j == 2 and idx == 3:
            for i in range(len(V)):
                print(f[i], dpsidf[j][i])

        ln, = ax[j+3].plot(f, dpsidf[j], label=f'{T} K')
        
    ln, = ax[6].plot(V, -dpsidf[1]/dpsidf[0], label=f'{T} K') 
    ln, = ax[7].plot(V, KT, label=f'{T} K')    
    ln, = ax[7].plot(V, KT/dpsidf[2], label=f'{T} K', linestyle=':')    
    ln, = ax[8].plot(f, Pth, label=f'{T} K')


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


data = np.array(data).T


def solved_equation(functionString, data, a_fix, b_min):

    equation = UserDefined3DFunction(inUserFunctionString=functionString)
    pyeq3.dataConvertorService().ProcessData(data, equation, False)

    equation.lowerCoefficientBounds = [None, b_min, b_min, b_min, b_min,
                                       b_min, b_min, b_min, b_min]
    equation.upperCoefficientBounds = [None, None, None,
                                       None, None, None,
                                       None, None, None]
    equation.fixedCoefficients = [a_fix, None, None, None, None,
                                  None, None, None, None]
    equation.Solve()
    popt = equation.solvedCoefficients
    if np.abs(popt[2] - popt[4]) < 1.e-2 or np.abs(popt[7] - popt[8]) < 1.e-2:
        print('Recalculating as exponents equal')
        equation.lowerCoefficientBounds = [None, b_min, popt[2]+0.01, b_min,
                                           b_min, b_min, b_min, popt[7]+0.01,
                                           b_min]
        equation.upperCoefficientBounds = [None, None,
                                           None, None,
                                           popt[2]-0.01, None,
                                           None, None, popt[7]-0.01]

        equation.Solve()
    return equation


# data = np.array([[d[0], d[2]] for d in data if (d[1] > 10 and d[1] < 17)])
# print(data)

if False:
    functionStringX = '(b)*exp((c)*X) + (g)*exp((h)*X)'
    lowers = [-1., 1., 10., 17.]
    uppers = [1., 10., 17., 25.]

    for i in range(4):

        equation = UserDefined2DFunction(inUserFunctionString=functionStringX)
        pyeq3.dataConvertorService().ProcessData(
            data[:, [0, 3]][np.all([data[:, 1] > lowers[i],
                                    data[:, 1] < uppers[i]],
                                   axis=0)],
            equation, False)
        equation.Solve()

        print(equation)

        fig = plt.figure(figsize=(8, 6))
        axes = fig.add_subplot(1, 1, 1)
        pyeq3.Graphics.Graphics2D.ModelScatterConfidenceGraph(equation, axes)
        plt.show()


cmap = plt.get_cmap('viridis')
norm = plt.Normalize(np.min(data[:, 1]), np.max(data[:, 1]))

fig = plt.figure(figsize=(8, 4))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

for i in range(2):
    functionStringXY = 'a + (b + j*Y)*exp((c+l*Y)*X) + (g+k*Y)*exp((h+m*Y)*X)'
    functionStringXY = ('a + b*exp(c*X) + g*exp(h*X) + '
                        'Y*(j*exp(l*X) + k*exp(m*X))')
    if i == 0:
        equation = solved_equation(
            functionStringXY, data[:, [0, 1, 2]], 1./3., 0.)
    else:
        equation = solved_equation(
            functionStringXY, data[:, [0, 1, 3]], 0., -10000.)
        """
        equation = UserDefined3DFunction(inUserFunctionString=functionStringXY)
        equation.fixedCoefficients = [0., None, None, None, None,
                                      None, None, None, None]
        pyeq3.dataConvertorService().ProcessData(
            data[:, [0, 1, 3]], equation, False)
        equation.Solve()
        """

    print(equation)

    pts = ax[i].scatter(np.exp(data[:, 0]), data[:, i+2], c=data[:, 1])

    VoverV0 = np.linspace(0.65, 1.15, 1001)

    for Pth in [0., 5, 10, 15, 20]:
        f = np.log(VoverV0)
        Pths = f*0. + Pth
        y = equation.CalculateModelPredictionsFromNewData(np.array([f, Pths]))
        ax[i].plot(VoverV0, y, color=cmap(norm(Pth)))

    minx = np.exp(np.min(data[:, 0]))
    #minx = np.min([minx, VoverV0[0]])
    maxx = np.exp(np.max(data[:, 0]))
    rangex = maxx - minx
    miny = np.min(data[:, i+2])
    #miny = np.min([miny, np.min(y)])
    maxy = np.max(data[:, i+2])
    rangey = maxy - miny
    ax[i].set_xlim(minx - 0.05*rangex, maxx + 0.05*rangex)
    ax[i].set_ylim(miny - 0.05*rangey, maxy + 0.05*rangey)

    ax[i].set_xlabel('$V / V_0$')

    if i == 0:
        ax[i].set_ylabel('$S_{T11} / \\beta_T$')
    else:
        ax[i].set_ylabel('$S_{T44} / \\beta_T$')


# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%",
                          pad=0.05)

cbar = plt.colorbar(pts, cax=cax)
cbar.set_label('$P_{th}$ (GPa)')

fig.set_tight_layout(True)
plt.show()
