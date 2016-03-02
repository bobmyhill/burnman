# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))


from scipy.interpolate import interp1d
from scipy.optimize import fsolve, curve_fit
import burnman
from burnman import minerals
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()

def quadratic(X, *params):
    y0, y1, a = params
    return y1*X + y0*(1. - X) + a*X*(1. - X)


def quad_fit(X, Y, maxx):
    X1 = []
    Y1 = []
    for i, x in enumerate(X):
        if x < maxx:
            X1.append(X[i])
            Y1.append(Y[i])
    popt, pcov = curve_fit(quadratic, X1, Y1, [Y1[0], Y1[-1], 0.])
    print popt
    return popt, pcov

def linear(X, *params):
    y0, y1 = params
    return y1*X + y0*(1. - X)


def lin_fit(X, Y, maxx):
    X1 = []
    Y1 = []
    for i, x in enumerate(X):
        if x < maxx:
            X1.append(X[i])
            Y1.append(Y[i])
    popt, pcov = curve_fit(linear, X1, Y1, [Y1[0], Y1[-1]])
    print popt
    return popt, pcov
    
# at% Tmelt (K) rho (kg/m^3) alpha (kg/m^3/K) uncertainty (kg/m^3)
volume_data = burnman.tools.array_from_file('data/Mizuno_Fe_Si_liquid_volumes.dat')
Fe_volume_data = burnman.tools.array_from_file('data/Mizuno_Fe_melt_VT.dat')

X, Tmelt, rhomelt, drhodT, U = volume_data
X = X/100.

TFe, rhoFe = Fe_volume_data
VFe = 0.055845/rhoFe

plt.plot(TFe, VFe, marker='o', linestyle='None')
plt.plot(TFe, 0.055845/(rhomelt[0] + (TFe - 1808.)*drhodT[0]))
plt.show()

rho1808 = rhomelt + drhodT*(1808. - Tmelt)
Vmelt =  0.055845/rhomelt # kg/mol/kg*m^3
Vmelt1808 = 0.055845/rho1808

plt.plot(X, Vmelt1808, marker='o', linestyle='None')

popt, pcov = lin_fit(X, Vmelt1808, 0.51)
plt.plot(X, linear(X, *popt))
popt, pcov = quad_fit(X, Vmelt1808, 1.01)
plt.plot(X, quadratic(X, *popt))
plt.show()

# M = [kg/mol]
# rho = [kg/m^3]


# alpha = (1/V)*(dV/dT) = (rho/M)*(dVdrho*drhodT) = (rho/M)*(-M/rho/rho * drhodT) = -drhodT/rho
alpha = -drhodT/rho1808 
plt.plot(X, alpha, marker='o', linestyle='None')

popt, pcov = lin_fit(X, alpha, 0.51)
plt.plot(X, linear(X, *popt))
popt, pcov = lin_fit(X, alpha, 1.01)
plt.plot(X, linear(X, *popt))
plt.show()

plt.plot(X, Tmelt)
plt.show()
