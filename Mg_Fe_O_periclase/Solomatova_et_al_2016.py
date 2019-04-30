# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import Mineral, minerals
from burnman.processchemistry import dictionarize_formula, formula_mass

def linear(x, a, b):
    return a + b*x

# Xfe, Ptr, Perr, Pwidth
Ptr = np.array([[0.1, 52.1, 1.9, 2.3],
                [0.17, 49.2, 1.3, 8.2],
                [0.25, 62.6, 7.0, 15.9],
                [0.35, 63.7, 1.7, 13.3],
                [0.39, 56.6, 1.5, 17.2],
                [0.39, 73.5, 1.0, 10.3],
                [0.48, 68.8, 2.7, 18.7],
                [0.60, 78.6, 5.1, 25.1]])


from scipy.optimize import curve_fit


popt0, pcov = curve_fit(linear, xdata=Ptr.T[0], ydata=Ptr.T[1]-Ptr.T[3])
popt1, pcov = curve_fit(linear, xdata=Ptr.T[0], ydata=Ptr.T[1]+Ptr.T[3])

X = np.array([0., 1.])


plt.plot(X, linear(X, *popt0))
plt.plot(X, linear(X, *popt1))

plt.errorbar(Ptr.T[0], Ptr.T[1], yerr=Ptr.T[2], linestyle='None')
plt.errorbar(Ptr.T[0], Ptr.T[1] - Ptr.T[3], yerr=Ptr.T[2], linestyle='None')
plt.errorbar(Ptr.T[0], Ptr.T[1] + Ptr.T[3], yerr=Ptr.T[2], linestyle='None')
plt.scatter(Ptr.T[0], Ptr.T[1])
plt.scatter(Ptr.T[0], Ptr.T[1] - Ptr.T[3], label='20 %')
plt.scatter(Ptr.T[0], Ptr.T[1] + Ptr.T[3], label='80 %')
plt.legend(loc='best')
plt.show()
