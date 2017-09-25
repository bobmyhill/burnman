from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from excess_modelling import *

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import minerals

from scipy.optimize import brentq, curve_fit


data = np.loadtxt('data/Fort_Moore_xrhovphi.dat')

for datum in data:
    C1, C2 = datum[0:2]
    
    xs = np.array([datum[i] for i in [2, 5, 8, 11, 14, 17]])
    rho = np.array([datum[i] for i in [3, 6, 9, 12, 15, 18]])
    vphi = np.array([datum[i] for i in [4, 7, 10, 13, 16, 19]])


    K_S = vphi*vphi*rho
    V = 1./rho
    fig = plt.figure()
    ax = [fig.add_subplot(2, 1, i) for i in range(1,3)]
    ax[0].scatter(xs, V - (xs*V[-1] + (1. - xs)*V[0]))
    ax[1].scatter(xs, K_S - (xs*K_S[-1] + (1. - xs)*K_S[0]))
    plt.show()
