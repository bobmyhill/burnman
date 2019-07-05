
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import minimize, fsolve

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman


class Murnaghan_EOS(object):
    def __init__(self, V0, K, Kprime):
        self.V0 = V0
        self.K = K
        self.Kprime = Kprime
        self.V = lambda P: self.V0*np.power(1. + P*(self.Kprime/self.K),
                                            -1./self.Kprime) 


M_mwd = Murnaghan_EOS(4.2206e-5, 146.2544e9, 4.21) # Frost, 2003 at 1673.15 K


mwd = burnman.minerals.SLB_2011.mg_wadsleyite()
mwd2 = burnman.minerals.HP_2011_ds62.mwd()

mwd_this_study = burnman.minerals.HP_2011_ds62.mwd()
mwd_this_study.name = 'mwd (this study)'

T, awd, awderr, bwd, bwderr, cwd, cwderr, Vwd, Vwderr, arw, arwerr, Vrw, Vrwerr = np.genfromtxt('data/Inoue_2004_RT_volumes_mwd_mrw.dat', unpack=True)

T = T+273.15

# WADSLEYITE

Vwd = Vwd*burnman.constants.Avogadro*1.e-30/8.
Vwderr = Vwderr*burnman.constants.Avogadro*1.e-30/8.

plt.errorbar(T, Vwd, yerr=Vwderr, linestyle='None')
plt.scatter(T, Vwd)


temperatures = np.linspace(300., 1700., 101)
pressures = 1.e5 + 0.*temperatures
plt.plot(temperatures, mwd.evaluate(['V'], pressures, temperatures)[0], label=mwd.name)
plt.plot(temperatures, mwd2.evaluate(['V'], pressures, temperatures)[0], label=mwd2.name)


PTV = np.array([1.e5 + 0.*T, T, Vwd]).T
nul = 0.*PTV.T[0]
PTV_covariances = np.array([[0.03*PTV.T[0], nul, nul], [nul, nul, nul], [nul, nul, Vwderr]]).T


params = ['V_0', 'a_0']
fitted_eos = burnman.eos_fitting.fit_PTV_data(mwd_this_study,
                                              params, PTV, PTV_covariances, verbose=False)

for i, p in enumerate(params):
    print('wd {0}: {1} +/- {2} /K'.format(p, fitted_eos.popt[i], np.sqrt(fitted_eos.pcov[i][i])))

plt.plot(temperatures, mwd_this_study.evaluate(['V'], pressures, temperatures)[0], label=mwd_this_study.name)

plt.scatter(1673.15, M_mwd.V(1.e5), label='Frost')
plt.scatter(1673.15, M_mwd.V(17.e9), label='Frost (17 GPa)')

plt.plot(temperatures, mwd2.evaluate(['V'], 17.e9 + 0.*pressures, temperatures)[0], label=mwd2.name+' (17 GPa)')
plt.plot(temperatures, mwd_this_study.evaluate(['V'], 17.e9 + 0.*pressures, temperatures)[0], label=mwd_this_study.name+' (17 GPa)')
plt.legend()
plt.show()
