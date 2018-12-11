
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


M_mrw = Murnaghan_EOS(4.1484e-5, 145.3028e9, 4.4) # Frost, 2003 at 1673.15 K


mrw = burnman.minerals.SLB_2011.mg_ringwoodite()
mrw2 = burnman.minerals.HP_2011_ds62.mrw()

mrw_this_study = burnman.minerals.HP_2011_ds62.mrw()
mrw_this_study.name = 'mrw (this study)'

T, awd, awderr, bwd, bwderr, cwd, cwderr, Vwd, Vwderr, arw, arwerr, Vrw, Vrwerr = np.genfromtxt('data/Inoue_2004_RT_volumes_mwd_mrw.dat', unpack=True)

T = T+273.15

# RINGWOODITE

Vrw = Vrw*burnman.constants.Avogadro*1.e-30/8.
Vrwerr = Vrwerr*burnman.constants.Avogadro*1.e-30/8.

plt.errorbar(T, Vrw, yerr=Vrwerr, linestyle='None')
plt.scatter(T, Vrw)


temperatures = np.linspace(300., 1700., 101)
pressures = 1.e5 + 0.*temperatures
plt.plot(temperatures, mrw.evaluate(['V'], pressures, temperatures)[0], label=mrw.name)
plt.plot(temperatures, mrw2.evaluate(['V'], pressures, temperatures)[0], label=mrw2.name)


PTV = np.array([1.e5 + 0.*T, T, Vrw]).T
nul = 0.*PTV.T[0]
PTV_covariances = np.array([[0.03*PTV.T[0], nul, nul], [nul, nul, nul], [nul, nul, Vrwerr]]).T


params = ['a_0']
fitted_eos = burnman.eos_fitting.fit_PTV_data(mrw_this_study,
                                              params, PTV, PTV_covariances, verbose=False)

print('rw alpha: {0} +/- {1} /K'.format(fitted_eos.popt[0], np.sqrt(fitted_eos.pcov[0][0])))

plt.plot(temperatures, mrw_this_study.evaluate(['V'], pressures, temperatures)[0], label=mrw_this_study.name)

plt.scatter(1673.15, M_mrw.V(1.e5), label='Frost')
plt.scatter(1673.15, M_mrw.V(20.e9), label='Frost (20 GPa)')

plt.plot(temperatures, mrw_this_study.evaluate(['V'], 20.e9 + 0.*pressures, temperatures)[0], label=mrw_this_study.name+' (20 GPa)')
plt.legend()
plt.show()
