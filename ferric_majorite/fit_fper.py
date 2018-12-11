
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


# Frost equations of state at 1673.15 K
M_per = Murnaghan_EOS(1.1932e-5, 125.9e9, 4.1)
M_fper = Murnaghan_EOS(1.2911e-5, 152.6e9, 4.)


per = burnman.minerals.HP_2011_ds62.per()
per_SLB = burnman.minerals.SLB_2011.periclase()
fper = burnman.minerals.HP_2011_ds62.fper()
fper_SLB = burnman.minerals.SLB_2011.wuestite()



pressures = np.linspace(1.e5, 100.e9, 101)
temperatures = 1673. + 0.*pressures

plt.plot(pressures/1.e9, per.evaluate(['V'], pressures, temperatures)[0], label=per.name)
plt.plot(pressures/1.e9, per_SLB.evaluate(['V'], pressures, temperatures)[0], label=per_SLB.name)
plt.plot(pressures/1.e9, fper.evaluate(['V'], pressures, temperatures)[0], label=fper.name)
plt.plot(pressures/1.e9, fper_SLB.evaluate(['V'], pressures, temperatures)[0], label=fper_SLB.name)
plt.plot(pressures/1.e9, M_per.V(pressures), label='Frost per')
plt.plot(pressures/1.e9, M_fper.V(pressures), label='Frost fper')
plt.legend()
plt.show()



print('SLB', fper_SLB.params['V_0'], fper_SLB.params['K_0'], fper_SLB.params['Kprime_0'])
print('HP', fper.params['V_0'], fper.params['K_0'], fper.params['Kprime_0'])
print(per.params['a_0'], fper.params['a_0'])

# Find the ideal volumes for fper from the solid solution (assuming periclase is accurate)
Z = 4.

for (x, V_0, K_0, Kprime_0) in [[0.39, 77.48*burnman.constants.Avogadro/1.e30/Z, 156.e9, 4.], # Fei et al., 2007
                                [0.48, 77.29*burnman.constants.Avogadro/1.e30/Z, 160.e9, 4.12], # Solomatova et al., 2016
                                [0.25, 76.34*burnman.constants.Avogadro/1.e30/Z, 162.e9, 4.], # Mao et al., 2011
                                [1.0, 12.256e-6, 149.4e9, 3.6]]: # Fischer et al., 2011

    f = {'Mg': 1. - x, 'Fe': x, 'O': 1.}
    fper_ss = burnman.Mineral(params = {'name': 'fper'+str(int(x*100)),
                                        'formula': f, 
                                        'equation_of_state': 'bm3',
                                        'V_0': V_0,
                                        'K_0': K_0,
                                        'Kprime_0': Kprime_0,
                                        'molar_mass': burnman.processchemistry.formula_mass(f)})
    pressures = np.linspace(1.e5, 100.e9, 101)
    temperatures = 300. + 0.*pressures
    per_volumes = per.evaluate(['V'], pressures, temperatures)[0]
    fper_ss_volumes = fper_ss.evaluate(['V'], pressures, temperatures)[0]
    plt.plot(pressures/1.e9, per_volumes, label=per.name)
    plt.plot(pressures/1.e9, fper_ss_volumes, label=fper_ss.name)
    
    fper_volumes = (fper_ss_volumes - (1. - x)*per_volumes)/x
    PTV = np.array([pressures, temperatures, fper_volumes]).T
    nul = 0.*PTV.T[0]
    PTV_covariances = np.array([[0.03*PTV.T[0], nul, nul], [nul, nul, nul], [nul, nul, 0.01*fper_volumes]]).T
    
    fper_this_study = burnman.minerals.HP_2011_ds62.fper()
    fper_this_study.name = 'fper from {0}'.format(fper_ss.name)
    params = ['V_0', 'K_0', 'Kprime_0']
    si_to_unit = [1.e6, 1.e-9, 1.]
    unit=['cm^3', 'GPa', '']
    fitted_eos = burnman.eos_fitting.fit_PTV_data(fper_this_study,
                                                  params, PTV, PTV_covariances, verbose=False)
    
    for i, p in enumerate(params):
        print('{0}: {1} +/- {2} {3}'.format(p, fitted_eos.popt[i]*si_to_unit[i], np.sqrt(fitted_eos.pcov[i][i])*si_to_unit[i], unit[i]))
        
    plt.plot(pressures/1.e9, fper_this_study.evaluate(['V'], pressures, temperatures)[0], label=fper_this_study.name)
        

plt.plot(pressures/1.e9, fper.evaluate(['V'], pressures, temperatures)[0], label='HP')
plt.plot(pressures/1.e9, fper_SLB.evaluate(['V'], pressures, temperatures)[0], label='SLB')
plt.plot(pressures/1.e9, per_SLB.evaluate(['V'], pressures, temperatures)[0], label='SLB')


fper_this_study = burnman.minerals.HP_2011_ds62.fper()
fper_this_study.name = 'fper (this study)'




plt.legend()
plt.show()
