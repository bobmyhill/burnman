from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from excess_modelling import *

plt.rcParams['figure.figsize'] = 6, 4 # inches


# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import minerals

from scipy.optimize import brentq, curve_fit



salts = {'KCl': (62.36/1.e30*burnman.constants.Avogadro,                            17.1e9,  5.5,    'Dewaele et al., 2012'),
         'NaCl': (2.702e-05,                                                         23.83e9, 5.09,   'Dorogokupets and Dewaele, 2007'),
         'RbCl': (burnman.processchemistry.formula_mass({'Rb': 1., 'Cl': 1.})/2818., 15.6e9,  5.5001, 'Chang and Barsch, 1971'),
         'KBr':  (71.89/1.e30*burnman.constants.Avogadro,                            14.2e9,  5.5,    'Dewaele et al., 2012'),
         'NaBr': (burnman.processchemistry.formula_mass({'Na': 1., 'Br': 1.})/3210., 18.5e9,  5.8,    'Sato-Sorensen, 1983'),
         'RbBr': (burnman.processchemistry.formula_mass({'Rb': 1., 'Br': 1.})/3359., 13.0e9,  5.5001, 'Chang and Barsch, 1971'),
         'KI':   (burnman.processchemistry.formula_mass({'K': 1., 'I': 1.})/3120.,   16.0e9,  5.5,    'Sceats et al., 2006'),
         'NaI':  (burnman.processchemistry.formula_mass({'Na': 1., 'I': 1.})/3670.,  14.7e9,  5.7,    'Sato-Sorensen, 1983'),
         'RbI':  (burnman.processchemistry.formula_mass({'Rb': 1., 'I': 1.})/3564.,  10.5e9,  5.5001, 'Chang and Barsch, 1971')}


k, v = salts.keys(), salts.values()
for i, key in enumerate(k):
    print(key, v[i][0])


salt_binaries = [('KI',   'RbI',   2410., 2950.),
                 ('KCl',  'RbCl',  3371., 4084.),
                 ('KBr',  'KCl',   4030., 4030.),
                 ('NaBr',  'KCl',  5227., 6001.),
                 ('KBr',  'KI',    7437., 7437.),
                 ('KBr',  'NaBr',  12274., 16944.),
                 ('KI',   'NaI',   10614., 10614.),
                 ('NaBr', 'NaI',   7981.,  11846.),
                 ('KCl',  'NaCl', 18485., 18485.)]

meanKT = np.array([(salts[m2][1] + salts[m1][1])/2. for (m1, m2, W1, W2) in salt_binaries])
DeltaVs_obs = np.array([np.abs(2.*(salts[m2][0] - salts[m1][0])/(salts[m2][0] + salts[m1][0])) for (m1, m2, W1, W2) in salt_binaries])

print(DeltaVs_obs)
print(meanKT)

Ws_obs = np.array([0.5*(W1 + W2) for (m1, m2, W1, W2) in salt_binaries])


formula = burnman.processchemistry.dictionarize_formula('MgO')
m1_params = {'name': 'm1',
             'P_0': 1.e5,
             'T_0': 300.,
             'V_0': 1.e-5,
             'K_0': 100.e9,
             'Kprime_0': 5.5,
             'equation_of_state': 'vinet',
             'formula': formula,
             'molar_mass': burnman.processchemistry.formula_mass(formula)}

formula = burnman.processchemistry.dictionarize_formula('FeO')
m2_params = {'name': 'm2',
             'P_0': 1.e5,
             'T_0': 300.,
             'V_0': 1.e-5,
             'K_0': 100.e9,
             'Kprime_0': 5.5,
             'equation_of_state': 'vinet',
             'formula': formula,
             'molar_mass': burnman.processchemistry.formula_mass(formula)}

m1 = burnman.Mineral(params=m1_params)
m2 = burnman.Mineral(params=m2_params)

pressure = 1.e5
temperature = 300.
x_m1 = 0.5
cluster_size = 2.


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
colors = [next(ax._get_lines.prop_cycler)['color'] for i in range(10)]


from scipy.optimize import brentq


def internal_energy(volume, params):
    """
    Returns the internal energy :math:`\mathcal{E}` of the mineral. :math:`[J/mol]`
    """
    x = np.power(volume/params['V_0'], 1./3.)
    eta = (3. / 2.) * (params['Kprime_0'] - 1.)
    
    
    intPdV = (9.* params['V_0'] * params['K_0'] / (eta*eta) *
              (1. - eta*(1. - x))*np.exp(eta*(1. - x)))

    intPdV0 = (9.* params['V_0'] * params['K_0'] / (eta*eta))
    
    return - intPdV + intPdV0 + params['E_0']
    
def _deltaPvinet(volume, pressure, p1, m1, m2, mix):
    dV = 1.e-11
    E01 = internal_energy(volume-dV/2., m1.params)
    E02 = internal_energy(volume-dV/2., m2.params)
    E0 = p1*E01 + (1. - p1)*E02
    E11 = internal_energy(volume+dV/2., m1.params)
    E12 = internal_energy(volume+dV/2., m2.params)
    E1 = p1*E11 + (1. - p1)*E12


    E1 = internal_energy(volume, m1.params)
    E2 = internal_energy(volume, m2.params)
    mix.E = p1*(E1) + (1. - p1)*(E2)
    P = -(E1 - E0)/dV
    return P - pressure           

class dummy(object):
    E = 0.


mix = dummy()


Vmax = 0.4
for j, (Vm, K_m1, K_m2) in enumerate(np.array([(4.5e-5, 10.e9, 10.e9),
                                               (4.5e-5, 15.e9, 15.e9),
                                               (4.5e-5, 20.e9, 20.e9),
                                               (4.5e-5, 25.e9, 25.e9)])):
    m1.params['K_0'] = K_m1
    m2.params['K_0'] = K_m2

    DeltaVs = np.linspace(0., Vmax, 101)
    enthalpies = np.empty_like(DeltaVs)
    for i, DeltaV in enumerate(DeltaVs):
        m1.params['V_0'] = Vm - DeltaV*Vm/2.
        m2.params['V_0'] = Vm + DeltaV*Vm/2.

        volume = brentq(_deltaPvinet, 1.e-5, 6.e-5, args=(0., x_m1, m1, m2, mix))
        enthalpies[i] = (mix.E)/1000./cluster_size


    ax.plot(DeltaVs, enthalpies,
             label='$\\bar{{V}}$ = 45 cm$^3$/mol, $K_T$ = {0} GPa'.format(K_m1/1.e9, K_m2/1.e9),
             color = colors[j])
    
ax.scatter(DeltaVs_obs, Ws_obs/1000./4., label='Davies and Navrotsky, 1983')
ax.set_xlim(0., 0.35)
ax.set_ylim(0., 8.)
ax.set_xlabel('$\Delta V$')
ax.set_ylabel('$\mathcal{H}_{excess}$ (kJ/mol)')
ax.legend(loc='best')

fig.savefig("enthalpy_salts.pdf", bbox_inches='tight', dpi=100)
plt.show()
