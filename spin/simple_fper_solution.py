from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import Mineral
from burnman.processchemistry import dictionarize_formula, formula_mass

periclase = burnman.minerals.SLB_2011.periclase

class high_spin_wuestite (Mineral):

    def __init__(self):
        formula = 'FeO'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'high spin Wuestite',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -242146.0,
            'V_0': 12.264e-06,  # SLB
            'K_0': 1.794442e+11,  # SLB
            'Kprime_0': 4.9376,  # SLB
            'Debye_0': 454.1592,  # SLB
            'grueneisen_0': 1.53047,  # SLB
            'q_0': 1.7217,  # SLB
            #'K_0': 160.0e9,
            #'Kprime_0': 4.0,
            #'Debye_0': 800.,
            #'grueneisen_0': 1.8,
            #'q_0': 1.5,
            'G_0': 59000000000.0,
            'Gprime_0': 1.44673,
            'eta_s_0': -0.05731,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}

        self.property_modifiers = [['linear', {'delta_E': 0.,
                                               'delta_S': 8.31446*np.log(5),
                                               'delta_V': 0.}]]
        Mineral.__init__(self)


class low_spin_wuestite (Mineral):

    def __init__(self):
        formula = 'FeO'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'low spin Wuestite',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -242146.0 + 60000.,
            'V_0': 11.24e-6 - 0.20e-6,
            'K_0': 160.2e9 + 25.144e9,
            #'Kprime_0': 4.9376,  # SLB
            #'Debye_0': 454.1592,  # SLB
            #'grueneisen_0': 1.53047,  # SLB
            #'q_0': 1.7217,  # SLB
            'Kprime_0': 4.0,
            'Debye_0': 800.,
            'grueneisen_0': 1.8,
            'q_0': 1.5,
            'G_0': 59000000000.0,
            'Gprime_0': 1.44673,
            'eta_s_0': -0.05731,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}

        Mineral.__init__(self)

class ferropericlase(burnman.SolidSolution):

    """
    Solid solution class for ferropericlase
    that includes a new method "set_equilibrium_composition"
    that finds the equilibrium distribution of high spin and low spin iron
    at the current state.
    """

    def __init__(self, molar_fractions=None):
        self.name = 'ferropericlase'
        self.solution_type = 'symmetric'
        self.endmembers = [[periclase(), '[Mg]O'],
                           [high_spin_wuestite(), '[Fehs]O'],
                           [low_spin_wuestite(), '[Fels]O']]
        self.energy_interaction = [[11.e3, 11.e3],
                                   [11.e3]]
        burnman.SolidSolution.__init__(self, molar_fractions=molar_fractions)

    def set_equilibrium_composition(self, molar_fraction_FeO):

        def delta_mu(p_LS):
            self.set_composition([1. - molar_fraction_FeO,
                                  molar_fraction_FeO*(1. - p_LS),
                                  molar_fraction_FeO*p_LS])
            return self.partial_gibbs[1] - self.partial_gibbs[2]

        try:
            p_LS = brentq(delta_mu, 0., 1.)
        except ValueError:
            self.set_composition([1. - molar_fraction_FeO,
                                  molar_fraction_FeO, 0.])
            G0 = fper.gibbs
            self.set_composition([1. - molar_fraction_FeO, 0.,
                                  molar_fraction_FeO])
            G1 = fper.gibbs
            if G0 < G1:
                p_LS = 0.
            else:
                p_LS = 1.

        self.set_composition([1. - molar_fraction_FeO,
                              molar_fraction_FeO*(1. - p_LS),
                              molar_fraction_FeO*p_LS])


fper = ferropericlase()

pressures = np.linspace(10.e9, 150.e9, 101)
volumes = np.empty_like(pressures)
volumes_HS = np.empty_like(pressures)
volumes_LS = np.empty_like(pressures)
delta_MgO_volumes = np.empty_like(pressures)
p_LS = np.empty_like(pressures)

fig = plt.figure(figsize=(15,5))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]


dat = np.loadtxt('Komabayashi_2010.dat')

per = burnman.minerals.SLB_2011.periclase()
#per = burnman.minerals.HP_2011_ds62.per()
expt_MgO_volumes = np.empty_like(dat[:,0])
for i, d in enumerate(dat):
    per.set_state(d[2]*1.e9, d[0])
    expt_MgO_volumes[i] = per.V

ax[0].scatter(dat[:, 2], dat[:, 6]*6.022/4./10.)
ax[2].scatter(dat[:, 2], dat[:, 6] - expt_MgO_volumes/6.022*4.*10.e6)

X_Fe = 0.19 # Komabayashi et al. (2010) composition
for T in [300., 1800., 3300.]:
    for i, P in enumerate(pressures):

        fper.set_state(P, T)
        fper.set_equilibrium_composition(X_Fe)

        volumes[i] = fper.V
        p_LS[i] = (fper.molar_fractions[2]
                   / (fper.molar_fractions[1]+fper.molar_fractions[2]))

        fper.set_composition([1., 0., 0.])
        delta_MgO_volumes[i] = volumes[i] - fper.V

        fper.set_composition([1.-X_Fe, X_Fe, 0.])
        volumes_HS[i] = fper.V
        fper.set_composition([1.-X_Fe, 0., X_Fe])
        volumes_LS[i] = fper.V

    ax[0].plot(pressures/1.e9, volumes*1.e6, label=f'{T} K')
    ax[0].plot(pressures/1.e9, volumes_HS*1.e6, linestyle=':')
    ax[0].fill_between(pressures/1.e9, volumes_HS, volumes_LS, alpha=0.2)
    ax[1].plot(pressures/1.e9, 1.-p_LS, label=f'{T} K')
    ax[2].plot(pressures/1.e9, delta_MgO_volumes/6.022*4.*10.e6, label=f'{T} K')


ax[0].set_xlabel('Pressure (GPa)')
ax[1].set_xlabel('Pressure (GPa)')
ax[0].set_ylabel('Volume (cm$^3$/mol)')
ax[1].set_ylabel('high spin fraction')

ax[0].set_ylim(7,13)
ax[0].legend()
ax[1].legend()
plt.show()
