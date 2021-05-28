import numpy as np
from scipy.optimize import brentq
import burnman_path

import burnman
from burnman import Mineral
from burnman.processchemistry import dictionarize_formula, formula_mass

assert burnman_path  # silence pyflakes warning

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
            'G_0': 59000000000.0,
            'Gprime_0': 1.44673,
            'eta_s_0': -0.05731,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}

        # add magnetic entropy according to Stixrude,
        # could change to R ln (15), which would be the Tsuchiya value...
        # or could leave it as a free variable.
        self.property_modifiers = [['linear', {'delta_E': 0.,
                                               'delta_S': burnman.constants.gas_constant*np.log(5),
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
            G0 = self.gibbs
            self.set_composition([1. - molar_fraction_FeO, 0.,
                                  molar_fraction_FeO])
            G1 = self.gibbs
            if G0 < G1:
                p_LS = 0.
            else:
                p_LS = 1.

        self.set_composition([1. - molar_fraction_FeO,
                              molar_fraction_FeO*(1. - p_LS),
                              molar_fraction_FeO*p_LS])
