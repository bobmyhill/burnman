from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

"""
"""
import burnman
from burnman.mineral import Mineral
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
            'V_0': 12.264e-06,
            'K_0': 160.0e9,
            'Kprime_0': 4.0,
            'Debye_0': 454.1592,
            'grueneisen_0': 1.53047,
            'q_0': 1.7217,
            'G_0': 59000000000.0,
            'Gprime_0': 1.44673,
            'eta_s_0': -0.05731,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}

        Mineral.__init__(self)


class low_spin_wuestite (Mineral):

    def __init__(self):
        formula = 'FeO'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'low spin Wuestite',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -242146.0 + 70000.,
            'V_0': 11.24e-6 - 0.309e-6,
            'K_0': 160.2e9 + 25.144e9,
            'Kprime_0': 4.0,
            'Debye_0': 454.1592,
            'grueneisen_0': 1.53047,
            'q_0': 1.7217,
            'G_0': 59000000000.0,
            'Gprime_0': 1.44673,
            'eta_s_0': -0.05731,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}

        Mineral.__init__(self)


class high_spin_wuestite (Mineral):

    def __init__(self):
        formula = 'FeO'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'high spin Wuestite',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -242146.0,
            'V_0': 12.264e-06,
            'K_0': 160.0e9,
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


class low_spin_wuestite (Mineral):

    def __init__(self):
        formula = 'FeO'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'low spin Wuestite',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -242146.0 + 65000.,
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
