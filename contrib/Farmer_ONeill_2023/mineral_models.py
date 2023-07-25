from __future__ import absolute_import

from burnman.classes.mineral import Mineral
from burnman.utils.chemistry import dictionarize_formula, formula_mass
from burnman.classes.solution import Solution
from burnman.classes.solutionmodel import IdealSolution


class rs_MgO(Mineral):
    def __init__(self):
        structure = "rocksalt"
        formula_string = "MgO"
        formula = dictionarize_formula(formula_string)
        self.params = {
            "name": f"{structure}-structured {formula_string}",
            "formula": formula,
            "equation_of_state": "hp_tmt",
            "H_0": -601.55e3,
            "S_0": 26.5,
            "V_0": 1.125e-05,
            "Cp": [
                60.5,
                0.000362,
                -535800.0,
                -299.2,
            ],  # note typo in paper, Cp_D should be -ve, figures seem to use correct value
            "a_0": 3.11e-05,
            "K_0": 161600e6,
            "Kprime_0": 3.95,
            "Kdprime_0": -2.4e-11,
            "n": sum(formula.values()),
            "molar_mass": formula_mass(formula),
        }
        Mineral.__init__(self)


class wz_MgO(Mineral):
    def __init__(self):
        structure = "wurtzite"
        formula_string = "MgO"
        formula = dictionarize_formula(formula_string)
        self.params = {
            "name": f"{structure}-structured {formula_string}",
            "formula": formula,
            "equation_of_state": "hp_tmt",
            "H_0": -578.19e3,
            "S_0": 29.11,
            "V_0": 1.293e-05,
            "Cp": [63.1, 0.000973, -260000.0, -381.0],
            "a_0": 3.78e-05,
            "K_0": 152100e6,
            "Kprime_0": 3.775,
            "Kdprime_0": -2.48e-11,
            "n": sum(formula.values()),
            "molar_mass": formula_mass(formula),
        }
        Mineral.__init__(self)


class wz_ZnO(Mineral):
    def __init__(self):
        structure = "wurtzite"
        formula_string = "ZnO"
        formula = dictionarize_formula(formula_string)
        self.params = {
            "name": f"{structure}-structured {formula_string}",
            "formula": formula,
            "equation_of_state": "hp_tmt",
            "H_0": -350.5e3,
            "S_0": 43.16,
            "V_0": 1.434e-05,
            "Cp": [43.5, 0.007660, -757300.0, 54.56],
            "a_0": 4.45e-05,
            "K_0": 142600e6,
            "Kprime_0": 3.6,
            "Kdprime_0": -2.52e-11,
            "n": sum(formula.values()),
            "molar_mass": formula_mass(formula),
        }
        Mineral.__init__(self)


class rs_ZnO(Mineral):
    def __init__(self):
        structure = "rocksalt"
        formula_string = "ZnO"
        formula = dictionarize_formula(formula_string)
        self.params = {
            "name": f"{structure}-structured {formula_string}",
            "formula": formula,
            "equation_of_state": "hp_tmt",
            "H_0": -335.93e3,
            "S_0": 44.57,
            "V_0": 1.179e-05,
            "Cp": [61.4, 0.007220, -017090.0, -398.3],
            "a_0": 5.20e-05,
            "K_0": 191000e6,
            "Kprime_0": 3.54,
            "Kdprime_0": -1.85e-11,
            "n": sum(formula.values()),
            "molar_mass": formula_mass(formula),
        }
        Mineral.__init__(self)


class rocksalt(Solution):
    def __init__(self, molar_fractions=None):
        self.name = "MgO-ZnO rocksalt"
        self.solution_model = IdealSolution(
            endmembers=[
                [rs_MgO(), "[Mg]O"],
                [rs_ZnO(), "[Zn]O"],
            ]
        )

        Solution.__init__(self, molar_fractions=molar_fractions)


class wurtzite(Solution):
    def __init__(self, molar_fractions=None):
        self.name = "MgO-ZnO wurtzite"
        self.solution_model = IdealSolution(
            endmembers=[
                [wz_MgO(), "[Mg]O"],
                [wz_ZnO(), "[Zn]O"],
            ]
        )

        Solution.__init__(self, molar_fractions=molar_fractions)
