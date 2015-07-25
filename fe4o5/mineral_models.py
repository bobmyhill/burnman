import burnman
from burnman.mineral import Mineral
from burnman.processchemistry import *
atomic_masses=read_masses()

# High magnetite
class high_mt (Mineral):
    def __init__(self):
       formula='Fe3.0O4.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'High pressure magnetite',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -1057460.0 ,
            'S_0': 172.4 ,
            'V_0': 4.189e-05 ,
            'Cp': [262.5, -0.007205, -1926200.0, -1655.7] ,
            'a_0': 3.59e-05 ,
            'K_0': 2.020e+11 ,
            'Kprime_0': 4.00 ,
            'Kdprime_0': 0.0e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)

# Fe4O5
class Fe4O5 (Mineral):
    def __init__(self):
       formula='Fe4.0O5.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'Fe4O5',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -1357000.0 ,
            'S_0': 218. ,
            'V_0': 5.376e-05 ,
            'Cp': [306.9, 0.001075, -3140400.0, -1470.5] ,
            'a_0': 2.36e-05 ,
            'K_0': 1.857e+11 ,
            'Kprime_0': 4.00 ,
            'Kdprime_0': -2.154e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)

# Fe4O5
class Fe5O6 (Mineral):
    def __init__(self):
       formula='Fe5.0O6.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'Fe5O6',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -1618400. , # stable at 9.5 GPa.
            'S_0': 278.6 , # Fe4O5 + FeO
            'V_0': 6.633e-05 , # Lavina and Meng, 2014, 440.6 A^3
            'Cp': [351.3, 0.009355, -4354600.0, -1285.3] , # Sum FeO, Fe3O4
            'a_0': 1.435e-05 , # Lavina and Meng, 2014
            'K_0': 1.730e+11 , # Lavina and Meng, 2014
            'Kprime_0': 4.00 , # Lavina and Meng, 2014
            'Kdprime_0': -2.312e-11 , # Heuristic
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)


class periclase (Mineral):
    def __init__(self):
       formula='Mg1.0O1.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'periclase',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -601530.0 ,
            'S_0': 26.5 ,
            'V_0': 1.125e-05 ,
            'Cp': [60.5, 0.000362, -535800.0, -299.2] ,
            'a_0': 3.11e-05 ,
            'K_0': 1.616e+11 ,
            'Kprime_0': 3.95 ,
            'Kdprime_0': -2.4e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)
       
class wustite (Mineral):
    def __init__(self):
       formula='Fe1.0O1.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'fper',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -264800. , # -264353.6
            'S_0': 59.,
            'V_0': 1.2239e-05 , # From Simons (1980)
            'Cp': [5.33343160e+01,   7.79203541e-03,  -3.25553876e+05,  -7.50233740e+01] ,
            'a_0': 3.22e-05 ,
            'K_0': 1.52e+11 ,
            'Kprime_0': 4.9 ,
            'Kdprime_0': -3.2e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)

class defect_wustite (Mineral):
    def __init__(self):
       formula='Fe2/3O1.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'fper',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -271800. ,
            'S_0': 28.,
            'V_0': 1.10701e-05 , # From Simons (1980)
            'Cp': [-3.64959181e+00,   1.29193873e-02,  -1.07988127e+06,   1.11241795e+03] ,
            'a_0': 3.22e-05 ,
            'K_0': 1.52e+11 ,
            'Kprime_0': 4.9 ,
            'Kdprime_0': -3.2e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)

# Configurational entropy
class ferropericlase(burnman.SolidSolution):
    def __init__(self):
        # Name
        self.name='non-stoichiometric wuestite, ferric and ferrous iron treated as separate atoms in Sconf'

        base_material = [[periclase(), '[Mg]O'],[wustite(), '[Fe]O'],[defect_wustite(), '[Fef1/2Vc1/2]Fef1/6O']]

        # Interaction parameters
        enthalpy_interaction=[[11.0e3, 11.0e3], [2.0e3]]

        burnman.SolidSolution.__init__(self, base_material, \
                          burnman.solutionmodel.SymmetricRegularSolution(base_material, enthalpy_interaction) )
