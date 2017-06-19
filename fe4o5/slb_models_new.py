import burnman
from burnman.mineral import Mineral
from burnman.processchemistry import *
atomic_masses=read_masses()


class forsterite (Mineral):
    def __init__(self):
        formula='Mg2SiO4'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Forsterite',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -2055000.0 - 146105.680393 ,
            'V_0': 4.36e-05 ,
            'K_0': 1.28e+11 ,
            'Kprime_0': 4.2 ,
            'Debye_0': 809.0 ,
            'grueneisen_0': 0.99 ,
            'q_0': 2.1 ,
            'G_0': 82000000000.0 ,
            'Gprime_0': 1.5 ,
            'eta_s_0': 2.3 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        self.uncertainties = {
            'err_F_0': 2000.0 ,
            'err_V_0': 0.0 ,
            'err_K_0': 2000000000.0 ,
            'err_K_prime_0': 0.2 ,
            'err_Debye_0': 1.0 ,
            'err_grueneisen_0': 0.03 ,
            'err_q_0': 0.2 ,
            'err_G_0': 2000000000.0 ,
            'err_Gprime_0': 0.1 ,
            'err_eta_s_0': 0.1 }
        Mineral.__init__(self)

class fayalite (Mineral):
    def __init__(self):
        formula='Fe2SiO4'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Fayalite',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -1371000.0 - 151968.524763,
            'V_0': 4.629e-05 ,
            'K_0': 1.35e+11 ,
            'Kprime_0': 4.2 ,
            'Debye_0': 619.0 ,
            'grueneisen_0': 1.06 ,
            'q_0': 3.6 ,
            'G_0': 51000000000.0 ,
            'Gprime_0': 1.5 ,
            'eta_s_0': 1.0 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        self.uncertainties = {
            'err_F_0': 1000.0 ,
            'err_V_0': 0.0 ,
            'err_K_0': 2000000000.0 ,
            'err_K_prime_0': 1.0 ,
            'err_Debye_0': 2.0 ,
            'err_grueneisen_0': 0.07 ,
            'err_q_0': 1.0 ,
            'err_G_0': 2000000000.0 ,
            'err_Gprime_0': 0.5 ,
            'err_eta_s_0': 0.6 }
        Mineral.__init__(self)

class mg_wadsleyite (Mineral):
    def __init__(self):
        formula='Mg2SiO4'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Mg_Wadsleyite',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -2028000.0 - 146105.680393,
            'V_0': 4.052e-05 ,
            'K_0': 1.69e+11 ,
            'Kprime_0': 4.3 ,
            'Debye_0': 844.0 ,
            'grueneisen_0': 1.21 ,
            'q_0': 2.0 ,
            'G_0': 1.12e+11 ,
            'Gprime_0': 1.4 ,
            'eta_s_0': 2.6 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        self.uncertainties = {
            'err_F_0': 2000.0 ,
            'err_V_0': 0.0 ,
            'err_K_0': 3000000000.0 ,
            'err_K_prime_0': 0.2 ,
            'err_Debye_0': 7.0 ,
            'err_grueneisen_0': 0.09 ,
            'err_q_0': 1.0 ,
            'err_G_0': 2000000000.0 ,
            'err_Gprime_0': 0.2 ,
            'err_eta_s_0': 0.4 }
        Mineral.__init__(self)

class fe_wadsleyite (Mineral):
    def __init__(self):
        formula='Fe2SiO4'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Fe_Wadsleyite',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -1365000.0 - 151968.524763,
            'V_0': 4.28e-05 ,
            'K_0': 1.69e+11 ,
            'Kprime_0': 4.3 ,
            'Debye_0': 665.0 ,
            'grueneisen_0': 1.21 ,
            'q_0': 2.0 ,
            'G_0': 72000000000.0 ,
            'Gprime_0': 1.4 ,
            'eta_s_0': 1.0 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        self.uncertainties = {
            'err_F_0': 7000.0 ,
            'err_V_0': 0.0 ,
            'err_K_0': 13000000000.0 ,
            'err_K_prime_0': 1.0 ,
            'err_Debye_0': 21.0 ,
            'err_grueneisen_0': 0.3 ,
            'err_q_0': 1.0 ,
            'err_G_0': 12000000000.0 ,
            'err_Gprime_0': 0.5 ,
            'err_eta_s_0': 1.0 }
        Mineral.__init__(self)

class mg_ringwoodite (Mineral):
    def __init__(self):
        formula='Mg2SiO4'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Mg_Ringwoodite',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -2017000.0 - 146105.680393,
            'V_0': 3.949e-05 ,
            'K_0': 1.85e+11 ,
            'Kprime_0': 4.2 ,
            'Debye_0': 878.0 ,
            'grueneisen_0': 1.11 ,
            'q_0': 2.4 ,
            'G_0': 1.23e+11 ,
            'Gprime_0': 1.4 ,
            'eta_s_0': 2.3 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        self.uncertainties = {
            'err_F_0': 2000.0 ,
            'err_V_0': 0.0 ,
            'err_K_0': 2000000000.0 ,
            'err_K_prime_0': 0.2 ,
            'err_Debye_0': 8.0 ,
            'err_grueneisen_0': 0.1 ,
            'err_q_0': 0.4 ,
            'err_G_0': 2000000000.0 ,
            'err_Gprime_0': 0.1 ,
            'err_eta_s_0': 0.5 }
        Mineral.__init__(self)

class fe_ringwoodite (Mineral):
    def __init__(self):
        formula='Fe2SiO4'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Fe_Ringwoodite',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -1363000.0 - 151968.524763,
            'V_0': 4.186e-05 ,
            'K_0': 2.13e+11 ,
            'Kprime_0': 4.2 ,
            'Debye_0': 679.0 ,
            'grueneisen_0': 1.27 ,
            'q_0': 2.4 ,
            'G_0': 92000000000.0 ,
            'Gprime_0': 1.4 ,
            'eta_s_0': 1.8 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        self.uncertainties = {
            'err_F_0': 2000.0 ,
            'err_V_0': 0.0 ,
            'err_K_0': 7000000000.0 ,
            'err_K_prime_0': 1.0 ,
            'err_Debye_0': 8.0 ,
            'err_grueneisen_0': 0.23 ,
            'err_q_0': 1.0 ,
            'err_G_0': 10000000000.0 ,
            'err_Gprime_0': 0.5 ,
            'err_eta_s_0': 1.0 }
        Mineral.__init__(self)


from burnman.solidsolution import SolidSolution

class mg_fe_olivine(SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='olivine'
        self.type='symmetric'
        self.endmembers = [[forsterite(), '[Mg]2SiO4'],[fayalite(), '[Fe]2SiO4']]
        self.enthalpy_interaction=[[7.81322e3]]

        SolidSolution.__init__(self, molar_fractions)


class mg_fe_wadsleyite(SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='wadsleyite'
        self.type='symmetric'
        self.endmembers = [[mg_wadsleyite(), '[Mg]2SiO4'],[fe_wadsleyite(), '[Fe]2SiO4']]
        self.enthalpy_interaction=[[16.74718e3]]

        SolidSolution.__init__(self, molar_fractions)

        
class mg_fe_ringwoodite(SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='ringwoodite'
        self.type='symmetric'
        self.endmembers = [[mg_ringwoodite(), '[Mg]2SiO4'],[fe_ringwoodite(), '[Fe]2SiO4']]
        self.enthalpy_interaction=[[9.34084e3]]

        SolidSolution.__init__(self, molar_fractions)
