import burnman
from burnman.mineral import Mineral
from burnman.processchemistry import *
atomic_masses=read_masses()

hen=burnman.minerals.HHPH_2013.hen()
hfs=burnman.minerals.HHPH_2013.hfs()
DeltaH_fm=-6950.
class ordered_fm_hpx (Mineral):
    def __init__(self):
       formula='Mg1.0Fe1.0Si2.0O6.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'hfm',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': (hen.params['H_0'] + hfs.params['H_0'])/2.+DeltaH_fm ,
            'S_0': (hen.params['S_0'] + hfs.params['S_0'])/2. ,
            'V_0': (hen.params['V_0'] + hfs.params['V_0'])/2. ,
            'Cp': [(hen.params['Cp'][0] + hfs.params['Cp'][0])/2., 
                   (hen.params['Cp'][1] + hfs.params['Cp'][1])/2., 
                   (hen.params['Cp'][2] + hfs.params['Cp'][2])/2., 
                   (hen.params['Cp'][3] + hfs.params['Cp'][3])/2.] ,
            'a_0': (hen.params['a_0'] + hfs.params['a_0'])/2. ,
            'K_0': (hen.params['K_0'] + hfs.params['K_0'])/2. ,
            'Kprime_0': (hen.params['Kprime_0'] + hfs.params['Kprime_0'])/2. ,
            'Kdprime_0': -1.*(hen.params['Kprime_0'] + hfs.params['Kprime_0']) \
                / (hen.params['K_0'] + hfs.params['K_0']) ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)

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
    def __init__(self, molar_fractions=None):
        self.name='non-stoichiometric wuestite, ferric and ferrous iron treated as separate atoms in Sconf'
        self.type='symmetric'
        self.endmembers= [[periclase(), '[Mg]O'],[wustite(), '[Fe]O'],[defect_wustite(), '[Fef1/2Vc1/2]Fef1/6O']]
        self.enthalpy_interaction=[[11.0e3, 11.0e3], [2.0e3]]

        burnman.SolidSolution.__init__(self, molar_fractions)


        
# (Mg,Fe)2Fe2O5 - ol/wad/rw equilibrium

'''
class MgFe3O5 (Mineral): # for SLB
    def __init__(self):
       formula='Mg1.0Fe3.0O5.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'MgFe3O5',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -1705000.0 ,
            'S_0': 185.9 ,
            'V_0': 5.333e-05 ,
            'Cp': [323.0, -0.006843, -2462000.0, -1954.9] ,
            'a_0': 2.36e-05 ,
            'K_0': 1.500e+11 ,
            'Kprime_0': 4.00 ,
            'Kdprime_0': -3.080e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
'''

# Best fit for change in Kd with pressure for each mineral
class MgFe3O5 (Mineral):
    def __init__(self):
       formula='Mg1.0Fe3.0O5.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'MgFe3O5',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -1688000.0 ,
            'S_0': 185.9 ,
            'V_0': 5.333e-05 ,
            'Cp': [323.0, -0.006843, -2462000.0, -1954.9] ,
            'a_0': 2.36e-05 ,
            'K_0': 1.400e+11 ,
            'Kprime_0': 4.00 ,
            'Kdprime_0': -3.080e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)

# Best fit for ol and wad (but pressure dependence within minerals ill-fitting
'''
class MgFe3O5 (Mineral):
    def __init__(self):
       formula='Mg1.0Fe3.0O5.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'MgFe3O5',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -1685500.0 ,
            'S_0': 185.9 ,
            'V_0': 5.333e-05 ,
            'Cp': [323.0, -0.006843, -2462000.0, -1954.9] ,
            'a_0': 2.36e-05 ,
            'K_0': 1.200e+11 ,
            'Kprime_0': 4.00 ,
            'Kdprime_0': -3.080e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
'''

class MgFeFe2O5(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='(Mg,Fe)2Fe2O5'
        self.type='symmetric'
        self.endmembers = [[MgFe3O5(), '[Mg]Fe3O5'],
                           [Fe4O5(), '[Fe]Fe3O5']]
        self.enthalpy_interaction=[[0.0e3]]

        burnman.SolidSolution.__init__(self, molar_fractions)

class olivine(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='olivine'
        self.type='symmetric'
        self.endmembers = [[burnman.minerals.HHPH_2013.fo(), '[Mg]2SiO4'],
                           [burnman.minerals.HHPH_2013.fa(), '[Fe]2SiO4']]
        self.enthalpy_interaction=[[9.0e3]]

        burnman.SolidSolution.__init__(self, molar_fractions)

class wadsleyite(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='wadsleyite'
        self.type='symmetric'
        self.endmembers = [[burnman.minerals.HHPH_2013.mwd(), '[Mg]2SiO4'],
                           [burnman.minerals.HHPH_2013.fwd(), '[Fe]2SiO4']]
        self.enthalpy_interaction=[[13.0e3]]

        burnman.SolidSolution.__init__(self, molar_fractions)

class ringwoodite(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='ringwoodite'
        self.type='symmetric'
        self.endmembers = [[burnman.minerals.HHPH_2013.mrw(), '[Mg]2SiO4'],
                           [burnman.minerals.HHPH_2013.frw(), '[Fe]2SiO4']]
        self.enthalpy_interaction=[[4.0e3]]

        burnman.SolidSolution.__init__(self, molar_fractions)

class orthopyroxene(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Fe-Mg orthopyroxene'
        self.type='symmetric'
        self.endmembers=[[burnman.minerals.HHPH_2013.hen(), '[Mg][Mg]Si2O6'],
                         [burnman.minerals.HHPH_2013.hfs(), '[Fe][Fe]Si2O6'],
                         [ordered_fm_hpx(), '[Mg][Fe]Si2O6']]
        self.enthalpy_interaction=[[6.8e3, 4.5e3],
                                   [4.5e3]]

        burnman.SolidSolution.__init__(self, molar_fractions)


class CFMASO_garnet(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='garnet'
        self.type='asymmetric'
        self.endmembers = [[burnman.minerals.HP_2011_ds62.py(), '[Mg]3[Al]2Si3O12'],
                           [burnman.minerals.HP_2011_ds62.alm(), '[Fe]3[Al]2Si3O12'],
                           [burnman.minerals.HP_2011_ds62.gr(), '[Ca]3[Al]2Si3O12'],
                           [burnman.minerals.HP_2011_ds62.andr(), '[Ca]3[Fe]2Si3O12']]
        self.alphas = [1.0, 1.0, 2.7, 2.7]
        self.enthalpy_interaction=[[2.5e3, 30.1e3, 56.59e3],
                                   [1.0e3, 49.79e3],
                                   [2.96e3]]
        self.volume_interaction=[[0., 0.169e-6, 0.129e-6],
                                 [0.122e-6, 0.0288e-6],
                                 [-0.0285e-6]]
        burnman.SolidSolution.__init__(self, molar_fractions)


'''
# Powell model   
class CFMASO_garnet(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='garnet'
        self.type='asymmetric'
        self.endmembers = [[burnman.minerals.HP_2011_ds62.py(), '[Mg]3[Al]2Si3O12'],
                           [burnman.minerals.HP_2011_ds62.alm(), '[Fe]3[Al]2Si3O12'],
                           [burnman.minerals.HP_2011_ds62.gr(), '[Ca]3[Al]2Si3O12'],
                           [burnman.minerals.HP_2011_ds62.andr(), '[Ca]3[Fe]2Si3O12']]
        self.alphas = [1.0, 1.0, 2.7, 2.7]
        self.enthalpy_interaction=[[2.5e3, 31.e3, 53.2e3],
                                   [5.e3, 53.2*0.7e3],
                                   [2.e3]]
        self.volume_interaction=[[0., 0., 0.],
                                 [0., 0.],
                                 [0.]]
        burnman.SolidSolution.__init__(self, molar_fractions)
'''    
