import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
import numpy as np
from scipy import optimize, integrate
import matplotlib.pyplot as plt
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass

atomic_masses=read_masses()

# Cp = a + bT + cT^-2 + dT^-0.5
class KCl_solid (burnman.Mineral):
    def __init__(self):
        formula='K1.0Cl1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'KCl',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -436684.1 ,
            'S_0': 82.55032 ,
            'V_0': 4.366e-05 , # 
            'Cp': [40.01578, 25.46801e-3, 3.64845e5, 0.0] ,
            'a_0': 2.85e-05 , # 
            'K_0': 1.285e+11 , # 
            'Kprime_0': 3.84 , # 
            'Kdprime_0': -3e-11 , # 
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        burnman.Mineral.__init__(self)

class KCl_liquid (burnman.Mineral):
    def __init__(self):
        formula='K1.0Cl1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'KCl liquid',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -421824.9 ,
            'S_0': 86.52250 ,
            'V_0': 4.366e-05 , # 
            'Cp': [73.59656, 0., 0., 0.] ,
            'a_0': 2.85e-05 , # 
            'K_0': 1.285e+11 , # 
            'Kprime_0': 3.84 , # 
            'Kdprime_0': -3e-11 , # 
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        burnman.Mineral.__init__(self)

class MgCl2_solid (burnman.Mineral):
    def __init__(self):
        formula='Mg1.0Cl2.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'MgCl2 solid',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -641616.0 ,
            'S_0': 89.62900 ,
            'V_0': 4.366e-05 , # 
            'Cp': [54.58434, 21.42127e-3, -11.12119e5, 399.1767],# w/o T^2 term
            'a_0': 2.85e-05 , # 
            'K_0': 1.285e+11 , # 
            'Kprime_0': 3.84 , # 
            'Kdprime_0': -3e-11 , # 
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        burnman.Mineral.__init__(self)

class MgCl2_liquid (burnman.Mineral):
    def __init__(self):
        formula='Mg1.0Cl2.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'MgCl2 liquid',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -606887.4 ,
            'S_0': 117.29708 ,
            'V_0': 4.366e-05 , # 
            'Cp': [92.048, 0., 0., 0.] ,
            'a_0': 2.85e-05 , # 
            'K_0': 1.285e+11 , # 
            'Kprime_0': 3.84 , # 
            'Kdprime_0': -3e-11 , # 
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        burnman.Mineral.__init__(self)

class KCl_MgCl2_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='garnet'
        self.endmembers = [[KCl_liquid(), '[K]Cl'], [MgCl2_liquid(), '[Mg]Cl2']]
        self.type='subregular'
        self.enthalpy_interaction=[[[-60.e3, -60.e3]]]

        burnman.SolidSolution.__init__(self, molar_fractions)

KCl = KCl_solid()
MgCl2 = MgCl2_solid()
liq = KCl_MgCl2_liquid()


XMgCl2s = np.linspace(0.0, 1.0, 101)
excess_enthalpy = np.empty_like(XMgCl2s)
for i, XMgCl2 in enumerate(XMgCl2s):
    liq.set_composition([1.-XMgCl2, XMgCl2])
    liq.set_state(1.e5, 1073.)
    excess_enthalpy[i] = liq.excess_enthalpy

plt.plot(XMgCl2s, excess_enthalpy)
plt.show()

XMgCl2s = np.linspace(0.0, 1.0, 101)
activity_KCl = np.empty_like(XMgCl2s)
T = 1073.15
liq.set_composition([1., 0.])
liq.set_state(1.e5, T)
KCl_liq_gibbs = liq.gibbs
for i, XMgCl2 in enumerate(XMgCl2s):
    liq.set_composition([1.-XMgCl2, XMgCl2])
    liq.set_state(1.e5, T)
    KCl.set_state(1.e5, T)
    activity_KCl[i] = np.exp((liq.partial_gibbs[0] - KCl_liq_gibbs)/(burnman.constants.gas_constant*T))

data = [[0.10149750416,  0.863552751809],
        [0.153078202995,  0.770151266674],
        [0.201331114809,  0.650362904352],
        [0.247920133111,  0.523376307274],
        [0.297836938436,  0.384407282826],
        [0.34775374376 , 0.257428666052],
        [0.399334442596,  0.164027180917],
        [0.500831946755,  0.0755415634215],
        [0.600665557404,  0.0302174233991],
        [0.700499168053,  0.0160683433287],
        [0.79534109817 , 0.00670345587091]]

data = zip(*data)

plt.plot(XMgCl2s, activity_KCl)
plt.plot(data[0], data[1], linestyle='None', marker='o')
plt.show()



def find_liquidus(XMgCl2, T):
    liq.set_composition([1.-XMgCl2[0], XMgCl2[0]])
    liq.set_state(1.e5, T)
    KCl.set_state(1.e5, T)

    return [liq.partial_gibbs[0] - KCl.gibbs]

temperatures = np.linspace(673.15, 1043.15, 101)
XMgCl2s = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    XMgCl2s[i] = optimize.fsolve(find_liquidus, [0.01], args=(T))[0]

data=[[0.0519262981575,  748.91598916],
      [0.0971524288107,  725.067750678],
      [0.0988274706868,  709.891598916],
      [0.15242881072 , 690.379403794],
      [0.195979899497,  640.514905149],
      [0.19932998325 , 634.010840108],
      [0.204355108878,  619.918699187],
      [0.249581239531,  560.298102981],
      [0.256281407035,  531.029810298],
      [0.284757118928,  481.165311653],
      [0.29648241206 , 462.737127371],
      [0.299832495812,  442.140921409],
      [0.308207705193,  434.552845528]]

data = zip(*data)


plt.plot(XMgCl2s, temperatures-273.15)
plt.plot(data[0], data[1], linestyle='None', marker='o')
plt.show()
