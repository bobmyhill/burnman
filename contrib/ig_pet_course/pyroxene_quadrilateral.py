import numpy as np
import matplotlib.pyplot as plt
from burnman.minerals import HP_2011_ds62
from burnman.minerals import JH_2015
from burnman import SolidSolution, Composite, equilibrate


class orthopyroxene(SolidSolution):

    def __init__(self, molar_fractions=None):
        self.name = 'orthopyroxene (CFMS)'
        self.endmembers = [[HP_2011_ds62.en(),   '[Mg][Mg][Si]0.5Si1.5O6'],
                           [JH_2015.odi(),  '[Mg][Ca][Si]0.5Si1.5O6']] 
        self.solution_type = 'asymmetric'
        self.alphas = [1., 1.2]
        self.energy_interaction = [[32.2e3]]
        self.volume_interaction = [[0.12e-5]]

        SolidSolution.__init__(self, molar_fractions=molar_fractions)



class clinopyroxene(SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name = 'clinopyroxene (CFMS)'
        self.endmembers = [[HP_2011_ds62.di(),   '[Mg][Ca][Si]1/2O6'],
                           [JH_2015.cen(),  '[Mg][Mg][Si]1/2O6']]
        self.solution_type = 'asymmetric'
        self.alphas = [1.2, 1.0]
        self.energy_interaction = [[29.8e3]]
        self.volume_interaction = [[-0.03e-5]]
        SolidSolution.__init__(self, molar_fractions=molar_fractions)
        

P = 0.1e9

cpx = clinopyroxene()
pig = clinopyroxene()
opx = orthopyroxene()


xMgO = 1.8
xCaO = 0.2
xSiO2 = 2.


composition = {'Mg': xMgO, 'Ca': xCaO, 'Si': xSiO2, 'O': 6.}
assemblage = Composite([cpx, opx, pig])

cpx.set_composition([0.99, 0.01])  # di, cen
pig.set_composition([0.01, 0.99])  # di, cen
opx.set_composition([0.99, 0.01])  # en, odi

equality_constraints = [['P', P],
                        ['phase_fraction', (pig, np.array([0.]))]]

sol, prm = equilibrate(composition, assemblage, equality_constraints, verbose=True)
Tpig = sol.assemblage.temperature


temperatures = np.linspace(500., Tpig, 101)

xMgO = 1.57
xCaO = 0.43
xSiO2 = 2.


composition = {'Mg': xMgO, 'Ca': xCaO, 'Si': xSiO2, 'O': 6.}
assemblage = Composite([cpx, opx])

cpx.set_composition([0.99, 0.01])  # di, cen
opx.set_composition([0.99, 0.01])  # en, odi

equality_constraints = [['P', P],
                        ['T', temperatures]]

sols, prm = equilibrate(composition, assemblage, equality_constraints, verbose=True)
xs = np.array([[sol.assemblage.temperature,
                sol.assemblage.phases[0].molar_fractions[0],
                sol.assemblage.phases[1].molar_fractions[1]]
               for sol in sols if sol.success]).T
plt.plot(xs[1], xs[0])
plt.plot(xs[2], xs[0])


temperatures = np.linspace(Tpig, Tpig+250., 41)

assemblage = Composite([cpx, pig])

cpx.set_composition([0.99, 0.01])  # di, cen
pig.set_composition([0.01, 0.99])  # en, odi

equality_constraints = [['P', P],
                        ['T', temperatures]]

sols, prm = equilibrate(composition, assemblage, equality_constraints, verbose=True)
xs = np.array([[sol.assemblage.temperature,
                sol.assemblage.phases[0].molar_fractions[0],
                sol.assemblage.phases[1].molar_fractions[0]]
               for sol in sols if sol.success]).T
plt.plot(xs[1], xs[0], color='k')
plt.plot(xs[2], xs[0], color='k')


plt.show()
exit()


class orthopyroxene(SolidSolution):

    def __init__(self, molar_fractions=None):
        self.name = 'orthopyroxene (CFMS)'
        self.endmembers = [[HP_2011_ds62.en(),   '[Mg][Mg][Si]0.5Si1.5O6'],
                           [HP_2011_ds62.fs(),   '[Fe][Fe][Si]0.5Si1.5O6'],
                           [JH_2015.fm(),   '[Fe][Mg][Si]0.5Si1.5O6'],
                           [JH_2015.odi(),  '[Mg][Ca][Si]0.5Si1.5O6']] # fm ordered phase, fake T-site multiplicity
        self.solution_type = 'asymmetric'
        self.alphas = [1., 1., 1., 1.2]
        self.energy_interaction = [[5.2e3, 4.e3, 32.2e3],
                                   [4.e3, 24.e3],
                                   [18.e3]]
        self.volume_interaction = [[0., 0., 0.12e-5],
                                   [0., 0.],
                                   [0.]]

        SolidSolution.__init__(self, molar_fractions=molar_fractions)



class clinopyroxene(SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name = 'clinopyroxene (CFMS)'
        self.endmembers = [[HP_2011_ds62.di(),   '[Mg][Ca][Si]1/2O6'],
                           [JH_2015.cfs(),  '[Fe][Fe][Si]1/2O6'],
                           [JH_2015.cen(),  '[Mg][Mg][Si]1/2O6'],
                           [JH_2015.cfm(),  '[Mg][Fe][Si]1/2O6']] # note cfm ordered endmember
        self.solution_type = 'asymmetric'
        self.alphas = [1.2, 1.0, 1.0, 1.0]
        self.energy_interaction = [[20.e3, 29.8e3, 18.e3],
                                   [7.e3, 4.e3],
                                   [4.e3]]
        self.volume_interaction = [[0., -0.03e-5, 0.],
                                   [0., 0.],
                                   [0.]]
        SolidSolution.__init__(self, molar_fractions=molar_fractions)

print('NOT WORKING YET')
cpx = clinopyroxene()
opx = orthopyroxene()
ol = JH_2015.olivine()
qtz = HP_2011_ds62.q()

P = 0.1e9
T = 1073.15

Mgnum = 0.1
xMgO = (Mgnum)*1.5
xFeO = (1. - Mgnum)*1.5
xCaO = 0.5
xSiO2 = 2.


composition = {'Mg': xMgO, 'Fe': xFeO, 'Ca': xCaO, 'Si': xSiO2, 'O': 6.}
print(composition)
assemblage = Composite([cpx, opx])

cpx.set_composition([0.6, 0.3, 0.00, 0.1]) # di (MgCa), cfs, cen, cfm (FeMg)
opx.set_composition([0.40, 0.18, 0.40, 0.02]) # en, fs, fm, odi
assemblage.set_state(P, T)
equality_constraints = [['P', P],
                        ['T', T]]

sol, prm = equilibrate(composition, assemblage, equality_constraints, verbose=True)
print(sol.assemblage)
print(sol.assemblage.phases[0].partial_gibbs)
print(sol.assemblage.phases[1].partial_gibbs)