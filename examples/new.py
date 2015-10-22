import os, sys, numpy as np, matplotlib.pyplot as plt
from scipy import optimize
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
import matplotlib.pyplot as plt


fo=burnman.minerals.HP_2011_ds62.fo() # this is forsterite

fo.set_state(1.e9, 1273.15) # 1 GPa, 1000 C
print fo.V
print fo.K_T/1.e9

fa=burnman.minerals.HP_2011_ds62.fa() # this is fayalite


class olivine(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='olivine'
        self.endmembers = [[fo, '[Mg]2SiO4'], [fa, '[Fe]2SiO4']]
        self.type='symmetric'
        self.enthalpy_interaction=[[4.e3]]
        burnman.SolidSolution.__init__(self, molar_fractions)


ol=olivine()

ol.set_composition([0.8, 0.2])
ol.set_state(1.e9, 1273.15)

print ol.K_T
print ol.gibbs
print ol.excess_gibbs

compositions=np.linspace(0.0, 0.999, 101)
excess_gibbs=np.empty_like(compositions)

for i, c in enumerate(compositions):
    ol.set_composition([1-c, c])
    ol.set_state(1.e9, 1273.15)
    excess_gibbs[i]=ol.partial_gibbs[0]

plt.plot(compositions, excess_gibbs)
plt.show()
