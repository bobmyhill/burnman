# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman.mineral import Mineral
from burnman.processchemistry import *
from burnman.chemicalpotentials import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import optimize

atomic_masses=read_masses()


class iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'HP_2011_ds62 iron (BCC, sort of)',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -0.0 ,
            'S_0': 27.09 ,
            'V_0': 7.09e-06 ,
            'Cp': [46.2, 0.005159, 723100.0, -556.2] ,
            'a_0': 3.56e-05 ,
            'K_0': 1.64e+11 ,
            'Kprime_0': 5.16 ,
            'Kdprime_0': -3.1e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses),
            'landau_Tc': 1042.0 ,
            'landau_Smax': 8.3 ,
            'landau_Vmax': 0.0 }
        Mineral.__init__(self)


class hcp_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'HCP iron (RM 12/2014)',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 5050. ,
            'S_0': 29.90 ,
            'V_0': 6.733e-06 ,
            'Cp': [52.2754, -0.000355, 790700.0, -619.1] ,
            'a_0': 4.29e-05 ,
            'K_0': 1.85827e+11 ,
            'Kprime_0': 4.37 ,
            'Kdprime_0': -2.35e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class fcc_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FCC iron (RM 12/2014)',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 7840. ,
            'S_0': 35.85 ,
            'V_0': 6.938e-06 ,
            'Cp': [52.2754, -0.000355, 790700.0, -619.1] ,
            'a_0': 5.13e-05 ,
            'K_0': 1.532e+11 ,
            'Kprime_0': 5.3 ,
            'Kdprime_0': -2.70e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

hcp=hcp_iron()
fcc=fcc_iron()


def fcc_hcp_eqm(arg, T):
    P=arg[0]
    hcp.set_state(P,T)
    fcc.set_state(P,T)
    return [hcp.gibbs - fcc.gibbs]


temperatures=np.linspace(400.,2400.,101)
hcp_fcc_pressures=np.empty_like(temperatures)

for idx, T in enumerate(temperatures):
    hcp_fcc_pressures[idx]=optimize.fsolve(fcc_hcp_eqm, 1.e9, args=(T))[0]

'''
f=open('Fe-O_boundaries_int.dat', 'r')
datastream = f.read()  # We need to re-open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
phase_boundaries=np.array(datalines, np.float32).T
phase_boundaries[0]=[1.0-x_to_y(phase_boundaries[0][i]) for i in range(len(phase_boundaries[0]))]
'''

plt.plot( hcp_fcc_pressures/1.e9, temperatures, 'r-', linewidth=3., label='HCP-FCC')


plt.title('Iron phase diagram')
plt.ylabel("Temperature (K)")
plt.xlabel("Pressure (GPa)")
plt.legend(loc='lower right')
plt.show()

