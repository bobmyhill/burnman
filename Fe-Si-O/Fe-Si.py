# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman import tools
from burnman.mineral import Mineral
from burnman.minerals import Myhill_calibration_iron
from burnman.processchemistry import *
from burnman.chemicalpotentials import *
from burnman import constants

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import optimize
atomic_masses=read_masses()

Pr=1.e5
nA=6.02214e23
voltoa=1.e30

Z_B2=1. # Fm-3m
Z_B20=4. # P2_13

basicerror=0.01 # Angstroms
FeSi_B20_data=[]
FeSi_B2_data=[]
for line in open('data/Fischer_et_al_FeSi_PVT_S2.dat'):
    content=line.strip().split()
    if content[0] != '%' and content[8] != '*' and content[8] != '**':
        if content[7] != '-': # T_B20, Terr_B20, P_B20, Perr_B20, aKbr_B20, aKbrerr_B20, a_B20, a_err_B20
            if float(content[8]) < 1.e-12:
                content[8]=basicerror
            FeSi_B20_data.append([float(content[1]), float(content[2]), float(content[3])*1.e9, float(content[4])*1.e9, float(content[5]), float(content[6]), float(content[7]), float(content[8])])
        if content[9] != '-': # T_B2, Terr_B2, P_B2, Perr_B2, aKbr_B2, aKbrerr_B2, a_B2, a_err_B2
            if float(content[10]) < 1.e-12:
                content[10]=basicerror
            FeSi_B2_data.append([float(content[1]), float(content[2]), float(content[3])*1.e9, float(content[4])*1.e9, float(content[5]), float(content[6]), float(content[9]), float(content[10])])


T_B20, Terr_B20, P_B20, Perr_B20, aKbr_B20, aKbrerr_B20, a_B20, a_err_B20 = zip(*FeSi_B20_data)
T_B2, Terr_B2, P_B2, Perr_B2, aKbr_B2, aKbrerr_B2, a_B2, a_err_B2 = zip(*FeSi_B2_data)

class FeSi_B20 (Mineral):
    def __init__(self):
        formula='Fe1.0Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FeSi B20',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 5050. ,
            'S_0': 29.90 ,
            'V_0': 1.359e-05 ,
            'Cp': [0.0, 0.0, 0.0, 0.0],
            'a_0': 2.811e-05 ,
            'K_0': 2.057e+11 ,
            'Kprime_0': 4.0 ,
            'Kdprime_0': -4.0/2.057e+11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class FeSi_B2 (Mineral):
    def __init__(self):
        formula='Fe1.0Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FeSi B2',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 5050. ,
            'S_0': 29.90 ,
            'V_0': 1.300e-05 ,
            'Cp': [0.0, 0.0, 0.0, 0.0],
            'a_0': 3.064e-05 ,
            'K_0': 2.199e+11 ,
            'Kprime_0': 4.0 ,
            'Kdprime_0': -4.0/2.199e+11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

B20=FeSi_B20()
B2=FeSi_B2()

def fit_PVT_data(mineral):
    def fit_data(PT, V_0, K_0, a_0):
        mineral.params['V_0'] = V_0
        mineral.params['K_0'] = K_0
        Kprime_0 = 4.0
        mineral.params['Kprime_0'] = Kprime_0
        mineral.params['Kdprime_0'] = -Kprime_0/K_0
        mineral.params['a_0'] = a_0

        volumes=[]
        for P, T in zip(*PT):
            mineral.set_state(P, T)
            volumes.append(mineral.V)

        return volumes
    return fit_data


'''
B20 fitting
'''

# Pressures and Temperatures
PT_B20=np.array([P_B20,T_B20])

a_B20=np.array(a_B20)
a_err_B20=np.array(a_err_B20)

# Volumes and uncertainties
V_B20=a_B20*a_B20*a_B20*(nA/Z_B20/voltoa)
Verr_B20=3.*a_B20*a_B20*a_err_B20*(nA/Z_B20/voltoa)

# Guesses
guesses=[B20.params['V_0'], B20.params['K_0'], B20.params['a_0']]

popt, pcov = optimize.curve_fit(fit_PVT_data(B20), PT_B20, V_B20, guesses, Verr_B20)
print 'B20 params:', popt


'''
B2 fitting
'''

# Pressures and Temperatures
PT_B2=np.array([P_B2,T_B2])

a_B2=np.array(a_B2)
a_err_B2=np.array(a_err_B2)

# Volumes and uncertainties
V_B2=a_B2*a_B2*a_B2*(nA/Z_B2/voltoa)
Verr_B2=3.*a_B2*a_B2*a_err_B2*(nA/Z_B2/voltoa)

# Guesses
guesses=[B2.params['V_0'], B2.params['K_0'], B2.params['a_0']]

popt, pcov = optimize.curve_fit(fit_PVT_data(B2), PT_B2, V_B2, guesses, Verr_B2)
print 'B2 params:', popt


# PLOTTING

pressures=[]
volumes_calculated_B20=[]
volumes_observed_B20=[]

for i, PT in enumerate(zip(*PT_B20)):
    P, T = PT
    B20.set_state(P, T)
    if (T-300.)*(T-300.) < 0.1:
        pressures.append(P)
        volumes_calculated_B20.append(B20.V)
        volumes_observed_B20.append(V_B20[i])

pressures_B2=np.linspace(10.e9, 100.e9, 101)
volumes_calculated_B2=np.empty_like(pressures_B2)

T=300.
for i, P in enumerate(pressures_B2):
    B2.set_state(P, T)
    pressures_B2[i] = P
    volumes_calculated_B2[i] = B2.V

plt.plot( np.array(pressures)/1.e9, volumes_calculated_B20, linewidth=1)
plt.plot( np.array(pressures)/1.e9, volumes_observed_B20, marker=".", linestyle="None")
plt.plot( pressures_B2/1.e9, volumes_calculated_B2, linewidth=1)

plt.title('Volume fit')
plt.xlabel("Pressure (GPa)")
plt.ylabel("Volume (m^3/mol)")
plt.show()



