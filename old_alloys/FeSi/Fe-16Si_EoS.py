# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman import tools
from burnman.mineral import Mineral
from burnman.chemicalpotentials import *
from burnman import constants

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from fitting_functions import *
from scipy import optimize

atomic_masses=read_masses()


'''
This script takes the 16 wt% B2 data from Fischer et al., 2014 and the EoSs from B2 Fe and FeSi to estimate a high pressure dV for mixing
'''

class dummy (Mineral):
    def __init__(self):
        formula='Fe0.72530Si0.27470'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Fe-16Si (BCC)',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 49200., # SGTE data
            'S_0': 18.820 + 20.8, # Barin, SGTE data
            'V_0': 7.0e-06 , # hcp Fe is 6.766e-06, Si is 8.8e-6
            'Cp': [22.826, 0.003856857, -353888.416, -0.0596068], # Barin
            'a_0': 7.757256e-06 ,
            'K_0': 57.44e9 , # 72 = Duclos et al 
            'Kprime_0': 5.28 , # Fit to Mujica et al. # 3.9 for Duclos et al 
            'Kdprime_0': -5.28/57.44e9 , # Duclos et al 
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)


'''
First, we import the minerals we will use 
'''

FeSi_B2=minerals.Fe_Si_O.FeSi_B2()
Fe_bcc=minerals.Myhill_calibration_iron.bcc_iron()
Fe16Si_B2=dummy()

alloy=16. # wt% Si
mass_Fe=formula_mass({'Fe': 1.}, atomic_masses)
mass_Si=formula_mass({'Si': 1.}, atomic_masses)
molar_fraction_Si=(alloy/mass_Si)/((100.-alloy)/mass_Fe + alloy/mass_Si)
composition=[1.-molar_fraction_Si, molar_fraction_Si]

# convert composition into fractions of Fe and (FeSi)0.5
composition = [composition[0]-(composition[1]), 2.*composition[1]]
print composition



'''
Here are some important constants
'''
Pr=1.e5
nA=6.02214e23
voltoa=1.e30
Z_hcp=2. # HCP, P6_3/mmc

'''
Now we read in Fischer et al. PVT data for the hcp Fe-9Si
Remember that for hcp, V = np.sqrt(3)/2.*a*a*a*(c/a)
'''

# 0: assemblage
# 1: P (GPa) 2: +/-
# 3: T up, meas (K)* 4: T down, meas (K)* 5: T average, meas (K)*	
# 6: T sample (K)* 7: +/-	8: T KBr (K)*	9: +/-	10: a KBr (A)	11: +/-	
# 12: V D03 Fe-Si (cm3/mol)	13: +/-	14: V B2 Fe-Si (cm3/mol) 15: +/-	
# 16: V hcp Fe-Si (cm3/mol)	17: +/-	18: a hcp Fe-Si	19: +/-	
# 20: c hcp Fe-Si	21: +/-	22: c/a hcp Fe-Si	23: +/-
Fe16Si_B2_data=[]
basicerror=0.05
for line in open('data/Fischer_et_al_Fe16Si_PVT_S2.dat'):
    content=line.strip().split()
    if content[0] == 'B2': # T_B2, Terr_B2, P_B2, Perr_B2, aKbr_B2, aKbrerr_B2, a_B2, a_err_B2, covera_B2, covera_err_B2
        Fe16Si_B2_data.append([float(content[1])*1.e9, float(content[2])*1.e9, 
                               float(content[6]), float(content[7]), 
                               float(content[14])*1.e-6, float(content[15])*1.e-6])


P_B2, Perr_B2, T_B2, Terr_B2, V_B2, Verr_B2 = zip(*Fe16Si_B2_data)
P_B2 = np.array(P_B2)
T_B2 = np.array(T_B2)
PT_B2 = [P_B2, T_B2]
print V_B2

'''
B2 PVT fitting
'''

def fit_excess_V(data, Vex):
    V = []

    for datum in data:
        P, T = datum
        FeSi_B2.set_state(P, T)
        Fe_bcc.set_state(P, T)
        V.append(Fe_bcc.V*composition[0] + FeSi_B2.V*composition[1] + Vex*composition[0]*composition[1])
    return V

guesses=[0.0]
popt, pcov = optimize.curve_fit(fit_excess_V, zip(*PT_B2), V_B2, guesses, Verr_B2)

Vex_B2=popt[0]
Vex_B2_err=np.sqrt(pcov[0][0])
print Vex_B2, '+/-', Vex_B2_err

diff=np.empty_like(V_B2)
for i, P in enumerate(P_B2):
    T=T_B2[i]
    Fe16Si_B2.set_state(P, T)
    FeSi_B2.set_state(P, T)
    Fe_bcc.set_state(P, T)
    print 'P, T, ideal-obs, obserr', P/1.e9, T, Fe_bcc.V*composition[0] + FeSi_B2.V*composition[1] - V_B2[i], Verr_B2[i]
    diff[i]=(Fe_bcc.V*composition[0] + FeSi_B2.V*composition[1] + Vex_B2*composition[0]*composition[1]) - V_B2[i]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(P_B2/1.e9, T_B2, diff, c='r', marker='o')

ax.set_xlabel('P (GPa)')
ax.set_ylabel('T (K)')
ax.set_zlabel('Volume difference (m^3/mol)')

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(P_B2/1.e9, T_B2, diff/Verr_B2, c='r', marker='o')

ax.set_xlabel('P (GPa)')
ax.set_ylabel('T (K)')
ax.set_zlabel('(Model-Obs)/uncertainty')

plt.show()
