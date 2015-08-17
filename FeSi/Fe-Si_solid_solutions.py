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

from burnman.solidsolution import SolidSolution
from burnman.solutionmodel import *
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()


'''
These models have no order-disorder

fcc and hcp are teated as fully disordered Fe-Si solid solutions
B2 is treated as fully ordered Fe[Fe] - Fe[Si]

'''

class hcp_Fe_Si(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='HCP Fe-Si solid solution'
        self.type='subregular'
        self.endmembers = [[minerals.Myhill_calibration_iron.hcp_iron(), '[Fe]'],[minerals.Fe_Si_O.Si_hcp_A3(), '[Si]']]
        self.enthalpy_interaction=[[[-50.e3, -50.e3]]]
        self.entropy_interaction=[[[0.e3, 0.e3]]]
        self.volume_interaction=[[[3.11e-7, 3.11e-7]]]
        burnman.SolidSolution.__init__(self, molar_fractions)

class fcc_Fe_Si(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='HCP Fe-Si solid solution'
        self.type='subregular'
        self.endmembers = [[minerals.Myhill_calibration_iron.fcc_iron(), '[Fe]'],[minerals.Fe_Si_O.Si_fcc_A1(), '[Si]']]
        self.enthalpy_interaction=[[[-50.e3, -50.e3]]]
        self.entropy_interaction=[[[0.e3, 0.e3]]]
        self.volume_interaction=[[[3.11e-7, 3.11e-7]]]
        burnman.SolidSolution.__init__(self, molar_fractions)


class B2_Fe_FeSi(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='HCP Fe-Si solid solution'
        self.type='subregular'
        self.endmembers = [[minerals.Myhill_calibration_iron.bcc_iron(), 'Fe0.5[Fe]0.5'],
                           [minerals.Fe_Si_O.FeSi_B2(), 'Fe0.5[Si]0.5']]
        self.enthalpy_interaction=[[[-50.e3, -50.e3]]]
        self.entropy_interaction=[[[0.e3, 0.e3]]]
        self.volume_interaction=[[[3.11e-7, 3.11e-7]]]
        burnman.SolidSolution.__init__(self, molar_fractions)


hcp=hcp_Fe_Si()
fcc=fcc_Fe_Si()
B2=B2_Fe_FeSi()


def fcc_hcp_eqm(data, P, T):
    Xfcc=data[0]
    Xhcp=data[1]
    fcc.set_composition([1.-Xfcc, Xfcc])
    fcc.set_state(P, T)
    hcp.set_composition([1.-Xhcp, Xhcp])
    hcp.set_state(P, T)
    return [fcc.partial_gibbs[0]-hcp.partial_gibbs[0],
            fcc.partial_gibbs[1]-hcp.partial_gibbs[1]]


def iron_B2_eqm(data, P, T, phase):
    Xphase=data[0]
    XB2=data[1] # proportion of Si in B2 structure. Must be 0 - 0.5
    phase.set_composition([1.-Xphase, Xphase])
    phase.set_state(P, T)
    B2.set_composition([1.-(XB2*2.), (XB2*2.)])
    B2.set_state(P, T)
    return [phase.partial_gibbs[0] - B2.partial_gibbs[0]*2., 
            phase.partial_gibbs[0] + phase.partial_gibbs[1] - B2.partial_gibbs[1]*4.] 


P=25.e9
temperatures=np.linspace(1000., 3000., 50)
temperatures_fcc_hcp=[]
X_fcc=[]
X_hcp=[]
for T in temperatures:
    solution=optimize.fsolve(fcc_hcp_eqm, [0.99, 0.01], args=(P, T), full_output=True)
    if solution[2]==1 and solution[0][1] < 1.0 and solution[0][0] > 0.0 :
        temperatures_fcc_hcp.append(T)
        X_fcc.append(solution[0][0])
        X_hcp.append(solution[0][1])

        
plt.plot( X_fcc, temperatures_fcc_hcp, linewidth=1, label='fcc')
plt.plot( X_hcp, temperatures_fcc_hcp, linewidth=1, label='hcp')


temperatures_iron_B2=[]
X_iron=[]
X_B2=[]
for T in temperatures:
    solution=optimize.fsolve(iron_B2_eqm, [0.01, 0.49], args=(P, T, hcp), full_output=True)
    if solution[2]==1 and solution[0][1] < 0.5 and solution[0][0] > 0.0 :
        temperatures_iron_B2.append(T)
        X_iron.append(solution[0][0])
        X_B2.append(solution[0][1])


plt.plot( X_iron, temperatures_iron_B2, '--', linewidth=1, label='iron')
plt.plot( X_B2, temperatures_iron_B2, '--', linewidth=1, label='B2')


temperatures_iron_B2=[]
X_iron=[]
X_B2=[]
for T in temperatures:
    solution=optimize.fsolve(iron_B2_eqm, [0.01, 0.49], args=(P, T, fcc), full_output=True)
    if solution[2]==1 and solution[0][1] < 0.5 and solution[0][0] > 0.0 :
        temperatures_iron_B2.append(T)
        X_iron.append(solution[0][0])
        X_B2.append(solution[0][1])


plt.plot( X_iron, temperatures_iron_B2, '.', linewidth=1, label='iron')
plt.plot( X_B2, temperatures_iron_B2, '.', linewidth=1, label='B2')
        
plt.title('Two phase region')
plt.xlabel("Composition")
plt.ylabel("Temperature (K)")
plt.legend(loc='upper right')
plt.xlim(0., 0.5)
plt.show()
