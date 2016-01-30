# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

# Tallon (1980) suggested that melting of simple substances was associated with an entropy change of
# Sfusion = burnman.constants.gas_constant*np.log(2.) + a*K_T*Vfusion
# Realising also that dT/dP = Vfusion/Sfusion, we can express the entropy 
# and volume of fusion in terms of the melting curve:
# Sfusion = burnman.constants.gas_constant*np.log(2.) / (1. - a*K_T*dTdP)
# Vfusion = Sfusion*dT/dP

from scipy.interpolate import interp1d
from scipy.optimize import fsolve, curve_fit, minimize
import burnman
from burnman import minerals
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()

dummy = burnman.minerals.HP_2011_ds62.py

class ternary_melt(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='ternary melt'
        self.type='symmetric'
        self.endmembers = [[dummy(), '[O]'],[dummy(), '[Fe]'],[dummy(), '[S]']]
        self.enthalpy_interaction=[[40.e3, 0.e3],[0.e3]]

        burnman.SolidSolution.__init__(self, molar_fractions)

'''
class ternary_melt(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='ternary melt'
        self.type='subregular'
        self.endmembers = [[dummy(), '[O]'],[dummy(), '[Fe]'],[dummy(), '[S]']]
        self.enthalpy_interaction=[[[50.e3, 50.e3], [0.e3, 0.e3]],[[0.e3, 0.e3]]]

        burnman.SolidSolution.__init__(self, molar_fractions)
'''        

if __name__ == "__main__":

    def find_compositions(args, X20, P, T, melt0, melt1):

        X00, X01, X21 = args
        
        c0 = [X00, 1. - X00 - X20, X20]
        c1 = [X01, 1. - X01 - X21, X21]

        melt0.set_composition(c0)
        melt1.set_composition(c1)

        melt0.set_state(P, T)
        melt1.set_state(P, T)

        activities0 = np.exp(melt0.excess_partial_gibbs/(burnman.constants.gas_constant*T))
        activities1 = np.exp(melt1.excess_partial_gibbs/(burnman.constants.gas_constant*T))
        return [activities0[0] - activities1[0],
                activities0[1] - activities1[1],
                activities0[2] - activities1[2]]
                          
    
    melt0 = ternary_melt()
    melt1 = ternary_melt()

    P = 1.e9
    T = 2000.

    Xs = np.linspace(0.0, 0.4, 101)
    X00_compositions = []
    X01_compositions = []
    X20_compositions = []
    X21_compositions = []

    
    for i, X in enumerate(Xs):
        
        #There should be a big immiscibility gap between component 0 and 1
        X20 = X
        soln = fsolve(find_compositions, [0.01, 0.99, X20], args=(X20, P, T, melt0, melt1), full_output=1)
        
        if soln[2]==1:
            X00, X01, X21 = soln[0]
            if np.abs(X01-X00) > 1.e-2:
                X00_compositions.append(X00)
                X01_compositions.append(X01)
                X20_compositions.append(X20)
                X21_compositions.append(X21)
                print X00, X01, X20, X21



    X00_compositions = np.array(X00_compositions)
    X01_compositions = np.array(X01_compositions)
    X20_compositions = np.array(X20_compositions)
    X21_compositions = np.array(X21_compositions)


    plt.plot(X00_compositions + 0.5*X20_compositions, X20_compositions, marker='o', linestyle='None')
    plt.plot(X01_compositions + 0.5*X21_compositions, X21_compositions, marker='o', linestyle='None')
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)

    plt.show()
    

    gibbs_0 = np.empty_like(X20_compositions)
    gibbs_1 = np.empty_like(X20_compositions)
    gibbs_2 = np.empty_like(X20_compositions)
    for i, X in enumerate(X20_compositions):
        Z = 0.0
        melt0.set_composition([X*(1. - Z), (1. - X)*(1. - Z), Z])
        melt0.set_state(P, T)
        gibbs_0[i] = np.exp(melt0.excess_partial_gibbs[0]/(burnman.constants.gas_constant*T))
        gibbs_1[i] = np.exp(melt0.excess_partial_gibbs[1]/(burnman.constants.gas_constant*T))
        gibbs_2[i] = np.exp(melt0.excess_partial_gibbs[2]/(burnman.constants.gas_constant*T))

    plt.plot(X20_compositions, gibbs_0)
    plt.plot(X20_compositions, gibbs_1)
    plt.plot(X20_compositions, gibbs_2)
    plt.show()
