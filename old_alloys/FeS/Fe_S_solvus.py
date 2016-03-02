# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
from burnman.chemicalpotentials import *
from make_intermediate import *
from HP_convert import *
atomic_masses=read_masses()

# Here's a little mockup of a solid solution model for Fe-FeSi-FeS
Fe_liq=burnman.minerals.Myhill_calibration_iron.liquid_iron()
FeSi_liq = burnman.minerals.Fe_Si_O.FeSi_liquid()
FeS_liq = burnman.minerals.Fe_Si_O.FeSi_liquid() # THIS NEEDS CHANGING


# H_ex, S_ex, Sconf, V_ex, K_ex, a_ex
ideal_excesses = [0., 0., -burnman.constants.gas_constant*np.log(0.5), 0., 0., 0.]
FeFeSi_liq = make_intermediate(Fe_liq, FeSi_liq, ideal_excesses)
FeFeS_liq = make_intermediate(Fe_liq, FeS_liq, ideal_excesses)



arb_excesses = [25.e3, 0., -burnman.constants.gas_constant*np.log(0.5), -0.6e-6, 0., 0.]
FeSiFeS_liq = make_intermediate(FeSi_liq, FeS_liq, arb_excesses)

HP_convert(Fe_liq, 1809., 2400., 1809., 50.e9)

class Fe_Si_S_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='metallic liquid Fe-Si-O solution'
        self.type='full_subregular'
        self.endmembers = [[Fe_liq, '[Fe]'], [FeSi_liq, 'Fe[Si]'], [FeS_liq, 'Fe[S]']]
        self.intermediates = [[[FeFeSi_liq, FeFeSi_liq],
                               [FeFeS_liq,  FeFeS_liq]],
                              [[FeSiFeS_liq,  FeSiFeS_liq]]]

        burnman.SolidSolution.__init__(self, molar_fractions)




S_rich_liq = Fe_Si_S_liquid()
Si_rich_liq = Fe_Si_S_liquid()


P = 17.e9
T = 2000. 


FeFeS_liq.set_state(P, T)
print FeFeS_liq.gibbs

X_binary = np.linspace(0., 1.0, 101)
excess_gibbs = np.empty_like(X_binary)

X_Fes = np.linspace(0.0, 0.5, 2)
for X_Fe in X_Fes:
    for i, X in enumerate(X_binary):
        X_FeSi = X*(1. - X_Fe)
        Si_rich_liq.set_composition([X_Fe, X_FeSi, 1. - X_Fe - X_FeSi])
        Si_rich_liq.set_state(P, T)
        excess_gibbs[i] = Si_rich_liq.excess_gibbs
        
    plt.plot(X_binary, excess_gibbs)
plt.show()

def find_S_rich_liquid(args, X_FeSi, P, T):
    X_FeS, X_FeSi_S_rich, X_FeS_S_rich = args
    
    Si_rich_liq.set_composition([1. - X_FeSi - X_FeS, X_FeSi, X_FeS])
    S_rich_liq.set_composition([1. - X_FeSi_S_rich - X_FeS_S_rich, X_FeSi_S_rich, X_FeS_S_rich])
    
    Si_rich_liq.set_state(P, T)
    S_rich_liq.set_state(P, T)


    return [Si_rich_liq.partial_gibbs[0] - S_rich_liq.partial_gibbs[0],
            Si_rich_liq.partial_gibbs[1] - S_rich_liq.partial_gibbs[1],
            Si_rich_liq.partial_gibbs[2] - S_rich_liq.partial_gibbs[2]]

X_FeSis = np.linspace(1.0, 0.2, 21)
X_FeSs = np.empty_like(X_FeSis)
X_FeSis_S_rich = np.empty_like(X_FeSis)
X_FeSs_S_rich = np.empty_like(X_FeSis)


plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)

T = 2200.
pressures = np.linspace(10.e9, 18.e9, 5)
for P in pressures:
    print P/1.e9, 'GPa'
    guesses = [0.01, 0.01, 0.99]
    for i, X_FeSi in enumerate(X_FeSis):
        fractions=optimize.fsolve(find_S_rich_liquid, guesses, args = (X_FeSi, P, T))
        guesses = fractions
        X_FeSs[i], X_FeSis_S_rich[i], X_FeSs_S_rich[i] = fractions


    plt.plot(X_FeSis, X_FeSs)
    plt.plot(X_FeSis_S_rich, X_FeSs_S_rich)

plt.show()
