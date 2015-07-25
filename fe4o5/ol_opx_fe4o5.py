import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

# Benchmarks for the solid solution class
import burnman
from burnman import minerals
from burnman.chemicalpotentials import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scipy.optimize as optimize

from mineral_models import *

ol = olivine()
wad = wadsleyite()
rw=ringwoodite()

fm45 = MgFeFe2O5()

en = minerals.HP_2011_ds62.en()
fs = minerals.HP_2011_ds62.fs()
fm=ordered_fm_opx()

# EMOD minerals
mag = minerals.HP_2011_ds62.mag()
fo = minerals.HP_2011_ds62.fo()
diam = minerals.HP_2011_ds62.diam()

# QFM minerals
q= minerals.HP_2011_ds62.q()
fa = minerals.HP_2011_ds62.fa()
mt = minerals.HP_2011_ds62.mt()

P=0.1e9
T=1273.

en.set_state(P,T)
fs.set_state(P,T)
fm.set_state(P,T)

print en.gibbs, fs.gibbs, fm.gibbs, 0.5*(en.gibbs + fs.gibbs) - fm.gibbs

class orthopyroxene(burnman.SolidSolution):
    def __init__(self):
        # Name
        self.name='Fe-Mg orthopyroxene'

        base_material = [[burnman.minerals.HP_2011_ds62.en(), '[Mg][Mg]Si2O6'],[burnman.minerals.HP_2011_ds62.fs(), '[Fe][Fe]Si2O6'],[ordered_fm_opx(), '[Mg][Fe]Si2O6']]

        # Interaction parameters
        enthalpy_interaction=[[6.8e3, 4.5e3],[4.5e3]]

        burnman.SolidSolution.__init__(self, base_material, \
                          burnman.solutionmodel.SymmetricRegularSolution(base_material, enthalpy_interaction) )


opx=orthopyroxene()

################################
# Check order parameter finder #
################################

def opx_composition(arg, X_Mg, P, T):
    Q=arg[0]
    opx.set_composition([X_Mg-0.5*Q, 1-X_Mg-0.5*Q, Q])
    opx.set_state(P,T)
    return 0.5*(opx.partial_gibbs[0] + opx.partial_gibbs[1]) - opx.partial_gibbs[2] 


P=1.e9
T=1473.15
X_Mgs=np.linspace(0.0,1.0,101)
G=np.empty_like(X_Mgs)
Q=np.empty_like(X_Mgs)
for idx, X_Mg in enumerate(X_Mgs):
    optimize.fsolve(opx_composition, [0.2], args=(X_Mg, P, T))
    G[idx]=opx.gibbs
    Q[idx]=opx.molar_fractions[2]

plt.plot( X_Mgs, G, '-', linewidth=3., label='Gibbs opx')

plt.title('')
plt.xlabel("X_Mg opx")
plt.ylabel("G")
plt.legend(loc='lower right')
plt.show()


#################################
# Find ol-opx-Fe4O5 equilibrium #
#################################

# Constraints
# ol composition

# Unknowns
# opx order parameter
# opx composition
# Fe4O5 composition

# Reactions
# opx (order parameter)
# Fe-Mg exchange (ol-opx)
# Fe-Mg exchange (ol-Fe4O5)

def ol_opx_fm45_equilibrium(args, X_Mg_ol, P, T):
    Q, X_Mg_opx, X_Mg_Fe4O5 = args
    ol.set_composition([X_Mg_ol, 1.-X_Mg_ol])
    opx.set_composition([X_Mg_opx-0.5*Q, 1-X_Mg_opx-0.5*Q, Q])
    fm45.set_composition([X_Mg_Fe4O5, 1.-X_Mg_Fe4O5])

    ol.set_state(P, T)
    opx.set_state(P, T)
    fm45.set_state(P, T)

    equations = [ 0.5*(opx.partial_gibbs[0] + opx.partial_gibbs[1]) - opx.partial_gibbs[2],
                  ( ol.partial_gibbs[0] + opx.partial_gibbs[1] ) - ( ol.partial_gibbs[1] + opx.partial_gibbs[0] ), 
                  ( ol.partial_gibbs[0] + 2.*fm45.partial_gibbs[1] ) - ( ol.partial_gibbs[1] + 2.*fm45.partial_gibbs[0] ) ]
    
    return equations


# Define EMOD oxygen buffer
# enstatite, magnesite, olivine, diamond
EMOD = [en, mag, fo, diam]
QFM = [q, fa, mt]

O2=minerals.HP_2011_fluids.O2()
O2.set_method('cork')

pressures = np.linspace(8.e9, 16.e9, 9)
fO2_ol_opx_fm45 = np.empty_like(pressures)
fO2_EMOD = np.empty_like(pressures)
fO2_QFM = np.empty_like(pressures)

T = 1673.15
O2.set_state(1.e5,T)
X_Mg_ol = 0.9

for i, P in enumerate(pressures):
    print optimize.fsolve(ol_opx_fm45_equilibrium, [0.0, 0.94, 0.7], args=(X_Mg_ol, P, T))
    fO2_ol_opx_fm45[i] =  np.log10(fugacity(O2, [ol, opx, fm45]))

    for mineral in EMOD:
        mineral.set_state(P, T)
    for mineral in QFM:
        mineral.set_state(P, T)

    fO2_EMOD[i] = np.log10(fugacity(O2, EMOD))
    fO2_QFM[i] = np.log10(fugacity(O2, QFM))



plt.plot(pressures/1.e9, fO2_ol_opx_fm45, label='ol-opx-Fe4O5')
plt.plot(pressures/1.e9, fO2_EMOD, label='EMOD')
plt.plot(pressures/1.e9, fO2_QFM, label='QFM')

plt.legend(loc='lower left')
plt.show()
