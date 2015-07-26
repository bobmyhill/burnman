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
# ol+opx composition
# 1.4 (Mg+Fe):Si (i.e. 4 Mg2SiO4 + 3 Mg2Si2O6)


# Unknowns
# opx order parameter
# opx composition
# Fe4O5 composition

# Reactions
# opx (order parameter)
# Fe-Mg exchange (ol-opx)
# Fe-Mg exchange (ol-Fe4O5)

p_ol_polymorphs = 4./7.
p_opx = 3./7.
def ol_opx_fm45_equilibrium(args, X_Mg_olopx, P, T):
    Q, X_Mg_opx, X_Mg_Fe4O5 = args

    # X_Mg_olopx = 
    # (p_ol*Mg_ol + p_opx*Mg_opx) / (p_ol + p_opx)
    X_Mg_ol = (X_Mg_olopx * (p_ol_polymorphs + p_opx) - p_opx*X_Mg_opx)/p_ol_polymorphs
    
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


def wad_opx_fm45_equilibrium(args, X_Mg_wadopx, P, T):
    Q, X_Mg_opx, X_Mg_Fe4O5 = args

    # X_Mg_olopx = 
    # (p_ol*Mg_ol + p_opx*Mg_opx) / (p_ol + p_opx)
    X_Mg_wad = (X_Mg_wadopx * (p_ol_polymorphs + p_opx) - p_opx*X_Mg_opx)/p_ol_polymorphs
    
    wad.set_composition([X_Mg_wad, 1.-X_Mg_wad])
    opx.set_composition([X_Mg_opx-0.5*Q, 1-X_Mg_opx-0.5*Q, Q])
    fm45.set_composition([X_Mg_Fe4O5, 1.-X_Mg_Fe4O5])

    wad.set_state(P, T)
    opx.set_state(P, T)
    fm45.set_state(P, T)

    equations = [ 0.5*(opx.partial_gibbs[0] + opx.partial_gibbs[1]) - opx.partial_gibbs[2],
                  ( wad.partial_gibbs[0] + opx.partial_gibbs[1] ) - ( wad.partial_gibbs[1] + opx.partial_gibbs[0] ), 
                  ( wad.partial_gibbs[0] + 2.*fm45.partial_gibbs[1] ) - ( wad.partial_gibbs[1] + 2.*fm45.partial_gibbs[0] ) ]
    
    return equations


def ol_wad_opx_fm45_equilibrium(args, X_Mg_olwadopx, p_ol, T):
    Q, P, X_Mg_ol, X_Mg_wad, X_Mg_opx, X_Mg_Fe4O5 = args
    p_wad = p_ol_polymorphs - p_ol

    # X_Mg_olwadopx = 
    # (p_ol*Mg_ol + p_wad*Mg_wad + p_opx*Mg_opx) / (p_ol + p_wad + p_opx)
    X_Mg_ol_polymorphs = (X_Mg_olwadopx * (p_ol_polymorphs + p_opx) - p_opx*X_Mg_opx)/p_ol_polymorphs

    ol.set_composition([X_Mg_ol, 1.-X_Mg_ol])
    wad.set_composition([X_Mg_wad, 1.-X_Mg_wad])
    opx.set_composition([X_Mg_opx-0.5*Q, 1-X_Mg_opx-0.5*Q, Q])
    fm45.set_composition([X_Mg_Fe4O5, 1.-X_Mg_Fe4O5])

    ol.set_state(P, T)
    wad.set_state(P, T)
    opx.set_state(P, T)
    fm45.set_state(P, T)

    equations = [ X_Mg_wad*p_wad + p_ol * X_Mg_ol - p_ol_polymorphs * X_Mg_ol_polymorphs,
                  0.5*(opx.partial_gibbs[0] + opx.partial_gibbs[1]) - opx.partial_gibbs[2],
                  ol.partial_gibbs[0] - wad.partial_gibbs[0],
                  ol.partial_gibbs[1] - wad.partial_gibbs[1],
                  ( wad.partial_gibbs[0] + opx.partial_gibbs[1] ) - ( wad.partial_gibbs[1] + opx.partial_gibbs[0] ), 
                  ( wad.partial_gibbs[0] + 2.*fm45.partial_gibbs[1] ) - ( wad.partial_gibbs[1] + 2.*fm45.partial_gibbs[0] ) ]
    
    return equations






# Define EMOD oxygen buffer
# enstatite, magnesite, olivine, diamond
EMOD = [en, mag, fo, diam]
QFM = [q, fa, mt]

O2=minerals.HP_2011_fluids.O2()
O2.set_method('cork')

T = 1673.15
O2.set_state(1.e5,T)
X_Mg_olopx = 0.9
X_Mgs=np.linspace(0.8, 0.95, 4)
for X_Mg_olopx in X_Mgs:
    ol_proportions=np.linspace(0.0, p_ol_polymorphs, 11)
    pressures_ol_wad_opx_fm45=np.empty_like(ol_proportions)
    fO2_ol_wad_opx_fm45=np.empty_like(ol_proportions)
    for i, p_ol in enumerate(ol_proportions):
        pressures_ol_wad_opx_fm45[i] = optimize.fsolve(ol_wad_opx_fm45_equilibrium, [0.0, 13.e9, 0.87, 0.88, 0.94, 0.7], args=(X_Mg_olopx, p_ol, T))[1]
        fO2_ol_wad_opx_fm45[i] =  np.log10(fugacity(O2, [wad, opx, fm45]))


    ol_pressures = np.linspace(8.e9, pressures_ol_wad_opx_fm45[-1], 11)
    fO2_ol_opx_fm45 = np.empty_like(ol_pressures)

    for i, P in enumerate(ol_pressures):    
        sol = optimize.fsolve(ol_opx_fm45_equilibrium, [0.0, 0.94, 0.7], args=(X_Mg_olopx, P, T))
        fO2_ol_opx_fm45[i] =  np.log10(fugacity(O2, [ol, opx, fm45]))


    wad_pressures = np.linspace(pressures_ol_wad_opx_fm45[0], 16.e9, 11)
    fO2_wad_opx_fm45 = np.empty_like(wad_pressures)
    
    for i, P in enumerate(wad_pressures): 
        sol = optimize.fsolve(wad_opx_fm45_equilibrium, [0.0, 0.94, 0.7], args=(X_Mg_olopx, P, T))
        fO2_wad_opx_fm45[i] =  np.log10(fugacity(O2, [wad, opx, fm45]))

    
    plt.plot(ol_pressures/1.e9, fO2_ol_opx_fm45, label='ol-opx-Fe4O5')
    plt.plot(wad_pressures/1.e9, fO2_wad_opx_fm45, label='wad-opx-Fe4O5')
    plt.plot(pressures_ol_wad_opx_fm45/1.e9, fO2_ol_wad_opx_fm45, label='ol-wad-opx-Fe4O5')



    
buffer_pressures = np.linspace(8.e9, 16.e9, 9)
fO2_EMOD = np.empty_like(buffer_pressures)
fO2_QFM = np.empty_like(buffer_pressures)

for i, P in enumerate(buffer_pressures):
    for mineral in EMOD:
        mineral.set_state(P, T)
    for mineral in QFM:
        mineral.set_state(P, T)
        
    fO2_EMOD[i] = np.log10(fugacity(O2, EMOD))
    fO2_QFM[i] = np.log10(fugacity(O2, QFM))


plt.plot(buffer_pressures/1.e9, fO2_EMOD, label='EMOD')
plt.plot(buffer_pressures/1.e9, fO2_QFM, label='QFM')

plt.legend(loc='lower left')
plt.show()

# chemical potential of skiagite
print chemical_potentials([wad,opx,fm45], [burnman.processchemistry.dictionarize_formula('Fe5Si3O12')])

# chemical potential of khoharite
print chemical_potentials([wad,opx,fm45], [burnman.processchemistry.dictionarize_formula('Mg3Fe2Si3O12')])
