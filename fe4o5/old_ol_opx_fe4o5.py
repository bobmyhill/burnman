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

# Variables should be entropy, thermal expansion (although we have essentially no constraints on these), enthalpy and bulk modulus

class enstatite (Mineral):
    def __init__(self):
       formula='Mg2.0Si2.0O6.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'en',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -3090220.0 ,
            'S_0': 132.5 ,
            'V_0': 6.262e-05 ,
            'Cp': [356.2, -0.00299, -596900.0, -3185.3] ,
            'a_0': 2.27e-05 ,
            'K_0': 1.059e+11 ,
            'Kprime_0': 8.65 ,
            'Kdprime_0': -8.2e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)

class ferrosilite (Mineral):
    def __init__(self):
       formula='Fe2.0Si2.0O6.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'fs',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -2388710.0 ,
            'S_0': 189.9 ,
            'V_0': 6.592e-05 ,
            'Cp': [398.7, -0.006579, 1290100.0, -4058.0] ,
            'a_0': 3.26e-05 ,
            'K_0': 1.01e+11 ,
            'Kprime_0': 4.08 ,
            'Kdprime_0': -4e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)

DeltaH_fm=-6950.
class ordered_fm_opx (Mineral):
    def __init__(self):
       formula='Fe2.0Si2.0O6.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'fm',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': (-2388710.0-3090220.0)/2.+DeltaH_fm ,
            'S_0': (132.5+189.9)/2. ,
            'V_0': (6.592e-05+6.262e-05)/2. ,
            'Cp': [(398.7+356.2)/2., (-0.006579-0.00299)/2., (1290100.0-596900.0)/2., (-4058.0-3185.3)/2.] ,
            'a_0': (3.26e-05+2.27e-05)/2. ,
            'K_0': (1.01e+11+1.059e+11)/2. ,
            'Kprime_0': (4.08+8.65)/2. ,
            'Kdprime_0': -1.*(4.08+8.65)/(1.01e+11+1.059e+11) ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)
       
en=enstatite()
fs=ferrosilite()
fm=ordered_fm_opx()

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


def opx_composition(arg, X_Mg, P, T):
    Q=arg[0]
    opx.set_composition([X_Mg-0.5*Q, 1-X_Mg-0.5*Q, Q])
    opx.set_state(P,T)
    return opx.gibbs

P=1.e9
T=1473.15
X_Mgs=np.linspace(0.0,1.0,101)
G=np.empty_like(X_Mgs)
Q=np.empty_like(X_Mgs)
for idx, X_Mg in enumerate(X_Mgs):
    optimize.minimize(opx_composition, [-0.01], method='nelder-mead', args=(X_Mg, P, T))
    G[idx]=opx.gibbs
    Q[idx]=opx.molar_fractions[2]

plt.plot( X_Mgs, G, '-', linewidth=3., label='Gibbs opx')

X_Mg=0.5
optimize.minimize(opx_composition, [-0.01], method='nelder-mead', args=(X_Mg, P, T))
print opx.partial_gibbs, opx.gibbs, (opx.partial_gibbs[0]+opx.partial_gibbs[1])/2.

plt.title('')
plt.xlabel("X_Mg opx")
plt.ylabel("G")
plt.legend(loc='lower right')
plt.show()


# (Mg,Fe)2Fe2O5 - ol/wad/rw - opx/hpx diagram
# i.e. 
# 2(Mg,Fe)Fe3O5
# (Mg,Fe)2SiO4
# (Mg,Fe)2Si2O6


#bulk composition = (Mg,Fe)6 Fe6 Si3 O20
# X_Mg_bulk = sum(X_Mg)/3.

def foo_compositions(arg, ol_polymorph, bulk_XMg, P, T):
    # args are X_Mg_Fe4O5, X_Mg_ol_polymorph, Q_opx
    X_Mg_Fe4O5=float(arg[0])
    X_Mg_ol_polymorph=float(arg[1])
    X_Mg_opx=3.*bulk_XMg -1.0*X_Mg_Fe4O5 -1.0*X_Mg_ol_polymorph
    Q_opx=float(arg[2])

    fm45.set_composition([X_Mg_Fe4O5, 1.0-X_Mg_Fe4O5])
    ol_polymorph.set_composition([X_Mg_ol_polymorph, 1.0-X_Mg_ol_polymorph])


    p_en=X_Mg_opx - Q_opx/2.
    p_fm=Q_opx
    p_fs=1. - p_en - p_fm

    opx.set_composition([p_en, p_fs, p_fm])

    fm45.set_state(P,T)
    ol_polymorph.set_state(P,T)
    opx.set_state(P,T)

    return 2.*fm45.gibbs + ol_polymorph.gibbs + opx.gibbs


P=10.5e9
T=1373.
O2=minerals.HP_2011_fluids.O2()
O2.set_method('cork')
O2.set_state(1.e5,T)

Re=minerals.Metal_Metal_oxides.Re()
ReO2=minerals.Metal_Metal_oxides.ReO2()

bulk_XMgs=np.linspace(0.01, 0.8, 11)
for bulk_XMg in bulk_XMgs:
    ol_polymorph=ol
    optimize.minimize(foo_compositions, [0.00,0.0,-0.01], method='nelder-mead', args=(ol_polymorph, bulk_XMg, P, T))

    assemblage=[ol, opx, fm45]
    #print ol.partial_gibbs, opx.partial_gibbs
    print ol.molar_fractions[0], opx.molar_fractions[0] + 0.5*opx.molar_fractions[2], fm45.molar_fractions[0], np.log10(fugacity(O2, assemblage))

    Re.set_state(P,T)
    ReO2.set_state(P,T)
    print np.log10(fugacity(O2, [Re, ReO2]))
# find oxygen fugacity
 
