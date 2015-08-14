import numpy as np
from scipy import optimize
from burnman.chemicalpotentials import *

def eqm_pressure(P, T, minerals, multiplicities):
    gibbs = 0.
    for i, mineral in enumerate(minerals):
        mineral.set_state(P[0], T)
        gibbs += mineral.gibbs * multiplicities[i]
    return gibbs

def eqm_with_wus(comp, P, T, wus, mineral):
    XMgO = 0.
    c=comp[0]
    wus.set_composition([XMgO, (1.0-c)*(1.-XMgO), c*(1.-XMgO)])
    wus.set_state(P,T)
    mu_mineral=chemical_potentials([wus],[mineral.params['formula']])[0]
    return  mu_mineral-mineral.calcgibbs(P,T)

def wus_eqm_c_P(arg, T, wus, min1, min2):
    XMgO=0.0
    c=arg[0]
    P=arg[1]
    wus.set_composition([XMgO, (1.0-c)*(1.-XMgO), c*(1.-XMgO)])
    wus.set_state(P, T)
    min1.set_state(P, T)
    min2.set_state(P, T)
    mu=chemical_potentials([wus],
                           [min1.params['formula'],
                            min2.params['formula']])
    return [mu[0]-min1.calcgibbs(P,T), mu[1]-min2.calcgibbs(P,T)]


def eqm_with_wus_2(comp, P, T, wus, Fe2_assemblage):
    XMgO = 0.
    c=comp[0]
    wus.set_composition([XMgO, (1.0-c)*(1.-XMgO), c*(1.-XMgO)])
    wus.set_state(P,T)
    mu_FeO=chemical_potentials([wus],[dictionarize_formula('FeO')])[0]
    mu_FeO_2=chemical_potentials(Fe2_assemblage,[dictionarize_formula('FeO')])[0]
    return mu_FeO - mu_FeO_2 

def eqm_curve(assemblage, pressures, T, O2):
    fO2s = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        for mineral in assemblage:
            mineral.set_state(P,T)
        fO2s[i] = np.log10(fugacity(O2, assemblage))
    return fO2s

def eqm_curve_wus(mineral, wus, pressures, T, O2):
    fO2s = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        mineral.set_state(P,T)
        optimize.fsolve(eqm_with_wus, 0.16, args=(P, T, wus, mineral))
        assemblage=[wus, mineral]
        fO2s[i] = np.log10(fugacity(O2, assemblage))
    return fO2s

def eqm_curve_wus_2(Fe2_assemblage, wus, pressures, T, O2):
    fO2s = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        for mineral in Fe2_assemblage:
            mineral.set_state(P, T)
        optimize.fsolve(eqm_with_wus_2, 0.16, args=(P, T, wus, Fe2_assemblage))
        assemblage=Fe2_assemblage
        assemblage.append(wus)
        fO2s[i] = np.log10(fugacity(O2, assemblage))
    return fO2s
