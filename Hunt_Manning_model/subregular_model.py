#!/usr/python
import os, sys
import numpy as np

from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if not os.path.exists('burnman'):
    sys.path.insert(1,os.path.abspath('../'))


# Benchmarks for the solid solution class
import burnman
from burnman import minerals
from burnman import tools
from burnman.mineral import Mineral
from burnman.minerals import \
    DKS_2013_liquids_tweaked, \
    SLB_2011
from burnman.processchemistry import *
from burnman.chemicalpotentials import *
from burnman import constants
atomic_masses=read_masses()

R=8.31446 # from wiki

class dummy (Mineral):
    def __init__(self):
        formula='Mg2.0Si1.0O4.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'fo',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 0.0 ,
            'S_0': 95.1 ,
            'V_0': 4.366e-05 ,
            'Cp': [233.3, 0.001494, -603800.0, -1869.7] ,
            'a_0': 2.85e-05 ,
            'K_0': 1.285e+11 ,
            'Kprime_0': 3.84 ,
            'Kdprime_0': -3e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class dummy_int (Mineral):
    def __init__(self):
        formula='Mg2.0Si1.0O4.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'fo',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -100000. ,
            'S_0': 95.1 ,
            'V_0': 4.366e-05 ,
            'Cp': [233.3, 0.001494, -603800.0, -1869.7] ,
            'a_0': 2.85e-05 ,
            'K_0': 1.285e+11 ,
            'Kprime_0': 3.84 ,
            'Kdprime_0': -3e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

# Configurational entropy
class hydrous_melt(burnman.SolidSolution):
    def __init__(self):
        # Name
        self.name='Hydrous melt'
        self.type='symmetric'
        self.endmembers = [[dummy(), '[O]'],[dummy_int(), '[Oh]'],[dummy(), '[Ho]']]

        # Interaction parameters
        self.enthalpy_interaction=[[0., 0.], [0.]]
        self.volume_interaction=[[0., 0.], [0.]]
        self.entropy_interaction=[[0., 0.], [0.]]

        burnman.SolidSolution.__init__(self)


liq=hydrous_melt()

def eqm_order(X_int, c, P, T, n_H2Oint, deltaH):

    X_H2O = c*( 1. + X_int*n_H2Oint ) - X_int*n_H2Oint
    X_anh = 1. - X_int - X_H2O

    Wai = 0.
    Wah = 0.
    Wih = 0.
    RTlny_anh = (1.-X_anh)*X_H2O*Wah + (1.-X_anh)*X_int*Wai - X_int*X_H2O*Wih
    RTlny_int =  X_anh*(1.-X_H2O)*Wah - X_anh*X_int*Wai + (1.-X_int)*X_H2O*Wih
    RTlny_H2O =  -X_anh*X_H2O*Wah + X_anh*(1.-X_int)*Wai + X_int*(1.-X_H2O)*Wih

    return deltaH  + R*T*(np.log(X_int) - np.log(X_anh) - n_H2Oint*np.log(X_H2O)) + RTlny_int - RTlny_anh - n_H2Oint*RTlny_H2O

def RTlogactivities(X_int, c, P, T, n_H2Oint, deltaH):

    X_H2O = c*( 1. + X_int*n_H2Oint ) - X_int*n_H2Oint
    X_anh = 1. - X_int - X_H2O

    Wai = 0.
    Wah = 0.
    Wih = 0.
    RTlny_anh = (1.-X_anh)*X_H2O*Wah + (1.-X_anh)*X_int*Wai - X_int*X_H2O*Wih
    RTlny_int =  X_anh*(1.-X_H2O)*Wah - X_anh*X_int*Wai + (1.-X_int)*X_H2O*Wih
    RTlny_H2O =  -X_anh*X_H2O*Wah + X_anh*(1.-X_int)*Wai + X_int*(1.-X_H2O)*Wih

    return [R*T*np.log(X_anh) + RTlny_anh,  R*T*np.log(X_int) + RTlny_int, R*T*np.log(X_H2O) + RTlny_H2O]

def excess_gibbs(X_int, c, P, T, n_H2Oint, deltaH):
    # p_intermediate is the fraction of the maximum possible intermediate phase
    X_H2O = c*( 1. + X_int*n_H2Oint ) - X_int*n_H2Oint
    X_anh = 1. - X_int - X_H2O

    Wai = 0.
    Wah = 0.
    Wih = 0.
    RTlny_anh = (1.-X_anh)*X_H2O*Wah + (1.-X_anh)*X_int*Wai - X_int*X_H2O*Wih
    RTlny_int =  X_anh*(1.-X_H2O)*Wah - X_anh*X_int*Wai + (1.-X_int)*X_H2O*Wih
    RTlny_H2O =  -X_anh*X_H2O*Wah + X_anh*(1.-X_int)*Wai + X_int*(1.-X_H2O)*Wih


    Kd = X_int/(X_anh*np.power(X_H2O, n_H2Oint))
    G_excess = X_int*deltaH  + R*T*(X_int*np.log(X_int) + X_anh*np.log(X_anh) + X_H2O*np.log(X_H2O)) - (X_int*RTlny_int + X_anh*RTlny_anh + X_H2O*RTlny_H2O)

    # Here, proportions add to one, but the solution needs to be scaled 
    # so that there is one mole equivalent in the solution
    sum_moles = (1. + n_H2Oint)*X_int + X_H2O + X_anh 
    return  G_excess/sum_moles

P=13.e9
T =2000.
endmember = dummy() 
hydrous_component = dummy_int() 

compositions=np.linspace(0.0001, 0.9999, 101)
X_ints = np.empty_like(compositions)
Gex=np.empty_like(compositions)
Gex_2=np.empty_like(compositions)
for i, X in enumerate(compositions):
    n = 1.

    endmember.set_state(P, T)
    hydrous_component.set_state(P, T)
    deltaH = hydrous_component.gibbs - endmember.gibbs

    if X < n/(n+1.):
        max_value = X/(n*(1.-X))
    else:
        max_value = (1.-X)/(1.-n*(1.-X))


    res = optimize.minimize(excess_gibbs, max_value-0.000001, method='TNC', bounds=((0.00001, max_value-0.0001),), args=(X, P, T, n, deltaH), options={'disp': False})
    X_int = res.x[0]
    Gex[i]=res.fun[0]

    # alternative
    X_ints[i] = optimize.fsolve(eqm_order, max_value-0.000001, args=(X, P, T, n, deltaH))[0]
    Gex_2[i] = excess_gibbs(X_ints[i], X, P, T, n, deltaH)
    print X_int, X

i = 40
e = excess_gibbs(X_ints[i], compositions[i], P, T, n, deltaH)
a= RTlogactivities(X_ints[i], compositions[i], P, T, n, deltaH)
plt.plot( [0., 0.5, 1.], [a[0], 0.5*(deltaH + a[1]), a[2]], linewidth=1., label='activities')

plt.plot( compositions, Gex, '-', linewidth=2., label='model')
#plt.plot( compositions, Gex_2, '-', linewidth=2., label='model_from_eqm_order')
plt.ylabel("Excess Gibbs (J/mol)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()

exit()


####
# PHASE RELATIONS
####

fo=SLB_2011.forsterite()
Mg2SiO4_liq=DKS_2013_liquids_tweaked.Mg2SiO4_liquid()

def dGfo(temperature):
    fo.set_state(13.e9, temperature)
    Mg2SiO4_liq.set_state(13.e9, temperature)
    return (fo.gibbs - Mg2SiO4_liq.gibbs)


forsterite = []
enstatite=[]
chondrodite=[]
liquid=[]
for line in open('data/13GPa_fo-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[3] == 'f' or content[3] == 'sf' or content[3] == 'f_davide':
            forsterite.append([float(content[0])+273.15, (100. - float(content[1])*7./2.)/100.])
        if content[3] == 'e' or content[3] == 'se' or content[3] == 'e_davide':
            enstatite.append([float(content[0])+273.15, (100. - float(content[1])*7./2.)/100.])
        if content[3] == 'c':
            chondrodite.append([float(content[0])+273.15, (100. - float(content[1])*7./2.)/100.])
        if content[3] == 'l' or content[3] == 'l_davide':
            liquid.append([float(content[0])+273.15,(100. - float(content[1])*7./2.)/100.])

forsterite=zip(*forsterite)
enstatite=zip(*enstatite)
chondrodite=zip(*chondrodite)
liquid=zip(*liquid)
plt.plot( forsterite[1], forsterite[0], marker='.', linestyle='none', label='fo+liquid')
plt.plot( enstatite[1], enstatite[0], marker='.', linestyle='none', label='en+liquid')
plt.plot( chondrodite[1], chondrodite[0], marker='.', linestyle='none', label='chond+liquid')
plt.plot( liquid[1], liquid[0], marker='.', linestyle='none', label='superliquidus')

plt.ylim(1000.,3000.)
plt.xlim(0.,1.)
plt.ylabel("Temperature (K)")
plt.xlabel("X")
plt.legend(loc='upper right')
plt.show()
