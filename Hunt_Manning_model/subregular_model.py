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
R = 8.31446 # from wiki

#### TODO XXXX
#### Change from H excess to S excess
#### Plot at different temperatures
#### Compare to Silver and Stolper model
#### Fit experimental data
#### Better liquidus optimization

def W(T):
    # Wai, Wah, Wih
    #return -65.*T, 0.*T, -70.*T
    #return 0., 0.*T, -5*T
    return 0.*T, -00.*T, 0*T

n = 2.
class dummy_int (Mineral):
    def __init__(self):
        formula='Mg2.0Si1.0O4.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'fo',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 0000. ,
            'S_0': 100. + (100.), # 131*7./4.
            'V_0': 4.366e-05 ,
            'Cp': [233.3, 0.001494, -603800.0, -1869.7] ,
            'a_0': 2.85e-05 ,
            'K_0': 1.285e+11 ,
            'Kprime_0': 3.84 ,
            'Kdprime_0': -3e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class dummy (Mineral):
    def __init__(self):
        formula='Mg2.0Si1.0O4.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'fo',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 0.0 ,
            'S_0': 100. ,
            'V_0': 4.366e-05 ,
            'Cp': [233.3, 0.001494, -603800.0, -1869.7] ,
            'a_0': 2.85e-05 ,
            'K_0': 1.285e+11 ,
            'Kprime_0': 3.84 ,
            'Kdprime_0': -3e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

'''
r = 4./3. # on a one cation basis
fH2O = 1./7.
Xb=fH2O/(fH2O + r*(1.-fH2O)) # eq. 5.3b, (Mg0.5 Si0.25 O4/3 + H2O)
X0=1.-Xb-(0.5 - np.sqrt(0.25 - 1.*(Xb-Xb*Xb))) # eq. 5.3 for K = inf
print fH2O, Xb, X0
print -constants.gas_constant*r*np.log(X0)
'''

def eqm_order(X_int, c, P, T, n_H2Oint, deltaH):

    X_H2O = c*( 1. + X_int*n_H2Oint ) - X_int*n_H2Oint
    X_anh = 1. - X_int - X_H2O

    Wai, Wah, Wih = W(T)
    RTlny_anh = (1.-X_anh)*X_H2O*Wah + (1.-X_anh)*X_int*Wai - X_int*X_H2O*Wih
    RTlny_H2O =  X_anh*(1.-X_H2O)*Wah - X_anh*X_int*Wai + X_int*(1.-X_H2O)*Wih
    RTlny_int =  -X_anh*X_H2O*Wah + X_anh*(1.-X_int)*Wai + (1.-X_int)*X_H2O*Wih

    return deltaH  + R*T*(np.log(X_int) - np.log(X_anh) - n_H2Oint*np.log(X_H2O)) + (RTlny_int - RTlny_anh - n_H2Oint*RTlny_H2O)

def RTlogactivities(X_int, c, P, T, n_H2Oint, deltaH):

    X_H2O = c*( 1. + X_int*n_H2Oint ) - X_int*n_H2Oint
    X_anh = 1. - X_int - X_H2O

    Wai, Wah, Wih = W(T)
    RTlny_anh = (1.-X_anh)*X_H2O*Wah + (1.-X_anh)*X_int*Wai - X_int*X_H2O*Wih
    RTlny_H2O =  X_anh*(1.-X_H2O)*Wah - X_anh*X_int*Wai + X_int*(1.-X_H2O)*Wih
    RTlny_int =  -X_anh*X_H2O*Wah + X_anh*(1.-X_int)*Wai + (1.-X_int)*X_H2O*Wih

    return [R*T*np.log(X_anh) + RTlny_anh,  R*T*np.log(X_int) + RTlny_int, R*T*np.log(X_H2O) + RTlny_H2O]


def excess_gibbs(X_int, c, P, T, n_H2Oint, deltaH):
    
    X_H2O = c*( 1. + X_int*n_H2Oint ) - X_int*n_H2Oint
    X_anh = 1. - X_int - X_H2O

    Wai, Wah, Wih = W(T)
    RTlny_anh = (1.-X_anh)*X_H2O*Wah + (1.-X_anh)*X_int*Wai - X_int*X_H2O*Wih
    RTlny_H2O =  X_anh*(1.-X_H2O)*Wah - X_anh*X_int*Wai + X_int*(1.-X_H2O)*Wih
    RTlny_int =  -X_anh*X_H2O*Wah + X_anh*(1.-X_int)*Wai + (1.-X_int)*X_H2O*Wih

    G_excess = X_int*deltaH  + R*T*(X_int*np.log(X_int) + X_anh*np.log(X_anh) + X_H2O*np.log(X_H2O)) + (X_int*RTlny_int + X_anh*RTlny_anh + X_H2O*RTlny_H2O)

    # Here, proportions add to one, but the solution needs to be scaled 
    # so that there is one mole equivalent in the solution
    sum_moles = (1. + n_H2Oint)*X_int + X_H2O + X_anh 
    return  G_excess/sum_moles



endmember = dummy() 
hydrous_component = dummy_int() 
'''
endmember.set_state(13.e9, 1000.)
hydrous_component.set_state(13.e9, 1000.)
deltaH = hydrous_component.gibbs - endmember.gibbs
print deltaH / 1000.
exit()
'''
compositions=np.linspace(0.0001, 0.9999, 101)
X_ints = np.empty_like(compositions)
Gex=np.empty_like(compositions)
Gex_2=np.empty_like(compositions)

pressure=13.e9
temperatures = [1., 1000., 2000., 3000.]
for temperature in temperatures:

    endmember.set_state(pressure, temperature)
    hydrous_component.set_state(pressure, temperature)
    deltaH = hydrous_component.gibbs - endmember.gibbs

    for i, X in enumerate(compositions):
        
        if X < n/(n+1.):
            max_value = X/(n*(1.-X))
        else:
            max_value = (1.-X)/(1.-n*(1.-X))


        # alternative
        res = optimize.minimize(excess_gibbs, max_value-0.000001, method='TNC', bounds=((0.00001, max_value-0.0001),), args=(X, pressure, temperature, n, deltaH), options={'disp': False})
        X_int = res.x[0]
        Gex[i]=res.fun[0]

        #X_ints[i] = optimize.fsolve(eqm_order, max_value-0.000001, args=(X, pressure, temperature, n, deltaH))[0]
        #Gex_2[i] = excess_gibbs(X_ints[i], X, pressure, temperature, n, deltaH)
        #print X_ints[i], X

    i = 70
    e = excess_gibbs(X_ints[i], compositions[i], pressure, temperature, n, deltaH)
    a= RTlogactivities(X_ints[i], compositions[i], pressure, temperature, n, deltaH)
    plt.plot( [0., n/(n+1.), 1.], [a[0], 1./(1.+n)*(deltaH + a[1]), a[2]], linewidth=1., label='activities')

    plt.plot( compositions, Gex, '-', linewidth=2., label='model at '+str(temperature)+' K')
    
    #plt.plot( compositions, Gex_2, '-', linewidth=2., label='model_from_eqm_order')
plt.ylabel("Excess Gibbs (J/mol)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()


def delta_gibbs(temperature, pressure, phase1, phase2, factor1, factor2):
    phase1.set_state(pressure, temperature[0])
    phase2.set_state(pressure, temperature[0])
    return (phase1.gibbs*factor1 - phase2.gibbs*factor2)

def eqm_composition(X, P, T, n, deltaH, dG):
    if X < n/(n+1.):
        max_value = X/(n*(1.-X))
    else:
        max_value = (1.-X)/(1.-n*(1.-X))
    X_int = optimize.fsolve(eqm_order, max_value-0.000001, args=(X, P, T, n, deltaH))[0]
    return dG - RTlogactivities(X_int, X, P, T, n, deltaH)[0]

def eqm_composition_br(X, P, T, n, deltaH, dG):
    if X < n/(n+1.):
        max_value = X/(n*(1.-X))
    else:
        max_value = (1.-X)/(1.-n*(1.-X))
    X_int = optimize.fsolve(eqm_order, max_value-0.000001, args=(X, P, T, n, deltaH))[0]
    dGbr = 0.5*dG + 0.5*(- 8330. + 20.*(T-1473.15))
    return dGbr - 0.5*(RTlogactivities(X_int, X, P, T, n, deltaH)[0] + RTlogactivities(X_int, X, P, T, n, deltaH)[2])


####
# PHASE RELATIONS
####
'''
fo=SLB_2011.forsterite()
Mg2SiO4_liq=DKS_2013_liquids_tweaked.Mg2SiO4_liquid()

pressure = 13.e9
T_melt = optimize.fsolve(delta_gibbs, 3000., args=(pressure, fo, Mg2SiO4_liq, 1./3., 1./3))[0]
print T_melt

temperatures = np.linspace(1200., T_melt, 101)
compositions = np.empty_like(temperatures)
for i, temperature in enumerate(temperatures):
    dG = delta_gibbs([temperature], 13.e9, fo, Mg2SiO4_liq, 1./3., 1./3.)
    endmember.set_state(pressure, temperature)
    hydrous_component.set_state(pressure, temperature)
    deltaH = hydrous_component.gibbs - endmember.gibbs
    compositions[i]=optimize.fsolve(eqm_composition, 1e-5, args=(pressure, temperature, 4./3., deltaH, dG))[0]

plt.plot(compositions, temperatures)
    
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

forsterite=np.array(zip(*forsterite))
enstatite=np.array(zip(*enstatite))
chondrodite=np.array(zip(*chondrodite))
liquid=np.array(zip(*liquid))
add_T = 50. # K
plt.plot( forsterite[1], forsterite[0] + add_T, marker='.', linestyle='none', label='fo+liquid')
plt.plot( enstatite[1], enstatite[0] + add_T, marker='.', linestyle='none', label='en+liquid')
plt.plot( chondrodite[1], chondrodite[0] + add_T, marker='.', linestyle='none', label='chond+liquid')
plt.plot( liquid[1], liquid[0] + add_T, marker='.', linestyle='none', label='superliquidus')

plt.ylim(1000.,3000.)
plt.xlim(0.,1.)
plt.ylabel("Temperature (K)")
plt.xlabel("X")
plt.legend(loc='upper right')
plt.show()

'''


'''
##########################
# PERICLASE
per=SLB_2011.periclase()
MgO_liq=DKS_2013_liquids_tweaked.MgO_liquid()


pressure = 13.e9
T_melt = optimize.fsolve(delta_gibbs, 3000., args=(pressure, per, MgO_liq, 1., 1.))[0]
print T_melt

temperatures = np.linspace(1200., T_melt, 101)
compositions = np.empty_like(temperatures)
compositions_br = np.empty_like(temperatures)
for i, temperature in enumerate(temperatures):
    dG = delta_gibbs([temperature], 13.e9, per, MgO_liq, 1., 1.)
    endmember.set_state(pressure, temperature)
    hydrous_component.set_state(pressure, temperature)
    deltaH = hydrous_component.gibbs - endmember.gibbs
    compositions[i]=optimize.fsolve(eqm_composition, 1e-5, args=(pressure, temperature, 1., deltaH, dG))[0]
    compositions_br[i]=optimize.fsolve(eqm_composition_br, 0.999, args=(pressure, temperature, 1., deltaH, dG))[0]

plt.plot(compositions, temperatures)
plt.plot(compositions_br, temperatures)

periclase=[]
brucite=[]
liquid=[]
for line in open('data/13GPa_per-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[2] == 'p' or content[2] == 'sp':
            periclase.append([float(content[0])+273.15, float(content[1])/100.])
        if content[2] == 'l':
            liquid.append([float(content[0])+273.15, float(content[1])/100.])
        if content[2] == 'b':
            brucite.append([float(content[0])+273.15, float(content[1])/100.])

periclase=zip(*periclase)
brucite=zip(*brucite)
liquid=zip(*liquid)
plt.plot( periclase[1], periclase[0], marker='.', linestyle='none', label='per+liquid')
plt.plot( brucite[1], brucite[0], marker='.', linestyle='none', label='br+liquid')
plt.plot( liquid[1], liquid[0], marker='.', linestyle='none', label='superliquidus')

plt.ylim(1000.,5500.)
plt.xlim(0.,1.)
plt.ylabel("Temperature (K)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()
'''




##########################
# STISHOVITE
stv=SLB_2011.stishovite()
SiO2_liq=DKS_2013_liquids_tweaked.SiO2_liquid()


pressure = 13.e9
T_melt = optimize.fsolve(delta_gibbs, 3000., args=(pressure, stv, SiO2_liq, 1., 1.))[0]
print T_melt

temperatures = np.linspace(1200., T_melt, 101)
compositions = np.empty_like(temperatures)
compositions_br = np.empty_like(temperatures)
for i, temperature in enumerate(temperatures):
    dG = delta_gibbs([temperature], 13.e9, stv, SiO2_liq, 1., 1.)
    endmember.set_state(pressure, temperature)
    hydrous_component.set_state(pressure, temperature)
    deltaH = hydrous_component.gibbs - endmember.gibbs
    compositions[i]=optimize.fsolve(eqm_composition, 1e-5, args=(pressure, temperature, n, deltaH, dG))[0]

plt.plot(compositions, temperatures)


stishovite = []
s_stishovite = []
liquid=[]
for line in open('data/13GPa_SiO2-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[2] == 's':
            stishovite.append([float(content[0])+273.15, 1.-float(content[1])])
        if content[2] == 'ss':
            s_stishovite.append([float(content[0])+273.15, 1.-float(content[1])])
        if content[2] == 'l':
            liquid.append([float(content[0])+273.15, 1.-float(content[1])])

stishovite=np.array(zip(*stishovite))
s_stishovite=np.array(zip(*s_stishovite))
liquid=np.array(zip(*liquid))
add_T = 50. # K
plt.plot( stishovite[1], stishovite[0]+add_T, marker='.', linestyle='none', label='stv+liquid')
plt.plot( s_stishovite[1], s_stishovite[0]+add_T, marker='.', linestyle='none', label='(stv)+liquid')
plt.plot( liquid[1], liquid[0]+add_T, marker='.', linestyle='none', label='superliquidus')

plt.ylim(1000.,3000.)
plt.xlim(0.,1.)
plt.ylabel("Temperature (K)")
plt.xlabel("X")
plt.legend(loc='upper right')
plt.show()
