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

class dummy (Mineral):
    def __init__(self):
        formula='Mg2.0Si1.0O4.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'fo',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 000.0 ,
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


class hydrous_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Asymmetric pyrope-almandine-grossular garnet'
        self.type='asymmetric'
        self.endmembers = [[dummy(), '[O]'],[dummy(), '[H]']]
        self.alphas=[1.0, 2.0]
        self.enthalpy_interaction=[[0.0e3]]
        self.entropy_interaction=[[40.0]]
        burnman.SolidSolution.__init__(self, molar_fractions)
        
class hydrous_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Asymmetric pyrope-almandine-grossular garnet'
        self.type='subregular'
        self.endmembers = [[dummy(), '[O]'],[dummy(), '[H]']]
        self.enthalpy_interaction=[[[0.0e3, 0.0e3]]]
        self.entropy_interaction=[[[20.0, 50.0]]]
        burnman.SolidSolution.__init__(self, molar_fractions)

def delta_gibbs(temperature, pressure, phase1, phase2, factor1, factor2):
    phase1.set_state(pressure, temperature[0])
    phase2.set_state(pressure, temperature[0])
    return (phase1.gibbs*factor1 - phase2.gibbs*factor2)

def eqm_composition(X, solution, P, T, dG):
    solution.set_composition([1.-X[0], X[0]])
    solution.set_state(pressure, temperature)
    return dG - solution.excess_partial_gibbs[0]


pressure=13.e9
temperature=2000.
liq = hydrous_liquid()
stv=SLB_2011.stishovite()
SiO2_liq=DKS_2013_liquids_tweaked.SiO2_liquid()

T_melt = optimize.fsolve(delta_gibbs, 3000., args=(pressure, stv, SiO2_liq, 1., 1.))[0]
print T_melt

temperatures=np.linspace(1000., T_melt, 101)
compositions=np.empty_like(temperatures)
for i, temperature in enumerate(temperatures):
    dG = delta_gibbs([temperature], pressure, stv, SiO2_liq, 1., 1.)
    compositions[i] = optimize.fsolve(eqm_composition, 0.001, args=(liq, pressure, temperature, dG))[0]

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

'''
temperatures=np.linspace(1400., T_melt, 5)
compositions=np.linspace(0.0001, 0.9999, 101)
excess_gibbs=np.empty_like(compositions)
for temperature in temperatures:
    for i, c in enumerate(compositions):
        liq.set_composition([1.-c, c])
        liq.set_state(pressure, temperature)
        excess_gibbs[i] = liq.excess_gibbs

    plt.plot(compositions, excess_gibbs, linewidth=1, label=str(temperature)+'K')
plt.legend(loc='lower left')
plt.show()
'''




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
        X_ints[i] = res.x[0]
        Gex[i]=res.fun[0]

        #X_ints[i] = optimize.fsolve(eqm_order, max_value-0.000001, args=(X, pressure, temperature, n, deltaH))[0]
        #Gex_2[i] = excess_gibbs(X_ints[i], X, pressure, temperature, n, deltaH)
        #print X_ints[i], X

    i = 70
    e = excess_gibbs(X_ints[i], compositions[i], pressure, temperature, n, deltaH)
    a= RTlogactivities(X_ints[i], compositions[i], pressure, temperature, n, deltaH)
    plt.plot( [compositions[i]], [e], marker='o', linestyle='None', label='activities')
    plt.plot( [0., 1.], [a[0], a[2]], linewidth=1., label='activities')

    plt.plot( compositions, Gex, '-', linewidth=2., label='model at '+str(temperature)+' K')
    
    #plt.plot( compositions, Gex_2, '-', linewidth=2., label='model_from_eqm_order')
plt.ylabel("Excess Gibbs (J/mol)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()
'''



##########################
# PERICLASE
per=SLB_2011.periclase()
MgO_liq=DKS_2013_liquids_tweaked.MgO_liquid()


pressure=13.e9
temperature=2000.
liq = hydrous_liquid()
stv=SLB_2011.stishovite()
SiO2_liq=DKS_2013_liquids_tweaked.SiO2_liquid()


liq.type='subregular'
liq.endmembers = [[dummy(), '[O]'],[dummy(), '[H]']]
liq.enthalpy_interaction=[[[0.0e3, 0.0e3]]]
liq.entropy_interaction=[[[70.0, 70.0]]]
burnman.SolidSolution.__init__(liq)


T_melt = optimize.fsolve(delta_gibbs, 3000., args=(pressure, per, MgO_liq, 1., 1.))[0]
print T_melt


temperatures=np.linspace(1000., T_melt, 101)
compositions=np.empty_like(temperatures)
for i, temperature in enumerate(temperatures):
    dG = delta_gibbs([temperature], pressure, per, MgO_liq, 1., 1.)
    compositions[i] = optimize.fsolve(eqm_composition, 0.001, args=(liq, pressure, temperature, dG))[0]

plt.plot(compositions, temperatures)


periclase=[]
brucite=[]
liquid=[]
add_T = 50. # K
for line in open('data/13GPa_per-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[2] == 'p' or content[2] == 'sp':
            periclase.append([float(content[0])+273.15, float(content[1])/100.])
        if content[2] == 'l':
            liquid.append([float(content[0])+273.15, float(content[1])/100.])
        if content[2] == 'b':
            brucite.append([float(content[0])+273.15, float(content[1])/100.])

periclase=np.array(zip(*periclase))
brucite=np.array(zip(*brucite))
liquid=np.array(zip(*liquid))
plt.plot( periclase[1], periclase[0] + add_T, marker='.', linestyle='none', label='per+liquid')
plt.plot( brucite[1], brucite[0] + add_T, marker='.', linestyle='none', label='br+liquid')
plt.plot( liquid[1], liquid[0] + add_T, marker='.', linestyle='none', label='superliquidus')

plt.ylim(1000.,5500.)
plt.xlim(0.,1.)
plt.ylabel("Temperature (K)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()


####
# FORSTERITE
####

fo=SLB_2011.forsterite()
Mg2SiO4_liq=DKS_2013_liquids_tweaked.Mg2SiO4_liquid()


liq.type='subregular'
liq.endmembers = [[dummy(), '[O]'],[dummy(), '[H]']]
liq.enthalpy_interaction=[[[0.0e3, 0.0e3]]]
liq.entropy_interaction=[[[10.0, 35.0]]] # 20 50 for stv, 70 70 for per
burnman.SolidSolution.__init__(liq)


T_melt = optimize.fsolve(delta_gibbs, 3000., args=(pressure, fo, Mg2SiO4_liq, 1./3., 1./3.))[0]
print T_melt


temperatures=np.linspace(1000., T_melt, 101)
compositions=np.empty_like(temperatures)
for i, temperature in enumerate(temperatures):
    dG = delta_gibbs([temperature], pressure, fo, Mg2SiO4_liq, 1./3., 1./3.)
    compositions[i] = optimize.fsolve(eqm_composition, 0.001, args=(liq, pressure, temperature, dG))[0]

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





###
# ENSTATITE
###


cen=SLB_2011.hp_clinoenstatite()
MgSiO3_liq=DKS_2013_liquids_tweaked.MgSiO3_liquid()


liq.type='subregular'
liq.endmembers = [[dummy(), '[O]'],[dummy(), '[H]']]
liq.enthalpy_interaction=[[[0.0e3, 0.0e3]]]
liq.entropy_interaction=[[[-70.0, 20.0]]] # 20 50 for stv, 70 70 for per
burnman.SolidSolution.__init__(liq)


T_melt = optimize.fsolve(delta_gibbs, 3000., args=(pressure, cen, MgSiO3_liq, 1./4., 1./2.))[0]
print T_melt


temperatures=np.linspace(1000., T_melt, 101)
compositions=np.empty_like(temperatures)
for i, temperature in enumerate(temperatures):
    dG = delta_gibbs([temperature], pressure, cen, MgSiO3_liq, 1./4., 1./2.)
    compositions[i] = optimize.fsolve(eqm_composition, 0.001, args=(liq, pressure, temperature, dG))[0]

plt.plot(compositions, temperatures)




enstatite=[]
liquid=[]
y_liquid=[]
for line in open('data/13GPa_en-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[2] == 'e' or content[2] == 'se' or content[2] == 'e_davide':
            enstatite.append([float(content[0])+273.15, (100. - float(content[1])*2.)/100.])
        if content[2] == 'l' or content[2] == 'l_davide':
            liquid.append([float(content[0])+273.15,(100. - float(content[1])*2.)/100.])
        if content[2] == 'l_Yamada':
            y_liquid.append([float(content[0])+273.15,(100. - float(content[1])*2.)/100.])

enstatite=np.array(zip(*enstatite))
liquid=np.array(zip(*liquid))
y_liquid=np.array(zip(*y_liquid))
add_T = 50. # K
plt.plot( y_liquid[1], y_liquid[0] + add_T, linewidth=1, label='liquidus (Yamada et al., 2004)')
plt.plot( enstatite[1], enstatite[0] + add_T, marker='.', linestyle='none', label='en+liquid')
plt.plot( liquid[1], liquid[0] + add_T, marker='.', linestyle='none', label='superliquidus')

plt.ylim(1000.,3000.)
plt.xlim(0.,1.)
plt.ylabel("Temperature (K)")
plt.xlabel("X")
plt.legend(loc='upper right')
plt.show()
