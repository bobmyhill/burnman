from __future__ import absolute_import

# Here we import standard python modules that are required for
# usage of BurnMan.  In particular, numpy is used for handling
# numerical arrays and mathematical operations on them, and
# matplotlib is used for generating plots of results of calculations
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman.equilibriumassemblage import *

from scipy.optimize import fsolve, minimize


from SLB_andradite import andradite
from liquid_iron import liquid_iron

Fe_liq = liquid_iron()

py = minerals.HP_2011_ds62.py()
alm = minerals.HP_2011_ds62.alm()
gr = minerals.HP_2011_ds62.gr()
andr = minerals.HP_2011_ds62.andr()
maj = minerals.HP_2011_ds62.maj()
mrw = minerals.HP_2011_ds62.mrw()
frw = minerals.HP_2011_ds62.frw()


# If we know the composition of FeS melt and ringwoodite, we
# can calculate the activity of skiagite.

# Additional constraint might be a certain bulk oxygen content.
# For this, we would need to know the amount of iron in the other phases...

'''
Endmembers for garnet (10 symmetric interaction parameters)
pyrope
almandine
grossular
andradite
mg-majorite

A ringwoodite-garnet majorite field is often present at ~18 GPa, with wuestite becoming stable at higher temperature and Ca-perovskite at higher pressure

Thus, it seems reasonable to take solid solutions for ringwoodite and garnet only.
'''


class garnet(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name = 'garnet'
        self.type = 'asymmetric'
        self.endmembers = [[py, '[Mg]3[Al][Al]Si3O12'],
                           [alm, '[Fe]3[Al][Al]Si3O12'],
                           [gr, '[Ca]3[Al][Al]Si3O12'],
                           [andr, '[Ca]3[Fe][Fe]Si3O12'],
                           [maj, '[Mg]3[Mg][Si]Si3O12']]

        
        self.energy_interaction = [[2.5e3, 31.e3, 53.2e3, 15.e3],
                                   [5.e3, 37.26e3, 18.e3],
                                   [2.e3, 48.e3],
                                   [-60.e3]]

        self.alphas = [1., 1., 2.7, 2.7, 1.]
        
        burnman.SolidSolution.__init__(self, molar_fractions)

gt = garnet()

class mg_fe_ringwoodite(burnman.SolidSolution):

    def __init__(self, molar_fractions=None):
        self.name = 'ringwoodite'
        self.type = 'symmetric'
        self.endmembers = [
            [mrw, '[Mg]2SiO4'], [frw, '[Fe]2SiO4']]
        self.energy_interaction = [[9.34084e3]]

        burnman.SolidSolution.__init__(self, molar_fractions)

rw = mg_fe_ringwoodite()
        
gt.set_composition([1.0, 0., 0., 0., 0.])
gt.set_state(1.e9, 300.)
print(gt.molar_mass)


# Fe-S model from Lee et al., 2007 (Thermodynamic Calculations on the Stability of Cu2S in Low Carbon Steels)

def Gxs_Fe_S(x_Fe, x_FeS, T):

    x_S = 1. - x_Fe - x_FeS
    
    L_FeFeS = lambda T, x_Fe, x_FeS, x_S:  51879. - 23.1867*T + 13765.*(x_Fe - x_FeS)
    L_FeSS = lambda T, x_Fe, x_FeS, x_S:   48313. - 21.807*T  + (x_FeS - x_S)*(-72983. + 24.7145*T)
    
    S_config = -burnman.constants.gas_constant*(x_Fe*np.log(x_Fe) + x_S*np.log(x_S) + x_FeS*np.log(x_FeS))
    FeS_contrib = x_FeS*(-104225. - 1.479*T)
    
    sum_atoms = x_Fe + x_S + 2.*x_FeS
    
    return (FeS_contrib -T*S_config + x_Fe*x_FeS*L_FeFeS(T, x_Fe, x_FeS, x_S) + x_FeS*x_S*L_FeSS(T, x_Fe, x_FeS, x_S))/sum_atoms # 1 atom basis

def gibbs_Fe_S_fixed_order(Q, X_S, T):
    n_FeS = Q[0]*(0.5 - np.abs(X_S-0.5))
    n_Fe = (1. - X_S) - n_FeS
    n_S = X_S - n_FeS
    n_total = 1. - n_FeS
    x_Fe = n_Fe/n_total
    x_FeS = n_FeS/n_total
    return Gxs_Fe_S(x_Fe, x_FeS, T)

def activity_Fe_in_Fe_S(X_S, T):
    G = minimize(gibbs_Fe_S_fixed_order, [0.9999], args = (X_S, T), method='Nelder-Mead', bounds=((0., 1.),)).fun
    dX = 1.e-3
    G1 = minimize(gibbs_Fe_S_fixed_order, [0.9999], args = (X_S-dX, T), method='Nelder-Mead', bounds=((0., 1.),)).fun
    G2 = minimize(gibbs_Fe_S_fixed_order, [0.9999], args = (X_S+dX, T), method='Nelder-Mead', bounds=((0., 1.),)).fun

    dG = (G2 - G1)/(2.*dX)
    return np.exp((G - dG*X_S)/(burnman.constants.gas_constant * T))

'''
for T in [1200., 1800., 2400., 3600.]:
    X_Ss = np.linspace(0.001, 0.999, 101)
    Gs = np.array([minimize(gibbs_Fe_S_fixed_order, [0.9999], args = (X_S, T), method='Nelder-Mead', bounds=((0., 1.),)).fun for X_S in X_Ss])
    # replace fun with x[0] to look at the order parameter
    plt.plot(X_Ss, Gs, label=str(T)+' K')

plt.xlim(0., 1.)
plt.xlabel('S/(Fe+S)')
plt.legend(loc='lower right')
plt.show()

for T in [1200., 1800., 2400., 3600.]:
    X_Ss = np.linspace(0.001, 0.999, 101)
    plt.plot(X_Ss, np.array([activity_Fe_in_Fe_S(X_S, T) for X_S in X_Ss]), label=str(T)+' K')

plt.xlim(0., 1.)
plt.xlabel('S/(Fe+S)')
plt.legend(loc='lower right')
plt.show()
'''







P= 20.e9
T = 2000.

c = burnman.processchemistry.component_to_atom_fractions({'CaO': 2.0, 'FeO': 18.4, 'MgO': 32.6, 'Al2O3': 2.5, 'SiO2': 44.0}, 'weight')
#c = burnman.processchemistry.component_to_atom_fractions({'CaO': 2.0, 'FeO': 8., 'MgO': 42., 'Al2O3': 2.5, 'SiO2': 44.0}, 'weight')
print(c)
'''
'''
n_gr = c['Ca']/3.
n_py_alm = c['Al']/2. - n_gr

n_Si_rw_and_maj = c['Si'] - n_py_alm*3. - n_gr*3.
n_Mg_Fe_rw_and_maj = c['Mg'] + c['Fe'] - n_py_alm*3.

# n_Mg_Fe_rw_and_maj = 4.*n_maj + 2.*n_rw
# n_Si_rw_and_maj = 4.*n_maj + 1.*n_rw

n_rw = n_Mg_Fe_rw_and_maj - n_Si_rw_and_maj 
n_maj = (n_Si_rw_and_maj - n_rw)/4.


# Assuming Kd = 1 for Mg, Fe partitioning
Mg = c['Mg'] - 4.*n_maj

n_py = n_py_alm*Mg/(Mg + c['Fe'])
n_alm = n_py_alm - n_py
n_mrw = n_rw*Mg/(Mg + c['Fe'])
n_frw = n_rw - n_mrw

n_Fe = 0.0001
print(n_Fe, n_py, n_alm + n_Fe, n_gr - n_Fe, n_Fe, n_maj, n_mrw, n_frw - 3.*n_Fe)
'''
'''

X_Ss = np.linspace(0.01, 0.5, 51)
ferric_fraction = np.empty_like(X_Ss)
majorite_fraction = np.empty_like(X_Ss)
redox_iron_amount = np.empty_like(X_Ss)
S_ppm = np.empty_like(X_Ss)
E0_Fe_liq = Fe_liq.params['E_0']

x_FeS_shallow = 0.5

for i, X_S in enumerate(X_Ss):
    a_Fe = activity_Fe_in_Fe_S(X_S, T)
    Fe_liq.params['E_0'] = E0_Fe_liq + burnman.constants.gas_constant*T*np.log(a_Fe)

    assemblage = burnman.Composite([Fe_liq, gt, rw])
    constraints = [['P', P], ['T', T]]
    sol = gibbs_minimizer(c, assemblage, constraints)
    #print(sol, gt.molar_fractions, rw.molar_fractions, gt.formula)
    
    # Fe3+/sum(Fe) = (2.*andr)/(3.*alm + 2.*andr)
    ferric_fraction[i] = 2.*gt.molar_fractions[3]/(3.*gt.molar_fractions[1] + 2.*gt.molar_fractions[3])
    majorite_fraction[i] = gt.molar_fractions[4]

    
    redox_iron_amount[i] = sol['c'][0] # n*y
    y = (1./X_S) - x_FeS_shallow
    n = redox_iron_amount[i] / y
    sulphur_amount = n

    bulk = copy.deepcopy(c)
    bulk['Fe'] = bulk['Fe'] + n*x_FeS_shallow
    bulk['S'] = sulphur_amount
    
    sulphur = {'S': sulphur_amount}
    S_ppm[i] = burnman.processchemistry.formula_mass(sulphur, burnman.processchemistry.read_masses())/burnman.processchemistry.formula_mass(bulk, burnman.processchemistry.read_masses())*1000000.
    print(S_ppm[i])
    
plt.plot(X_Ss, ferric_fraction)
plt.plot(X_Ss, majorite_fraction)
plt.plot(X_Ss, wt_percent_S)
plt.xlabel('S (atom fraction in core)')
plt.ylabel('Fe3+/sumFe (garnet)')
plt.show()


np.savetxt(fname='sulphur_in_mantle.dat', X=zip(*[X_Ss, ferric_fraction, majorite_fraction, S_ppm]), header='X_S (core, atomic fraction), Fe3+/sumFe (gt), majorite_fraction (gt), S (mantle, ppm)')

