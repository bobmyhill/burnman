from __future__ import absolute_import

# Here we import standard python modules that are required for
# usage of BurnMan.  In particular, numpy is used for handling
# numerical arrays and mathematical operations on them, and
# matplotlib is used for generating plots of results of calculations
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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
py = minerals.SLB_2011.pyrope()
alm = minerals.SLB_2011.almandine()
gr = minerals.SLB_2011.grossular()
andr = andradite()
mg_maj = minerals.SLB_2011.mg_majorite()
jd_maj = minerals.SLB_2011.jd_majorite()
mrw = minerals.SLB_2011.mg_ringwoodite()
frw = minerals.SLB_2011.fe_ringwoodite()

HP_py = minerals.HP_2011_ds62.py()
HP_alm = minerals.HP_2011_ds62.alm()
HP_gr = minerals.HP_2011_ds62.gr()
HP_andr = minerals.HP_2011_ds62.andr()
HP_maj = minerals.HP_2011_ds62.maj()
HP_mrw = minerals.HP_2011_ds62.mrw()
HP_frw = minerals.HP_2011_ds62.frw()


mineral_pairs = [[py, HP_py], [alm, HP_alm], [gr, HP_gr], [andr, HP_andr], [mg_maj, HP_maj], [mrw, HP_mrw], [frw, HP_frw]]

P = 1.e5
T = 300.
for pair in mineral_pairs:
    m1, m2 = pair
    m1.set_state(P, T)
    m2.set_state(P, T)
    m1.params['F_0'] = m1.params['F_0'] + (m2.gibbs - m1.gibbs)



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
        self.type = 'symmetric'
        self.endmembers = [[py, '[Mg]3[Al][Al]Si3O12'],
                           [alm, '[Fe]3[Al][Al]Si3O12'],
                           [gr, '[Ca]3[Al][Al]Si3O12'],
                           [andr, '[Ca]3[Fe][Fe]Si3O12'],
                           [mg_maj, '[Mg]3[Mg][Si]Si3O12'],
                           [jd_maj, '[Na2/3Al1/3]3[Al][Si]Si3O12']]

        
        self.energy_interaction = [[0.0, 30.e3, 0.0, 21.20278e3, 0.0],
                                   [0.0, 0.0, 0.0, 0.0],
                                   [0.0, 57.77596e3, 0.0],
                                   [57.77596e3, 0.0],
                                   [0.0]]

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
        
gt.set_composition([1.0, 0., 0., 0., 0., 0.])
gt.set_state(1.e9, 300.)
print(gt.molar_mass)


# Fe-S model from Waldner and Pelton, 2005
'''
def Gxs_Fe_S(n_FeFe, n_FeS, n_SS, T):
    n_total = n_FeFe + n_SS + n_FeS # [p. 26; WP2005]
    X_FeFe = n_FeFe/n_total # [p. 26; WP2005]
    X_SS = n_SS/n_total # [p. 26; WP2005]
    X_FeS = n_FeS/n_total # [p. 26; WP2005]
    
    Z_FeS = 2. # [p. 30; WP2005]
    Z_SFe = 2. # [p. 30; WP2005]
    Z_FeFe = 6. # [p. 30; WP2005]
    Z_SS = 6. # [p. 30; WP2005]
    
    Z_Fe = 1./((2.*n_FeFe/(2.*n_FeFe + n_FeS))/Z_FeFe + (n_FeS/(2.*n_FeFe + n_FeS))/Z_FeS) # [eq. 5; WP2005]
    Z_S = 1./((2.*n_SS/(2.*n_SS + n_FeS))/Z_SS + (n_FeS/(2.*n_SS + n_FeS))/Z_SFe) # [eq. 6; WP2005]

    n_Fe = (2.*n_FeFe + n_FeS)/Z_Fe # [eq. 2; PDERD2000]
    n_S = (2.*n_SS + n_FeS)/Z_S # [eq. 3; PDERD2000]
        
    X_Fe = n_Fe/(n_Fe + n_S) # [eq. 5; PDERD2000]
    X_S = n_S/(n_Fe + n_S) # [eq. 5; PDERD2000]
    
    Y_Fe = X_FeFe + X_FeS/2. # [eq. 7; PDERD2000]
    Y_S = X_S + X_FeS/2. # [eq. 8; PDERD2000]

    
    S_config = -burnman.constants.gas_constant*(n_Fe*np.log(X_Fe) + n_S*np.log(X_S)
                                                + n_FeFe*np.log(X_FeFe/(Y_Fe*Y_Fe)) + n_SS*np.log(X_SS/(Y_S*Y_S))
                                                + n_FeS*np.log(X_FeS/(2.*Y_Fe*Y_S))) # [eq. 10; PDERD2000]
    
    Delta_g_FeS = -104888.1 + 0.338*T \
      + (35043.32 - 9.880*T)*np.power(X_FeFe, 1.) \
      + (23972.27)*np.power(X_FeFe, 2.) \
      + (30436.82)*np.power(X_FeFe, 3.) \
      + (8626.26)*np.power(X_SS, 1.) \
      + (72954.29 - 26.178*T)*np.power(X_SS, 2.) \
      + (25106.)*np.power(X_SS, 4.) # [Table 1; WP2005]

    return X_S, (-T*S_config + (n_FeS/2.)*Delta_g_FeS)/(n_Fe + n_S) # [Eq. 3; WP2005] normalised to 1 atom

# We need to solve for the minimum excess energy given a particular value of S
# Try a nested solve, with the S value being found in the inner solve and the minimization in the outer.

from scipy.optimize import fsolve, minimize

def delta_XS(n_FeFe, n_FeS, T, X_S):
    n_SS = 1. - n_FeFe[0] - n_FeS
    sol = Gxs_Fe_S(n_FeFe[0], n_FeS, n_SS, T)
    return sol[0] - X_S

def minimize_G(T, X_S):
    n_FeSs = np.linspace(0.0001, 0.9999, 401)
    G = 10000000.
    X_FeS = 0.0001
    for i, n_FeS in enumerate(n_FeSs):
        n_FeFe = fsolve(delta_XS, [0.5], args=(n_FeS, T, X_S))[0]
        n_SS = 1. - n_FeFe - n_FeS
        sol = Gxs_Fe_S(n_FeFe, n_FeS, n_SS, T)
        if sol[1] < G:
            G = sol[1]
            n_total = n_FeFe + n_FeS + n_SS
            X_FeS = n_FeS/n_total
            X_FeFe = n_FeFe/n_total
            X_SS = n_SS/n_total
            
    return G, X_FeFe, X_FeS, X_SS

XSs = np.linspace(0.01, 0.999, 101)
for T in [1200., 1600., 2000.]:
    Gs = np.array([minimize_G(T, X_S) for X_S in XSs])
    plt.plot(XSs, Gs.T[0].T, label=str(T))
    
plt.xlim(0., 1.)
plt.xlabel('S/(Fe+S)')
plt.legend(loc='lower right')
plt.show()
'''

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


                           
def sulfide_silicate_equilibrium(args, P, T, X_S_melt, composition):
    n_py, n_alm, n_gr, n_andr, n_maj, n_Fe_metal, n_mg_rw, n_fe_rw = args

    n_jd_maj = 0.5*composition['Na']
    
    n_gt = n_py + n_alm + n_gr + n_andr + n_maj + n_jd_maj
    n_rw = n_mg_rw + n_fe_rw

    X_gt = [n_py/n_gt,
            n_alm/n_gt,
            n_gr/n_gt,
            n_andr/n_gt,
            n_maj/n_gt,
            n_jd_maj/n_gt]

    X_rw = [n_mg_rw/n_rw,
            n_fe_rw/n_rw]
    
    gt.set_composition(X_gt)
    rw.set_composition(X_rw)

    gt.set_state(P, T)
    rw.set_state(P, T)
    Fe_liq.set_state(P, T)

    
    # Governing equations
    # 3.Mg2SiO4 + 2.Fe3Al2Si3O12 -> 3.Fe2SiO4 + 2.Mg3Al2Si3O12 
    eqns = [3.*rw.partial_gibbs[0] + gt.partial_gibbs[1]
            - 3.*rw.partial_gibbs[1] - gt.partial_gibbs[0]]


    # Fe + Fe3Fe2Si3O12 -> 3.Fe2SiO4
    Fe_liq_partial_gibbs = Fe_liq.gibbs + burnman.constants.gas_constant*T*np.log(activity_Fe_in_Fe_S(X_S_melt, T))
    sk_partial_gibbs = gt.partial_gibbs[1] + gt.partial_gibbs[3] - gt.partial_gibbs[2] 
    eqns.append(Fe_liq_partial_gibbs + sk_partial_gibbs - 3.*rw.partial_gibbs[1])

    #n_Na = 2.*n_jd_maj
    n_Ca = 3.*n_gr + 3.*n_andr
    n_Fe = 3.*n_alm + 2.*n_andr + 1.*n_Fe_metal + 2.*n_fe_rw
    n_Mg = 3.*n_py + 4.*n_maj + 1.* 2.*n_mg_rw
    n_Al = 2.*(n_py + n_alm + n_gr + n_jd_maj)
    n_Si = 3.*n_gt + 1.*n_rw + 1.*(n_maj + n_jd_maj)
    n_O = 12.*n_gt + 4.*n_rw
    eqns.extend([n_Ca - composition['Ca'],
                 n_Fe - composition['Fe'],
                 n_Mg - composition['Mg'],
                 n_Al - composition['Al'],
                 n_Si - composition['Si'],
                 n_O - composition['O']])
    print(args)
    return eqns


# n_py, n_alm, n_gr, n_andr, n_maj, n_jd_maj, n_Fe, n_mg_rw, n_fe_rw
#guesses = [c['Mg']/2./3., c['Fe']/2./3., c['Ca']/3., 0.05, 0., 0., c['Mg']/2./2., c['Fe']/2./2.]
#fsolve(sulfide_silicate_equilibrium, guesses, args=(20.e9, 2000., 0.3, c))


c = burnman.processchemistry.component_to_atom_fractions({'Na2O': 0.5, 'CaO': 2.0, 'FeO': 18.4, 'MgO': 32.6, 'Al2O3': 2.5, 'SiO2': 44.0}, 'weight')
print(c)
# Assuming no ferric or metallic iron
n_jd_maj = c['Na']/2.
n_gr = c['Ca']/3.
n_py_alm = c['Al']/2. - n_jd_maj - n_gr


n_Si_rw_and_maj = c['Si'] - n_jd_maj*4. - n_py_alm*3. - n_gr*3.
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
print(0., n_py, n_alm, n_gr, 0., n_maj, n_jd_maj, n_mrw, n_frw)

composition = burnman.processchemistry.component_to_atom_fractions({'Mg3Al2Si3O12': n_py,
                                                                    'Fe3Al2Si3O12': n_alm,
                                                                    'Ca3Al2Si3O12': n_gr,
                                                                    'Mg4Si4O12': n_maj,
                                                                    'Na2Al2Si4O12': n_jd_maj,
                                                                    'Mg2SiO4': n_mrw,
                                                                    'Fe2SiO4': n_frw}, 'molar')
print(composition)

'''
m = np.array([0.5/61.9789, 2.0/56.0774, 18.4/71.844, 32.6/40.3044, 2.5/101.96, 44.0/60.08])
m2 = np.array([m[0]*2., m[1], m[2], m[3], m[4]*2., m[5], m[0] + m[1] + m[2] + m[3] + m[4]*3. + m[5]*2.])
m3 = m2/sum(m2)
print(m3)
'''


    
assemblage = burnman.Composite([Fe_liq, gt, rw])
constraints = [['P', 20.e9], ['T', 2000.]]
sol = gibbs_minimizer(composition, assemblage, constraints)
print(sol, gt.molar_fractions, rw.molar_fractions, gt.formula)
