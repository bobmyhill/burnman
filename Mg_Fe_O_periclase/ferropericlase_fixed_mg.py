# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import Mineral, minerals
from burnman.processchemistry import dictionarize_formula, formula_mass

MM = minerals.SLB_2011.periclase()
HH = minerals.SLB_2011.wuestite()

#####################
#   USER PROVIDED   #
#####################


T = 300.

# LOW SPIN FEO
# Solomatova compilation extrapolations/approximations
V0_low_spin = 11.24e-6 - 0.309e-6 
K0_low_spin = 160.2e9 + 25.144e9
Kprime_low_spin = 4.
P_crossover = 82.e9

# ENERGIES OF ORDERING
# Gibbs energies of the reactions A + B -> 2A0.5B0.5 (ordered)
deltaG_HH_LL = -20000. #-(P_crossover - P_start)*deltaV_ref/2. # Holmstrom and Stixrude suggest large negative deltaG. (complete disorder @ 4000 K)
deltaG_HH_MM = 0. # W(MgO,FeO) O'Neill (4160 J/mol) is similar to elastic prediction (4100 J/mol)

# Reciprocal ordered solutions
deltaG_LL_MM = -20000. # ~-20000 fit to experimental data at 300 K? Reduce Mg ordering during the spin transition?
deltaG_MM_LL = deltaG_LL_MM # symmetric solution as for the other ordered compounds


#####################
# END USER PROVIDED #
#####################


# Adjustments for volume change
P_LS = HH.method.pressure(300., V0_low_spin, HH.params)
debye = HH.method._debye_temperature(HH.params['V_0'] / V0_low_spin, HH.params)

HH.set_state(P_LS, 300.)
gr = HH.gr
G = HH.shear_modulus

dP = 1000.
HH.set_state(P_LS+dP, 300.)
q = np.log(HH.gr/gr)/np.log(HH.V/V0_low_spin)
Gprime = (HH.shear_modulus - G)/(dP)

# Low spin wuestite is different to high spin wuestite in several respects:
# Volume and bulk modulus are different
# Debye temperature should be adjusted for the smaller volume
# The excess entropy will be different (as low spin FeO is diamagnetic)

class low_spin_wuestite (Mineral):
    def __init__(self):
        formula = 'FeO'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Low spin wuestite',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': 0.0,
            'V_0': V0_low_spin, 
            'K_0': K0_low_spin,
            'Kprime_0': Kprime_low_spin, 
            'Debye_0': debye, 
            'grueneisen_0': gr,
            'q_0': q,
            'G_0': G,
            'Gprime_0': Gprime,
            'eta_s_0': -0.05731,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}

        self.property_modifiers = [
            ['linear', {'delta_E': 0.0, 'delta_S': 0., 'delta_V': 0.0}]]

        Mineral.__init__(self)


LL = low_spin_wuestite()

HH.set_state(P_crossover, 300.)
LL.set_state(P_crossover, 300.)

LL.property_modifiers[0][1]['delta_E'] = HH.gibbs - LL.gibbs

'''
pressures = np.linspace(30.e5, 120.e9, 101)
temperatures = [1.]*len(pressures)
plt.plot(pressures/1.e9,
         HH.evaluate(['V'], pressures, temperatures)[0] - LL.evaluate(['V'], pressures, temperatures)[0])

plt.show()
'''

# The next two definitions are for the ordered phases
# Here, we assume convergent ordering
HL = burnman.CombinedMineral([HH, LL],
                     [0.5, 0.5],
                     [deltaG_HH_LL, 0., 0.]) # FeHS_FeLS
HM = burnman.CombinedMineral([HH, MM],
                             [0.5, 0.5],
                             [deltaG_HH_MM, 0., 0.]) # FeHS_FeLS # FeHS_Mg

'''
pressures = np.linspace(30.e9, 120.e9, 101)
temperatures = [1.]*len(pressures)
for m in [HH, HL, LL]:
    plt.plot(pressures/1.e9,
             m.evaluate(['gibbs'], pressures, temperatures)[0] - HL.evaluate(['gibbs'], pressures, temperatures)[0])

plt.show()
'''

# Elastic parameters at ~50 GPa (based on maximum, ~symmetric, no cluster relaxation)
HH_LL_elastic = 13200. # W(FeHS, FeLS) [J/mol] # est 13100 at 1 bar, 13500 at 100 GPa
HH_MM_elastic = 8200. # W(FeHS, Mg) [J/mol] # est ~7100 at 1 bar, ~9100 at 100 GPa
LL_MM_elastic = 400. # W(FeLS, Mg) [J/mol] # est 700 at 1 bar, 0 at 100 GPa

HH_LL = HH_LL_elastic/2. + deltaG_HH_LL*2.
HH_MM = HH_MM_elastic/2. + deltaG_HH_MM*2.
LL_MM = LL_MM_elastic/2. + deltaG_LL_MM*2.

HH_HL = LL_HL = HH_LL_elastic/4. # assume completely elastic
HH_HM = MM_HM = HH_MM_elastic/4. # assume completely elastic
HL_HM = LL_MM_elastic/4. # assume completely elastic


LL_HM = (deltaG_HH_LL - deltaG_HH_MM + deltaG_LL_MM) + LL_HL + HL_HM
MM_HL = (-deltaG_HH_LL + deltaG_HH_MM + deltaG_MM_LL) + MM_HM + HL_HM


LL_ML = 1.0*HL_HM + 1.0*LL_HL - 1.0*LL_HM + 1.0*LL_MM - 1.0*MM_HL + 1.0*MM_HM
LL_LM = HL_HM
MM_ML = HL_HM
MM_LM = 1.0*(HL_HM) + 1.0*(LL_HL) - 1.0*(LL_HM) + 1.0*(LL_MM) - 1.0*(MM_HL) + 1.0*(MM_HM)
print(LL_ML, LL_LM, MM_ML, MM_LM, LL_MM)


W = {'HH': {'LL': HH_LL,
            'MM': HH_MM, 
            'HL': HH_HL,
            'HM': HH_HM,},
     'LL': {'MM': LL_MM,
            'HL': LL_HL,
            'HM': LL_HM},
     'MM': {'HL': MM_HL,
            'HM': MM_HM},
     'HL': {'HM': HL_HM}}

print(W)


class ferropericlase(burnman.SolidSolution):
    def __init__(self, molar_amounts=None):
        self.name = 'ferropericlase'
        self.solution_type = 'symmetric'
        self.endmembers = [[HH, '[Fehs]0.5[Fehs]0.5O'],
                           [LL, '[Fels]0.5[Fels]0.5O'],
                           [MM, '[Mg]0.5[Mg]0.5O'],
                           [HL, '[Fehs]0.5[Fels]0.5O'],
                           [HM, '[Fehs]0.5[Mg]0.5O']]
        self.energy_interaction = [[W['HH']['LL'], W['HH']['MM'], W['HH']['HL'], W['HH']['HM']],
                                   [W['LL']['MM'], W['LL']['HL'], W['LL']['HM']],
                                   [W['MM']['HL'], W['MM']['HM']],
                                   [W['HL']['HM']]]

        burnman.SolidSolution.__init__(self, molar_amounts=molar_amounts)


fper = ferropericlase()


def reaction_affinities(Q, X, P, T, fper):
    # Q0 = (LS[1] - LS[0])
    # Q1 = (Mg[1] - Mg[0])
    # Q2 = HS - LS
    pHL = Q[0]
    pHM = Q[1]
    pMM = 1. - X - Q[1]/2.
    pLL = (X - Q[0] - Q[2])/2.
    pHH = (X - Q[0] - Q[1] + Q[2])/2.
    c = [pHH, pLL, pMM, pHL, pHM]
    fper.set_composition(c)
    fper.set_state(P, T)
    
    eqns = np.array([2.*fper.partial_gibbs[3] - (fper.partial_gibbs[0] + fper.partial_gibbs[1]),
                     2.*fper.partial_gibbs[4] - (fper.partial_gibbs[0] + fper.partial_gibbs[2]),
                     fper.partial_gibbs[0] - fper.partial_gibbs[1]])/(burnman.constants.gas_constant*T)
    return eqns


def gibbs(Q, X, P, T, fper):
    pHL = Q[0]
    pHM = 0.
    pMM = 1. - X
    pLL = (X - Q[0] - Q[1])/2.
    pHH = (X - Q[0] + Q[1])/2.
    c = [pHH, pLL, pMM, pHL, pHM]
    fper.set_composition(c)
    fper.set_state(P, T)

    MM.set_state(P, T)
    HH.set_state(P, T)

    return (fper.gibbs - (MM.gibbs*(1. - X) + HH.gibbs*X))


from scipy.optimize import fsolve, minimize





Xs = np.linspace(0.01, 1.0, 6)
Xs = [0.8, 1.0]
fig = plt.figure()
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]


ax[0].plot([0., 150.], [0.2, 0.2], linestyle='--', color='black')
ax[0].plot([0., 150.], [0.8, 0.8], linestyle='--', color='black')


for X in Xs:
    print(X)
    data = []
    # proportion of ordered FeHSFeLSO, HS - LS iron
    for (guess, Ps) in [([X/2., X], np.linspace(1.e5, 75.e9, 101))]: #, ([X, -X], np.linspace(150.e9, 1.e5, 101))]:
        for i, P in enumerate(Ps):
            sol = minimize(gibbs, guess, args=(X, P, T, fper), bounds=((-X,X),(-X,X))) # 1.e-12 is super slow
            site_fractions = [fper.molar_amounts[2],
                              1. - fper.molar_amounts[1] - fper.molar_amounts[2],
                              fper.molar_amounts[1],
                              fper.molar_amounts[2]+fper.molar_amounts[4],
                              fper.molar_amounts[0],
                              fper.molar_amounts[1] + fper.molar_amounts[3]]
            
            if sol.success == True and np.min(site_fractions) > -1.e-12:
                print(P/1.e9, sol.x)
                guess = sol.x
                #print(P/1.e9, fper.partial_gibbs, '[Mg{0:.2f}FeHS{1:.2f}FeLS{2:.2f}]0.5[Mg{3:.2f}FeHS{4:.2f}FeLS{5:.2f}]0.5O'.format()
                data.append([P,
                             (fper.molar_amounts[0] + (fper.molar_amounts[3] + fper.molar_amounts[4])/2.)/X,
                             fper.molar_amounts[3],
                             fper.molar_amounts[4],
                             fper.V])

    data = np.array(data)
    data = data[np.argsort(data[:,0])]
    pressures, high_spin_fractions, Q0s, Q1s, Vs = data.T
    ax[0].plot(pressures/1.e9, high_spin_fractions, label='Mg$_{{{0:.1f}}}$Fe$_{{{1:.1f}}}$O'.format(1. - X, X))
    ax[1].plot(pressures/1.e9, Q0s, label='Mg$_{{{0:.1f}}}$Fe$_{{{1:.1f}}}$O'.format(1. - X, X))
    ax[2].plot(pressures/1.e9, Q1s, label='Mg$_{{{0:.1f}}}$Fe$_{{{1:.1f}}}$O'.format(1. - X, X))
    ax[3].plot(pressures/1.e9, Vs*1.e6, label='Mg$_{{{0:.1f}}}$Fe$_{{{1:.1f}}}$O'.format(1. - X, X))
        
ax[0].set_ylabel('High spin fraction')
ax[1].set_ylabel('Fe$^{LS}_{M2}$ - Fe$^{LS}_{M1}$')
ax[2].set_ylabel('Mg$_{M2}$ - Mg$_{M1}$ (0 by definition)')
ax[3].set_ylabel('Volume (cm$^3$/mol')
for i in range(4):
    ax[i].grid(True)
    ax[i].legend(loc='best')
    ax[i].set_xlabel('P (GPa)')
plt.show()

