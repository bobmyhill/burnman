import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()

def make_intermediate(mbr0, mbr1, params):
    H_ex, S_ex, Sconf, V_ex, K_ex, a_ex = params

    name = mbr0.params['name']+mbr1.params['name']

    P_0 = mbr0.params['P_0']
    T_0 = mbr0.params['T_0']
    T_einstein = (mbr0.params['T_einstein'] + mbr1.params['T_einstein'])/2.

    formula={}
    formula.update(mbr0.params['formula'])
    for key in mbr1.params['formula']:
        if key in formula:
            formula[key] += mbr1.params['formula'][key]
        else:
            formula.update({key: mbr1.params['formula'][key]})

    for key in formula:
        formula[key] /= 2.

    # Standard conditions are those for an ideal solid solution
    H_0 = (mbr0.params['H_0'] + mbr1.params['H_0'])*0.5 + H_ex
    S_0 = (mbr0.params['S_0'] + mbr1.params['S_0'])*0.5 + Sconf + S_ex

    V_ideal = (mbr0.params['V_0'] + mbr1.params['V_0'])*0.5

    V_0 = V_ideal + V_ex
    K_0 = V_ideal / (0.5*(mbr0.params['V_0']/mbr0.params['K_0'] \
                              + mbr1.params['V_0']/mbr1.params['K_0'])) + K_ex
    a_0 = 0.5*(mbr0.params['a_0']*mbr0.params['V_0'] \
        + mbr1.params['a_0']*mbr1.params['V_0'])/V_ideal + a_ex
    
    Kprime_0 = V_0*2./(mbr0.params['V_0']/(mbr0.params['Kprime_0'] + 1.) \
                       + mbr1.params['V_0']/(mbr1.params['Kprime_0'] + 1.)) \
                       - 1.

    Cp_0 = [(mbr0.params['Cp'][0] + mbr1.params['Cp'][0])*0.5,
               (mbr0.params['Cp'][1] + mbr1.params['Cp'][1])*0.5,
               (mbr0.params['Cp'][2] + mbr1.params['Cp'][2])*0.5,
               (mbr0.params['Cp'][3] + mbr1.params['Cp'][3])*0.5] 

    class intermediate (burnman.Mineral):
        def __init__(self):
            self.params = {
                'name': name,
                'formula': formula,
                'equation_of_state': 'hp_tmt',
                'T_0': T_0,
                'P_0': P_0,
                'T_einstein': T_einstein,
                'H_0': H_0,
                'S_0': S_0,
                'V_0': V_0,
                'Cp': Cp_0,
                'a_0': a_0, 
                'K_0': K_0,
                'Kprime_0': Kprime_0 ,
                'Kdprime_0': -Kprime_0/K_0 ,
                'n': sum(formula.values()),
                'molar_mass': formula_mass(formula, atomic_masses)}
            burnman.Mineral.__init__(self)

    return intermediate()
