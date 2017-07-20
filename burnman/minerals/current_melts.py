# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2017 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
Simple silicate and oxide melts
"""

from __future__ import absolute_import

from ..mineral import Mineral
from ..solidsolution import SolidSolution
from ..solutionmodel import *
from ..processchemistry import dictionarize_formula, formula_mass
from .HP_2011_ds62 import crst, pren, en, q
from .DKS_2013_solids import periclase as periclase_DKS
from .SLB_2011 import stishovite, seifertite, coesite, quartz, enstatite, periclase


qtz_HP = q()
crst_HP = crst()
qtz_SLB = quartz()


pren_HP = pren()
en_HP = en()
en_SLB = enstatite()

per_SLB = periclase()
per_DKS = periclase_DKS()

'''
# convert HP crst to SLB standard state for SiO2
qtz_HP.set_state(1.e5, 300.)
qtz_SLB.set_state(1.e5, 300.)
crst_HP.params['H_0'] = crst_HP.params['H_0'] - qtz_HP.gibbs + qtz_SLB.gibbs
'''

# convert HP pren to SLB standard for Mg2Si2O6
en_HP.set_state(1.e5, 300.)
en_SLB.set_state(1.e5, 300.)
pren_HP.params['H_0'] = pren_HP.params['H_0'] - en_HP.gibbs + en_SLB.gibbs

# convert DKS per to SLB standard for MgO
per_DKS.set_state(1.e5, 300.)
per_SLB.set_state(1.e5, 300.)
per_DKS.params['E_0'] = per_DKS.params['E_0'] - per_DKS.internal_energy + per_SLB.internal_energy

class SiO2_liquid (Mineral):
    def __init__(self):
        P_0 = 1.e5
        T_0 = 1999.
        crst_HP.set_state(P_0, T_0)
        F_0 = crst_HP.gibbs
        S_0 = crst_HP.S + 4.46
        V_0 = crst_HP.V
        Kprime_inf = 3.5
        grueneisen_inf = 0.5*Kprime_inf - 1./6.
        formula = 'SiO2'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'SiO2 liquid',
            'formula': formula,
            'equation_of_state': 'simple_melt',
            'V_0': V_0,
            'K_0': 13.2e9,
            'Kprime_0': 5.5,
            'Kprime_inf': Kprime_inf,
            'G_0': 0.e9, # melt
            'Gprime_inf': 1.,
            'grueneisen_0': 0.05,
            'grueneisen_inf': grueneisen_inf,
            'q_0': -1.6,
            'C_v': 83.,
            'P_0': P_0,
            'T_0': T_0,
            'F_0': F_0,
            'S_0': S_0,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}
        Mineral.__init__(self)


class MgO_liquid (Mineral):
    def __init__(self):
        P_0 = 1.e5
        T_0 = 3098. 
        per_DKS.set_state(P_0, T_0)
        F_0 = per_DKS.gibbs
        S_fus = 30. # de Koker et al.
        dTdP_fus = 125.e-9
        S_0 = per_DKS.S + S_fus
        V_0 = per_DKS.V + S_fus*dTdP_fus
        Kprime_inf = 3.2
        grueneisen_inf = 0.5*Kprime_inf - 1./6.
        formula = 'MgO'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'MgO liquid',
            'formula': formula,
            'equation_of_state': 'simple_melt',
            'V_0': V_0,
            'K_0': 40.e9,
            'Kprime_0': 4.75, # de Koker et al.
            'Kprime_inf': Kprime_inf,
            'G_0': 0.e9, # melt
            'Gprime_inf': 1.,
            'grueneisen_0': 0.65, # estimate from de Koker et al.
            'grueneisen_inf': grueneisen_inf,
            'q_0': -1.9, # estimate from de Koker et al.
            'C_v': 60., # estimate from de Koker et al.
            'P_0': P_0,
            'T_0': T_0,
            'F_0': F_0,
            'S_0': S_0,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}
        Mineral.__init__(self)




class Mg2Si2O6_liquid (Mineral):
    def __init__(self):
        P_0 = 1.e5
        T_0 = 1834. 
        formula = 'Mg2Si2O6'
        formula = dictionarize_formula(formula)
        molar_mass = formula_mass(formula)
        pren_HP.set_state(P_0, T_0)
        F_0 = pren_HP.gibbs
        S_fus = 76.7 # 76.7 +/- 6.6 from Richet
        dTdP_fus = 110.e-9
        S_0 = pren_HP.S + S_fus
        V_0 = pren_HP.V + S_fus*dTdP_fus
        print(V_0)
        K_0 = 2860.*2860.*molar_mass/V_0*1.15 # velocity from Bass, 1995 (AGU Monograph on Mineral physics and crystallography: a handbook of physical constants)
        Kprime_inf = 3.2
        grueneisen_inf = 0.5*Kprime_inf - 1./6.
        self.params = {
            'name': 'Mg2Si2O6 liquid',
            'formula': formula,
            'equation_of_state': 'simple_melt',
            'V_0': V_0,
            'K_0': K_0,
            'Kprime_0': 5.5, 
            'Kprime_inf': Kprime_inf,
            'G_0': 0.e9, # melt
            'Gprime_inf': 1.,
            'grueneisen_0': 1.1,
            'grueneisen_inf': grueneisen_inf,
            'q_0': 1.,
            'C_v': 110.*2., # JANAF Cp is 146.4*2.
            'P_0': P_0,
            'T_0': T_0,
            'F_0': F_0,
            'S_0': S_0,
            'n': sum(formula.values()),
            'molar_mass': molar_mass}
        Mineral.__init__(self)

