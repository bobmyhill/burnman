# This python file takes the eutectic P, T parameterisation of Baron et al., 2016
# and calculates the activities of MgO and SiO2 in the liquid
# (on the basis of some parameterisation of the liquid endmembers.

import numpy as np
from scipy.optimize import fsolve, brentq, root
import matplotlib.pyplot as plt


import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    DKS_2013_liquids, \
    DKS_2013_solids, \
    SLB_2011, \
    HP_2011_ds62
from burnman import constants
from burnman.chemicalpotentials import chemical_potentials
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

######

class FMS_solution():
    def __init__(self, molar_fractions=None):
        self.name = 'MgO-SiO2 solution'
        self.lmda = 1.43
        self.WA = lambda P, T: 80000. - T*70. # decent fit at 80 GPa, 3600 K
        self.WB = lambda P, T: -245000. - T*25. # decent fit at 80 GPa, 3600 K

    def _get_properties(self, P, T):

        

        X = self.molar_fractions[2]
        Y = X/(X + self.lmda*(1. - X))
        G_ideal = burnman.constants.gas_constant*T*(np.sum([xi * np.log(xi) for xi in self.molar_fractions if xi > 1.e-12]))
        self.delta_G = G_ideal + self.WA(P, T)*Y*Y*(1. - Y) + self.WB(P, T)*Y*(1. - Y)*(1. - Y) 
        self.delta_S = (self.WA(P, T-0.5) - self.WA(P, T+0.5))*Y*Y*(1. - Y) + (self.WB(P, T-0.5) - self.WB(P, T+0.5))*Y*(1. - Y)*(1. - Y)  - G_ideal/T
        self.delta_H = self.delta_G + T*self.delta_S
    
        return 1

    def _unit_vector_length(self, v):
        length = np.sqrt(np.sum([ vi*vi for vi in v ]))
        return v/length, length
    
    def set_state(self, P, T):

        molar_fractions = self.molar_fractions

        # Find partial gibbs
        # First, find vector towards MgO, FeO and SiO2
        dX = 0.001
        
        dB, XB = self._unit_vector_length(np.array([0., 1., 0.]) - self.molar_fractions)
        dB = dB*dX
        self.molar_fractions = molar_fractions + dB
        sol = self._get_properties(P, T)
        GB = self.delta_G

        dC, XC = self._unit_vector_length(np.array([0., 0., 1.]) - self.molar_fractions)
        dC = dC*dX
        self.molar_fractions = molar_fractions + dC
        sol = self._get_properties(P, T)
        GC = self.delta_G


        self.molar_fractions = molar_fractions
        sol = self._get_properties(P, T)
        G0 = self.delta_G

        self.partial_gibbs_excesses = np.array( [ 0.,
                                                  G0 + (GB - G0)/dX*XB,
                                                  G0 + (GC - G0)/dX*XC ] )

        MgO_liq.set_state(P, T)
        SiO2_liq.set_state(P, T)

        self.partial_gibbs = ( np.array( [ 0.,
                                           MgO_liq.gibbs,
                                           SiO2_liq.gibbs ] ) +
                               self.partial_gibbs_excesses )
        return sol


FMS = FMS_solution()

######


Simon_Glatzel = lambda Tr, Pr, A, C, P: Tr*np.power(1. + (P - Pr)/A, 1./C)


pressures = np.linspace(24.e9, 136.e9, 12)
per_pv_temperatures = Simon_Glatzel(2705., 24., 19.156, 3.7796, pressures/1.e9)
pv_stv_temperatures = Simon_Glatzel(2605., 24., 29.892, 3.677, pressures/1.e9)


model = 'DKS'
    
if model == 'DKS_mod':
    per = DKS_2013_solids.periclase()

    per.set_state(1.e5, 3098.)
    G0 = per.gibbs
    
    per = SLB_2011.periclase()
    per.params['q_0'] = 0.15
    per.params['grueneisen_0'] = 1.4 # not a perfect match of both HP heat capacity (~1.4) and HP volume (~1.6), but not bad...
  
    per.set_state(1.e5, 3098.)
    per.params['F_0'] += G0 - per.gibbs
  

    
    stv = DKS_2013_solids.stishovite()
    mpv = DKS_2013_solids.perovskite()
    mpv.property_modifiers = [['linear', {'delta_E': 15000., 'delta_S': -14.5, 'delta_V': 0.}]]
    
    MgO_liq =  DKS_2013_liquids.MgO_liquid()
    SiO2_liq = DKS_2013_liquids.SiO2_liquid()
    Tr = 3073.
    Pr = 13.7e9
    Pr, Tr, dG, dS, dV = (Pr, Tr, 0., 0., 1.e-7)
    SiO2_liq.property_modifiers = [['linear', {'delta_E': dG + Tr*dS - Pr*dV, 'delta_S': dS, 'delta_V': dV}]]

    
    int_liq = DKS_2013_liquids.Mg3Si2O7_liquid()

    
    FMS.WA = lambda P, T: -180000. - 0*(T - 3600.) # decent fit at 80 GPa, 3600 K
    FMS.WB = lambda P, T: -320000. - 0*(T - 3600.) # decent fit at 80 GPa, 3600 K
    
if model == 'DKS':
    per = DKS_2013_solids.periclase()
    stv = DKS_2013_solids.stishovite()
    mpv = DKS_2013_solids.perovskite()
    
    MgO_liq =  DKS_2013_liquids.MgO_liquid()
    SiO2_liq = DKS_2013_liquids.SiO2_liquid()
    int_liq = DKS_2013_liquids.Mg3Si2O7_liquid()

    
    FMS.WA = lambda P, T: 80000. - T*70. # decent fit at 80 GPa, 3600 K
    FMS.WB = lambda P, T: -245000. - T*25. # decent fit at 80 GPa, 3600 K
        
if model == 'new':


    FMS.lmda = 1.
    FMS.WA = lambda P, T: -300000.  # decent fit at 80 GPa, 3600 K
    FMS.WB = lambda P, T: -200000. # decent fit at 80 GPa, 3600 K
    
    # MgO melting curve
    per = SLB_2011.periclase()
    #per.params['F_0'] += -40036.738
    per.params['q_0'] = 0.15
    per.params['grueneisen_0'] = 1.6 # not a perfect match of both HP heat capacity (~1.4) and HP volume (~1.6), but not bad...
    
    MgO_liq = DKS_2013_liquids.MgO_liquid()

    Pr = 1.e5
    Tr = 3098.
    per.set_state(Pr, Tr)
    MgO_liq.set_state(Pr, Tr)
    Pr, Tr, dG, dS, dV = (Pr, Tr, per.gibbs - MgO_liq.gibbs, per.S - MgO_liq.S + (169.595 - 144.532), 0.)
    MgO_liq.property_modifiers = [['linear', {'delta_E': dG + Tr*dS - Pr*dV, 'delta_S': dS, 'delta_V': dV}]]
    
    
    # SiO2 melting curve
    stv = SLB_2011.stishovite()
    coe = SLB_2011.coesite()
    qtz = SLB_2011.quartz()
    SiO2_liq = DKS_2013_liquids.SiO2_liquid()
    
    Tr = 3073.
    Pr = 13.7e9
    coe.set_state(Pr, Tr)
    SiO2_liq.set_state(Pr, Tr)
    Pr, Tr, dG, dS, dV = (Pr, Tr, coe.gibbs - SiO2_liq.gibbs, coe.S - SiO2_liq.S + 30., 4.e-7)
    SiO2_liq.property_modifiers = [['linear', {'delta_E': dG + Tr*dS - Pr*dV, 'delta_S': dS, 'delta_V': dV}]]


    qtz.property_modifiers = [['linear', {'delta_E': 1.5e3, 'delta_S': 0., 'delta_V': 0.}]]
    coe.property_modifiers = [['linear', {'delta_E': 0.e3, 'delta_S': 0., 'delta_V': 0.}]]

    
    Tr = 3023.
    Pr = 13.e9
    Pr, Tr, dG, dS, dV = (Pr, Tr, -5.2e3, 0., 0.e-7)
    stv.property_modifiers = [['linear', {'delta_E': dG + Tr*dS - Pr*dV, 'delta_S': dS, 'delta_V': dV}]]
    

    mpv = SLB_2011.mg_perovskite()
    mpv.property_modifiers = [['linear', {'delta_E': -2.6e3, 'delta_S': 0., 'delta_V': 0.}]]
    int_liq = DKS_2013_liquids.Mg3Si2O7_liquid()

    int_liq.property_modifiers = [['linear', {'delta_E': 3.*MgO_liq.property_modifiers[0][1]['delta_E'] + 2.*SiO2_liq.property_modifiers[0][1]['delta_E'],
                                              'delta_S': 3.*MgO_liq.property_modifiers[0][1]['delta_S'] + 2.*SiO2_liq.property_modifiers[0][1]['delta_S'],
                                              'delta_V': 3.*MgO_liq.property_modifiers[0][1]['delta_V'] + 2.*SiO2_liq.property_modifiers[0][1]['delta_V']}]]


    

# Gibbs figure
P =  80.e9
T = Simon_Glatzel(2605., 24., 29.892, 3.677, P/1.e9)
for m in [MgO_liq, SiO2_liq, per, mpv, stv]:
    m.set_state(P, T)

S = SiO2_liq.gibbs
M = MgO_liq.gibbs
PV = 0.5*(S + M)
plt.plot([1.], [SiO2_liq.gibbs - S], marker='o')
plt.plot([0.], [MgO_liq.gibbs - M], marker='o')
plt.plot([0.5, 1.0], [mpv.gibbs/2. - PV, stv.gibbs - S])
plt.plot([0.0, 0.5], [per.gibbs - M, mpv.gibbs/2. - PV])


Xs = np.linspace(0.0001, 0.9999, 101)
Gs = np.empty_like(Xs)
for i, X in enumerate(Xs):
    FMS.molar_fractions = [0., 1. - X, X]
    FMS.set_state(P, T)
    Gs[i] = FMS.delta_G

plt.plot(Xs, Gs)
plt.show()


# Main figure
eutectics = [['per-pv', 'red', pressures, per_pv_temperatures, [0.40, 0.45, 0.50], per, mpv, MgO_liq, int_liq, SiO2_liq],
             ['pv-stv', 'blue', pressures, per_pv_temperatures, [0.55, 0.6, 0.65, 0.7, 0.75], mpv, stv, MgO_liq, int_liq, SiO2_liq]]

plt.subplot(133)
plt.plot(pressures/1.e9, per_pv_temperatures, color='red', label='per-pv eutectic')
plt.plot(pressures/1.e9, pv_stv_temperatures, color='blue', label='pv-stv eutectic')


def eqm_composition(X, P, T, liq, c_name):
    liq.molar_fractions = [0., 1. - X, X]
    liq.set_state(P, T)

    if c_name == 'stv':
        SiO2_liq.set_state(P, T)
        stv.set_state(P, T)
        return (stv.gibbs - SiO2_liq.gibbs) - liq.partial_gibbs_excesses[2]
    elif c_name == 'per':
        MgO_liq.set_state(P, T)
        per.set_state(P, T)
        return (per.gibbs - MgO_liq.gibbs) - liq.partial_gibbs_excesses[1]
    
def eqm_TX(args, P, liq, m1, m2):
    T, X = args
    liq.molar_fractions = [0., 1. - X, X]

    for m in [liq, m1, m2, MgO_liq, SiO2_liq]:
        m.set_state(P, T)
    mu = chemical_potentials([m1, m2], [{'Mg': 1., 'O': 1.}, {'Si': 1., 'O': 2.}])
    return [(mu[0] - MgO_liq.gibbs) - liq.partial_gibbs_excesses[1],
            (mu[1] - SiO2_liq.gibbs) - liq.partial_gibbs_excesses[2]]
    
for eutectic in eutectics:
    name, color, Ps, Ts, Xs, m1, m2, l1, l2, l3 = eutectic

    
    # Find the activities along the experimental eutectic
    activities = []
    for (P, T) in zip(*[Ps, Ts]):
        for m in [m1, m2, l1, l2, l3]:
            m.set_state(P, T)
        mu = chemical_potentials([m1, m2], [{'Mg': 1., 'O': 1.}, {'Mg': 3./5., 'Si': 2./5., 'O': 7./5.}, {'Si': 1., 'O': 2.}])
        liq_gibbs = [l1.gibbs, l2.gibbs/5., l3.gibbs]
        activities.append(np.exp((mu - liq_gibbs)/(burnman.constants.gas_constant*T)))

        
    # Find the activities at given liquid compositions
    model_activities = []
    for X in Xs:
        mas = []
        for (P, T) in zip(*[Ps, Ts]):
            FMS.molar_fractions = [0., 1. - X, X]
            for m in [FMS, m1, m2, l1, l2, l3]:
                m.set_state(P, T)
            partials = FMS.partial_gibbs_excesses
            
            f = 0.4
            FMS.molar_fractions = [0., 1. - f, f]
            FMS.set_state(P, T)
            dG_int = (1. - f)*partials[1] + f*partials[2] - FMS.delta_G
            mas.append(np.exp(np.array( [ partials[1],
                                          dG_int,
                                          partials[2]])/(burnman.constants.gas_constant*T)))
        model_activities.append(mas)
        

    # Find the activities at periclase or stishovite saturation
    eqm_model_activities = []
    eqm_model_temperatures = []
    for (P, T) in zip(*[Ps, Ts]):
        if name == 'per-pv':
            name_solid = 'per'
            #sol = brentq(eqm_composition, 0.1, 0.5, args=(P, T, FMS, name_solid))
            sol = fsolve(eqm_TX, [4000., 0.4], args=(P, FMS, m1, m2))
        else:
            name_solid = 'stv'
            #sol = brentq(eqm_composition, 0.4, 0.99, args=(P, T, FMS, name_solid))
            sol = fsolve(eqm_TX, [4000., 0.7], args=(P, FMS, m1, m2))
        partials = FMS.partial_gibbs_excesses

        FMS.molar_fractions = [0., 1. - f, f]
        FMS.set_state(P, T)
        dG_int = (1. - f)*partials[1] + f*partials[2] - FMS.delta_G
        eqm_model_activities.append(np.exp(np.array( [ partials[1],
                                                       dG_int,
                                                       partials[2]])/(burnman.constants.gas_constant*T)))
        eqm_model_temperatures.append(sol[0])
                   
        #print model_activities
    plt.subplot(131)
    plt.plot(Ps/1.e9, np.array(activities).T[0], color=color, label='a(MgO,liq) at the experimental '+name+' eutectic')
    for i, mas in enumerate(model_activities):
        plt.plot(Ps/1.e9, np.array(mas).T[0], linestyle=':', color=color)#, label='model a(MgO,liq) at X = '+str(Xs[i]))
    plt.plot(Ps/1.e9, np.array(eqm_model_activities).T[0], linestyle='--', color=color, label='a(MgO, liq) at the model '+name+' eutectic')
    plt.legend(loc='lower right')
    #plt.subplot(132)
    #plt.plot(Ps/1.e9, np.array(activities).T[1], label=name, color=color)
    #for i, mas in enumerate(model_activities):
    #    plt.plot(Ps/1.e9, np.array(mas).T[1], linestyle=':', color=color)# label=name)
    #plt.plot(Ps/1.e9, np.array(eqm_model_activities).T[1], label=name, linestyle='--', color=color)
    plt.subplot(132)
    plt.plot(Ps/1.e9, np.array(activities).T[2], color=color, label='a(SiO2,liq) at the experimental '+name+' eutectic')
    for i, mas in enumerate(model_activities):
        plt.plot(Ps/1.e9, np.array(mas).T[2], linestyle=':', color=color)#, label='model a(SiO2,liq) at X = '+str(Xs[i]))
    plt.plot(Ps/1.e9, np.array(eqm_model_activities).T[2], linestyle='--', color=color, label='a(SiO2, liq) at the model '+name+' eutectic')
    plt.legend(loc='lower right')
    plt.subplot(133)
    plt.plot(Ps/1.e9, np.array(eqm_model_temperatures), linestyle='--', color=color, label='temperature at the model '+name+' eutectic')


plt.subplot(131)
plt.ylabel('a(MgO)')
plt.xlabel('X SiO2 (mol %)')
#plt.subplot(132)
#plt.title('a(intermediate)')


plt.subplot(132)
plt.ylabel('a(SiO2)')
plt.xlabel('X SiO2 (mol %)')


'''
for P in [30.e9, 60.e9, 90.e9, 120.e9, 150.e9]:
    temperatures = np.linspace(3000., 6000., 21)
    S_MgO = np.empty_like(temperatures)
    S_SiO2 = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        MgO_liq.set_state(P, T)
        SiO2_liq.set_state(P, T)
        per.set_state(P, T)
        stv.set_state(P, T)
        S_MgO[i] = (MgO_liq.S - per.S)/burnman.constants.gas_constant
        S_SiO2[i] = (SiO2_liq.S - stv.S)/burnman.constants.gas_constant
        #S_MgO[i] = (MgO_liq.heat_capacity_v - per.heat_capacity_v) / per.heat_capacity_v
        #S_SiO2[i] = (SiO2_liq.heat_capacity_v - stv.heat_capacity_v) / stv.heat_capacity_v

    plt.subplot(234)
    plt.plot(temperatures, S_MgO)
    plt.subplot(235)
    plt.plot(temperatures, S_SiO2)
'''

per2 = DKS_2013_solids.periclase()
MgO_liq2 = DKS_2013_liquids.MgO_liquid()
stv2 = DKS_2013_solids.stishovite()
SiO2_liq2 = DKS_2013_liquids.SiO2_liquid()
    
plt.subplot(133)
melting_phases = [['MgO melting', '-', per, MgO_liq],
                  ['SiO2 melting', '-', stv, SiO2_liq],
                  ['MgO melting (DKS)', ':', per2, MgO_liq2],
                  ['SiO2 melting (DKS)', ':', stv2, SiO2_liq2]]
temperatures = np.empty_like(pressures)
for (name, linestyle, solid, liquid) in melting_phases:
    Tguess = 3000.
    for i, P in enumerate(pressures):
        temperatures[i] = burnman.tools.equilibrium_temperature([solid, liquid], [1.0, -1.0], P, Tguess)
        Tguess = temperatures[i]
    plt.plot(pressures/1.e9, temperatures, label=name, linestyle=linestyle)

plt.xlabel('Pressure (GPa)')
plt.ylabel('Temperature (K)')




plt.legend(loc='upper left')
plt.show()
