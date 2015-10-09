import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    DKS_2013_liquids, \
    DKS_2013_solids, \
    SLB_2011
from burnman import constants
import numpy as np
from scipy.optimize import fsolve, curve_fit
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class MgO_SiO2_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Subregular MgO-SiO2 liquid'
        self.type='subregular'

        self.endmembers = [[DKS_2013_liquids.MgO_liquid(), '[Mg]O'], 
                           [DKS_2013_liquids.MgSiO3_liquid(), '[Si]MgO3']]

        self.enthalpy_interaction = [[[-108600, -182300.]]]
        self.entropy_interaction   = [[[61.2, 15.5]]]
        self.volume_interaction  = [[[4.32e-7, 1.35e-7]]]

                        
        burnman.SolidSolution.__init__(self, molar_fractions)



liquid=MgO_SiO2_liquid()



phases = [DKS_2013_liquids.MgSiO3_liquid(),
         DKS_2013_liquids.Mg3Si2O7_liquid(),
         DKS_2013_liquids.Mg2SiO4_liquid(),
         DKS_2013_liquids.Mg5SiO7_liquid(),
         DKS_2013_liquids.MgO_liquid()
         ]


# Set up endmembers
MgO_liq = DKS_2013_liquids.MgO_liquid()
MgSiO3_liq = DKS_2013_liquids.MgSiO3_liquid()

# Set up points from which to query de Koker et al. (2013) liquid models
pts = [[25, 3000],
       [25, 4000],
       [25, 5000],
       [25, 6000],
       [50, 3000],
       [50, 4000],
       [50, 5000],
       [50, 6000],
       [75, 3000],
       [75, 4000],
       [75, 5000],
       [75, 6000]]


gibbs_excess_data=[]
for p, t in pts:
    #print 'Pressure (GPa):', p, ", Temperature (K):", t
    pressure=p*1.e9
    temperature=t*1.

    MgO_liq.set_state(pressure, temperature)
    MgSiO3_liq.set_state(pressure, temperature)
    MgO_gibbs = MgO_liq.gibbs
    MgSiO3_gibbs = MgSiO3_liq.gibbs

    MgO_H = MgO_liq.H
    MgSiO3_H = MgSiO3_liq.H

    MgO_S = MgO_liq.S
    MgSiO3_S = MgSiO3_liq.S

    MgO_V = MgO_liq.V
    MgSiO3_V = MgSiO3_liq.V

    MgO_K_T = MgO_liq.K_T
    MgSiO3_K_T = MgSiO3_liq.K_T

    fMgSiO3s=[]
    Gexs=[]
    Hexs=[]
    Sexs=[]
    Vexs=[]
    K_Ts=[]
    K_Texs=[]
    for phase in phases:
        
        nSi = phase.params['formula']['Si']
        nMg = phase.params['formula']['Mg']
        
        nMgSiO3 = nSi
        nMgO = nMg - nSi
        sum_units = nMgSiO3+nMgO
        fMgSiO3=nMgSiO3/sum_units
        

        #print phase.params['name'], fMgSiO3

        phase.set_state(pressure, temperature)
        Gex = phase.gibbs/sum_units - (fMgSiO3*MgSiO3_gibbs + (1.-fMgSiO3)*MgO_gibbs)       
        Hex = phase.H/sum_units - (fMgSiO3*MgSiO3_H + (1.-fMgSiO3)*MgO_H)

        Sex = phase.S/sum_units - (fMgSiO3*MgSiO3_S + (1.-fMgSiO3)*MgO_S)

        Vex = phase.V/sum_units - (fMgSiO3*MgSiO3_V + (1.-fMgSiO3)*MgO_V)

        K_T = phase.K_T
        K_Tex = (phase.K_T - (fMgSiO3*MgSiO3_K_T + (1.-fMgSiO3)*MgO_K_T))/K_T

        # data should be composition, pressure, temperature, gibbs_excess
        gibbs_excess_data.append([pressure, temperature, fMgSiO3, Gex])


        fMgSiO3s.append(fMgSiO3)
        Gexs.append(Gex)
        Hexs.append(Hex)
        Sexs.append(Sex)
        Vexs.append(Vex)
        K_Ts.append(K_T)
        K_Texs.append(K_Tex)


    plt.subplot(231)
    plt.title('Excess Gibbs') 
    plt.plot(fMgSiO3s, Gexs, marker='o', linestyle='None', label=str(p)+' GPa, '+str(t)+' K')
    plt.subplot(232)
    plt.title('Excess Enthalpies') 
    plt.plot(fMgSiO3s, Hexs, marker='o', linestyle='None', label=str(p)+' GPa, '+str(t)+' K')
    plt.subplot(233)
    plt.title('Excess Entropies') 
    plt.plot(fMgSiO3s, Sexs, marker='o', linestyle='None', label=str(p)+' GPa, '+str(t)+' K')
    plt.subplot(234)
    plt.title('Excess Volumes') 
    plt.plot(fMgSiO3s, Vexs, marker='o', linestyle='None', label=str(p)+' GPa, '+str(t)+' K')
    plt.subplot(235)
    plt.title('Fractional excess K_T') 
    plt.plot(fMgSiO3s, K_Texs, marker='o', linestyle='None', label=str(p)+' GPa, '+str(t)+' K')
    plt.subplot(236)
    plt.title('K_T') 
    plt.plot(fMgSiO3s, K_Ts, marker='o', linestyle='None', label=str(p)+' GPa, '+str(t)+' K')

plt.legend(loc='lower right')
plt.show()

gibbs_excess_data = np.array(gibbs_excess_data)
observed_gibbs_excesses = np.array(zip(*gibbs_excess_data)[3])

#################
# START FITTING #
#################

def fit_parameters(data, H0, H1, S0, V0):
    V1 = V0
    S1 = S0
    liquid.enthalpy_interaction = [[[H0, H1]]]
    liquid.entropy_interaction  = [[[S0, S1]]] 
    liquid.volume_interaction   = [[[V0, V1]]]  

    burnman.SolidSolution.__init__(liquid, [0., 1.])

    gibbs_excesses=[]
    for datum in data:
        pressure, temperature, fMgSiO3, Gex = datum
        liquid.set_composition([1.-fMgSiO3, fMgSiO3])
        liquid.set_state(pressure, temperature)
        gibbs_excesses.append(liquid.excess_gibbs)

    return gibbs_excesses


popt, pcov = curve_fit(fit_parameters, gibbs_excess_data, observed_gibbs_excesses) 

print 'enthalpy interaction:', liquid.enthalpy_interaction
print 'entropy interaction:', liquid.entropy_interaction
print 'volume interaction:', liquid.volume_interaction 

print 'sqrt(trace of covariance matrix):'
for i, cov in enumerate(pcov):
    print np.sqrt(cov[i])

#################
#  END FITTING  #
#################


fMgSiO3s=np.linspace(0., 1., 101)
gibbs_excesses=np.empty_like(fMgSiO3s)
for p, t in pts:
    #print 'Pressure (GPa):', p, ", Temperature (K):", t
    pressure=p*1.e9
    temperature=t*1.
    for i, fMgSiO3 in enumerate(fMgSiO3s):
        liquid.set_composition([1.-fMgSiO3, fMgSiO3])
        liquid.set_state(pressure, temperature)
        gibbs_excesses[i] = liquid.excess_gibbs
    plt.plot(fMgSiO3s, gibbs_excesses, linewidth=1, label=str(p)+' GPa, '+str(t)+' K')

    MgO_liq.set_state(pressure, temperature)
    MgSiO3_liq.set_state(pressure, temperature)
    MgO_gibbs = MgO_liq.gibbs
    MgSiO3_gibbs = MgSiO3_liq.gibbs

    observed_fMgSiO3s=[]
    observed_Gexs=[]
    for phase in phases:
        #print phase.params['name']
        nSi = phase.params['formula']['Si']
        nMg = phase.params['formula']['Mg']
        
        nMgSiO3 = nSi
        nMgO = nMg - nSi
        sum_units = nMgSiO3+nMgO
        fMgSiO3=nMgSiO3/sum_units
        
        observed_fMgSiO3s.append(fMgSiO3)

        phase.set_state(pressure, temperature)
        Gex = phase.gibbs/sum_units - (fMgSiO3*MgSiO3_gibbs + (1.-fMgSiO3)*MgO_gibbs)
        observed_Gexs.append(Gex)


    plt.plot(observed_fMgSiO3s, observed_Gexs, marker='o', linestyle='None', label=str(p)+' GPa, '+str(t)+' K')




plt.legend(loc='lower right')
plt.show()


