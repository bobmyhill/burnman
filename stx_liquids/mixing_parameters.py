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

        self.endmembers = [[DKS_2013_liquids.SiO2_liquid(), '[Mg]O'], 
                           [DKS_2013_liquids.MgO_liquid(), '[Si]O2']]

        self.enthalpy_interaction = [[[0., 0.]]]
        self.volume_interaction   = [[[0., 0.]]]
        self.entropy_interaction  = [[[0., 0.]]]
                        
        burnman.SolidSolution.__init__(self, molar_fractions)



liquid=MgO_SiO2_liquid()


phases = [DKS_2013_liquids.SiO2_liquid(),
         DKS_2013_liquids.MgSiO3_liquid(),
         DKS_2013_liquids.MgSi2O5_liquid(),
         DKS_2013_liquids.MgSi3O7_liquid(),
         DKS_2013_liquids.MgSi5O11_liquid(),
         DKS_2013_liquids.Mg2SiO4_liquid(),
         DKS_2013_liquids.Mg3Si2O7_liquid(),
         DKS_2013_liquids.Mg5SiO7_liquid(),
         DKS_2013_liquids.MgO_liquid()
         ]

# Set up endmembers
MgO_liq = DKS_2013_liquids.MgO_liquid()
SiO2_liq = DKS_2013_liquids.SiO2_liquid()

# Set up points from which to query de Koker et al. (2013) liquid models
pts = [[14, 2000],
       [14, 2500],
       [14, 3000],
       [14, 4000],
       [14, 5000],
       [14, 6000],
       [12, 2000],
       [16, 2000],
       [12, 3000],
       [16, 3000],
       [12, 4000],
       [16, 4000]]


gibbs_excess_data=[]
for p, t in pts:
    #print 'Pressure (GPa):', p, ", Temperature (K):", t
    pressure=p*1.e9
    temperature=t*1.

    MgO_liq.set_state(pressure, temperature)
    SiO2_liq.set_state(pressure, temperature)
    MgO_gibbs = MgO_liq.gibbs
    SiO2_gibbs = SiO2_liq.gibbs

    MgO_H = MgO_liq.H
    SiO2_H = SiO2_liq.H

    MgO_S = MgO_liq.S
    SiO2_S = SiO2_liq.S

    MgO_V = MgO_liq.V
    SiO2_V = SiO2_liq.V

    MgO_K_T = MgO_liq.K_T
    SiO2_K_T = SiO2_liq.K_T

    fSis=[]
    Gexs=[]
    Hexs=[]
    Sexs=[]
    Vexs=[]
    K_Ts=[]
    K_Texs=[]
    for phase in phases:
        #print phase.params['name']
        
        nSi = phase.params['formula']['Si']
        nMg = phase.params['formula']['Mg']
        
        sum_cations = nSi+nMg
        fSi=nSi/sum_cations
        
        phase.set_state(pressure, temperature)
        Gex = phase.gibbs/sum_cations - (fSi*SiO2_gibbs + (1.-fSi)*MgO_gibbs)       
        Hex = phase.H/sum_cations - (fSi*SiO2_H + (1.-fSi)*MgO_H)

        Sex = phase.S/sum_cations - (fSi*SiO2_S + (1.-fSi)*MgO_S)

        Vex = phase.V/sum_cations - (fSi*SiO2_V + (1.-fSi)*MgO_V)

        K_T = phase.K_T
        K_Tex = (phase.K_T - (fSi*SiO2_K_T + (1.-fSi)*MgO_K_T))/K_T

        # data should be composition, pressure, temperature, gibbs_excess
        gibbs_excess_data.append([pressure, temperature, fSi, Gex])


        fSis.append(fSi)
        Gexs.append(Gex)
        Hexs.append(Hex)
        Sexs.append(Sex)
        Vexs.append(Vex)
        K_Ts.append(K_T)
        K_Texs.append(K_Tex)


    plt.subplot(231)
    plt.title('Excess Gibbs') 
    plt.plot(fSis, Gexs, marker='o', linestyle='None', label=str(p)+' GPa, '+str(t)+' K')
    plt.subplot(232)
    plt.title('Excess Enthalpies') 
    plt.plot(fSis, Hexs, marker='o', linestyle='None', label=str(p)+' GPa, '+str(t)+' K')
    plt.subplot(233)
    plt.title('Excess Entropies') 
    plt.plot(fSis, Sexs, marker='o', linestyle='None', label=str(p)+' GPa, '+str(t)+' K')
    plt.subplot(234)
    plt.title('Excess Volumes') 
    plt.plot(fSis, Vexs, marker='o', linestyle='None', label=str(p)+' GPa, '+str(t)+' K')
    plt.subplot(235)
    plt.title('Fractional excess K_T') 
    plt.plot(fSis, K_Texs, marker='o', linestyle='None', label=str(p)+' GPa, '+str(t)+' K')
    plt.subplot(236)
    plt.title('K_T') 
    plt.plot(fSis, K_Ts, marker='o', linestyle='None', label=str(p)+' GPa, '+str(t)+' K')

plt.legend(loc='lower right')
plt.show()

gibbs_excess_data = np.array(gibbs_excess_data)
observed_gibbs_excesses = np.array(zip(*gibbs_excess_data)[3])


def fit_parameters(data, H0, H1, S0, S1, V0, V1):
    liquid.enthalpy_interaction = [[[H0, H1]]]
    liquid.entropy_interaction  = [[[S0, S1]]] 
    liquid.volume_interaction   = [[[V0, V1]]]  

    burnman.SolidSolution.__init__(liquid, [0., 1.])

    gibbs_excesses=[]
    for datum in data:
        pressure, temperature, fSi, Gex = datum
        liquid.set_composition([1.-fSi, fSi])
        liquid.set_state(pressure, temperature)
        gibbs_excesses.append(liquid.excess_gibbs)

    return gibbs_excesses


popt, pcov = curve_fit(fit_parameters, gibbs_excess_data, observed_gibbs_excesses) 

fSis=np.linspace(0., 1., 101)
gibbs_excesses=np.empty_like(fSis)
for p, t in pts:
    #print 'Pressure (GPa):', p, ", Temperature (K):", t
    pressure=p*1.e9
    temperature=t*1.
    for i, fSi in enumerate(fSis):
        liquid.set_composition([1.-fSi, fSi])
        liquid.set_state(pressure, temperature)
        gibbs_excesses[i] = liquid.excess_gibbs
    plt.plot(fSis, gibbs_excesses, linewidth=1, label=str(p)+' GPa, '+str(t)+' K')

    MgO_liq.set_state(pressure, temperature)
    SiO2_liq.set_state(pressure, temperature)
    MgO_gibbs = MgO_liq.gibbs
    SiO2_gibbs = SiO2_liq.gibbs

    observed_fSis=[]
    observed_Gexs=[]
    for phase in phases:
        #print phase.params['name']
        
        nSi = phase.params['formula']['Si']
        nMg = phase.params['formula']['Mg']
        
        sum_cations = nSi+nMg
        fSi=nSi/sum_cations
        observed_fSis.append(fSi)

        phase.set_state(pressure, temperature)
        Gex = phase.gibbs/sum_cations - (fSi*SiO2_gibbs + (1.-fSi)*MgO_gibbs)
        observed_Gexs.append(Gex)


    plt.plot(observed_fSis, observed_Gexs, marker='o', linestyle='None', label=str(p)+' GPa, '+str(t)+' K')




plt.legend(loc='lower right')
plt.show()



print 'enthalpy interaction:', liquid.enthalpy_interaction
print 'entropy interaction:', liquid.entropy_interaction
print 'volume interaction:', liquid.volume_interaction 

print 'sqrt(trace of covariance matrix):'
for i, cov in enumerate(pcov):
    print np.sqrt(cov[i])
