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
pts = [[10, 1500],
       [13, 1500],
       [16, 1500],
       [10, 2000],
       [13, 2000],
       [16, 2000],
       [10, 2500],
       [13, 2500],
       [16, 2500]]


gibbs_excess_data=[]
entropy_excess_data=[]
volume_excess_data=[]
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
        entropy_excess_data.append([pressure, temperature, fSi, Sex])
        volume_excess_data.append([pressure, temperature, fSi, Vex])

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


gibbs_excess_data = np.array(gibbs_excess_data)
entropy_excess_data = np.array(entropy_excess_data)
volume_excess_data = np.array(volume_excess_data)
observed_gibbs_excesses = np.array(zip(*gibbs_excess_data)[3])
observed_entropy_excesses = np.array(zip(*entropy_excess_data)[3])
observed_volume_excesses = np.array(zip(*volume_excess_data)[3])


def fit_all_parameters(data, H0, H1, S0, S1, V0, V1):
    liquid.enthalpy_interaction  = [[[H0, H1]]]  
    liquid.entropy_interaction  = [[[S0, S1]]]  
    liquid.volume_interaction  = [[[V0, V1]]]  
    burnman.SolidSolution.__init__(liquid, [0., 1.])

    gibbs_excesses=[]
    for datum in data:
        pressure, temperature, fSi, Gex = datum
        liquid.set_composition([1.-fSi, fSi])
        liquid.set_state(pressure, temperature)
        gibbs_excesses.append(liquid.excess_gibbs)

    return gibbs_excesses


def fit_entropy_parameters(data, S0, S1):
    liquid.entropy_interaction  = [[[S0, S1]]]  

    burnman.SolidSolution.__init__(liquid, [0., 1.])

    entropy_excesses=[]
    for datum in data:
        pressure, temperature, fSi, Gex = datum
        liquid.set_composition([1.-fSi, fSi])
        liquid.set_state(pressure, temperature)
        entropy_excesses.append(liquid.excess_entropy)

    return entropy_excesses

def fit_volume_parameters(data, V0, V1):
    liquid.volume_interaction  = [[[V0, V1]]]  

    burnman.SolidSolution.__init__(liquid, [0., 1.])

    volume_excesses=[]
    for datum in data:
        pressure, temperature, fSi, Gex = datum
        liquid.set_composition([1.-fSi, fSi])
        liquid.set_state(pressure, temperature)
        volume_excesses.append(liquid.excess_volume)

    return volume_excesses

print 'fitting...'
popt, pcov = curve_fit(fit_all_parameters, gibbs_excess_data, observed_gibbs_excesses) 

print popt, pcov

print 'fitting...'
popt, pcov = curve_fit(fit_entropy_parameters, entropy_excess_data, observed_entropy_excesses) 

print popt, pcov

print 'plotting...'
plt.subplot(233)
pressure = 13.e9
temperature = 2000.
compositions = np.linspace(0.0, 1.0, 21)
entropies = np.empty_like(compositions)
for i, fSi in enumerate(compositions):
        liquid.set_composition([1.-fSi, fSi])
        liquid.set_state(pressure, temperature)
        entropies[i] = liquid.excess_entropy

plt.plot(compositions, entropies, linewidth=1, label='fit')

print 'fitting...'
popt, pcov = curve_fit(fit_volume_parameters, volume_excess_data, observed_volume_excesses) 

print popt, pcov

print 'plotting...'
plt.subplot(234)
pressure = 13.e9
temperature = 2000.
compositions = np.linspace(0.0, 1.0, 21)
volumes = np.empty_like(compositions)
for i, fSi in enumerate(compositions):
        liquid.set_composition([1.-fSi, fSi])
        liquid.set_state(pressure, temperature)
        volumes[i] = liquid.excess_volume

plt.plot(compositions, volumes, linewidth=1, label='fit')



plt.subplot(236)
plt.legend(loc='lower right')
plt.show()


'''
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
'''
