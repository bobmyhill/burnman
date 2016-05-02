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

class MgO_SiO2_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Subregular MgO-SiO2 liquid'
        self.type='subregular'

        self.endmembers = [[DKS_2013_liquids.MgO_liquid(), '[Mg]O'], 
                           [DKS_2013_liquids.SiO2_liquid(), '[Si]O2']]

        self.enthalpy_interaction = [[[0., 0.]]]
        self.volume_interaction   = [[[0., 0.]]]
        self.entropy_interaction  = [[[0., 0.]]]
                        
        burnman.SolidSolution.__init__(self, molar_fractions)



liquid=MgO_SiO2_liquid()



# Set up endmembers
MgO_liq = DKS_2013_liquids.MgO_liquid()
MgSiO3_liq = DKS_2013_liquids.MgSiO3_liquid()
Mg2SiO4_liq = DKS_2013_liquids.Mg2SiO4_liquid()
SiO2_liq = DKS_2013_liquids.SiO2_liquid()


MgSiO3_liq.set_state(50.e9, 4000.)
Mg2SiO4_liq.set_state(50.e9, 4000.)
MgO_liq.set_state(50.e9, 4000.)

xi = 7./3.
V_ideal = 0.5*(MgO_liq.V + MgSiO3_liq.V)
V_nonideal = 0.5*(Mg2SiO4_liq.V)
V_xs = V_nonideal - V_ideal
K_ideal = V_nonideal / (0.5*MgO_liq.V/MgO_liq.K_T + 0.5*MgSiO3_liq.V/MgSiO3_liq.K_T)
print(K_ideal/1.e9, Mg2SiO4_liq.K_T/1.e9)
K_xs = V_xs*K_ideal/V_ideal/(np.power(V_nonideal/V_ideal, xi+1.) - 1.)

print(K_xs/1.e9)

# SOLUTION MODEL CREATION
class MgO_MgSiO3_binary(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='MgO-MgSiO3 liquid binary'
        self.type='full_subregular'
        self.endmembers = [[DKS_2013_liquids.MgO_liquid(),  '[Mg]O'],
                           [DKS_2013_liquids.MgSiO3_liquid(), 'Mg[Si]O3']]
        self.energy_interaction = [[[0., 0.]]]
        self.volume_interaction = [[[-1.5e-6, -1.5e-6]]]
        self.modulus_interaction = [[[66.e9, 66.e9]]]
        self.theta_interaction = [[[4000., 4000.]]]
        burnman.SolidSolution.__init__(self, molar_fractions)

liq = MgO_MgSiO3_binary()

phases = [DKS_2013_liquids.MgSiO3_liquid(),
         DKS_2013_liquids.MgSi2O5_liquid(),
         DKS_2013_liquids.MgSi3O7_liquid(),
         DKS_2013_liquids.MgSi5O11_liquid(),
         DKS_2013_liquids.Mg2SiO4_liquid(),
         DKS_2013_liquids.Mg3Si2O7_liquid(),
         DKS_2013_liquids.Mg5SiO7_liquid()
         ]
phases = [DKS_2013_liquids.Mg2SiO4_liquid()]




temperatures = np.linspace(3000., 7000., 41)
Pth = np.empty_like(temperatures)
V_excesses = np.empty_like(temperatures)
phase = phases[0]
for P in np.linspace(1.e10, 10.e10, 10):
    for i, T in enumerate(temperatures):
        #P = phase.method.pressure(T, phase.params['V_0']*0.5, phase.params)
        #Pth[i] = P

        
        phase.set_state(P, T)
        MgO_liq.set_state(P, T)
        MgSiO3_liq.set_state(P, T)
        
        MgO_V = MgO_liq.V
        MgSiO3_V = MgSiO3_liq.V
        V_excesses[i] = phase.V - (MgSiO3_V + MgO_V)
        
        
    plt.plot(1./temperatures, V_excesses)
plt.xlim(0., 4e-4)
plt.show()

volume_fractions = np.linspace(0.45, 0.75, 31)
temperatures = np.linspace(2000., 7000., 6)
pressures = np.empty_like(volume_fractions)
V_excesses = np.empty_like(volume_fractions)
V_excesses_model = np.empty_like(volume_fractions)
a_excesses = np.empty_like(volume_fractions)
aKT_excesses = np.empty_like(volume_fractions)
for phase in phases:
    nSi = phase.params['formula']['Si']
    nMg = phase.params['formula']['Mg']
    
    sum_cations = nSi+nMg
    fSi=nSi/sum_cations
    for T in temperatures:
        for i, Vfrac in enumerate(volume_fractions):
            P = MgSiO3_liq.method.pressure(T, Vfrac*MgSiO3_liq.params['V_0'], MgSiO3_liq.params)
            pressures[i] = P
            print(P, T, fSi)
            MgO_liq.set_state(P, T)
            MgSiO3_liq.set_state(P, T)
            #SiO2_liq.set_state(P, T)
            
            phase.set_state(P, T)
            MgO_V = MgO_liq.V
            MgSiO3_V = MgSiO3_liq.V
            V_excesses[i] = (phase.V - (MgSiO3_V + MgO_V))/2.

            MgO_a = MgO_liq.alpha
            MgSiO3_a = MgSiO3_liq.alpha
            a_excesses[i] = (phase.alpha - (MgSiO3_a + MgO_a)/2.)/(phase.alpha)

            liq.set_composition([0.5, 0.5])
            liq.set_state(P, T)
            V_excesses_model[i] = liq.excess_volume
            
            MgO_aKT = MgO_liq.alpha*MgO_liq.K_T
            MgSiO3_aKT = MgSiO3_liq.alpha*MgSiO3_liq.K_T
            aKT_excesses[i] = (phase.alpha*phase.K_T - (MgSiO3_aKT + MgO_aKT)/2.)/(phase.alpha*phase.K_T)
         
        plt.subplot(221)   
        plt.plot(pressures, a_excesses, label=str(fSi)+': '+str(T)+' K')
        plt.subplot(222)   
        plt.plot(pressures, aKT_excesses, label=str(fSi)+': '+str(T)+' K')
        plt.subplot(223)   
        plt.plot(pressures, V_excesses, label=str(fSi)+': '+str(T)+' K')
        plt.subplot(224)   
        plt.plot(pressures, V_excesses_model, label=str(fSi)+': '+str(T)+' K (model)')
        #plt.ylim(-10.e-7, -1.e-7)
plt.legend(loc='lower right')
plt.show()

