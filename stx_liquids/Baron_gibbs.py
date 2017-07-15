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



MgO_liq = DKS_2013_liquids.MgO_liquid()
SiO2_liq = DKS_2013_liquids.SiO2_liquid()
DKS_per = DKS_2013_solids.periclase()
DKS_stv = DKS_2013_solids.stishovite()
DKS_pv = DKS_2013_solids.perovskite()

SLB_per = SLB_2011.periclase()
SLB_stv = SLB_2011.stishovite()
SLB_pv = SLB_2011.mg_perovskite()


phases = [DKS_2013_liquids.MgO_liquid(),
          DKS_2013_liquids.Mg5SiO7_liquid(),
          DKS_2013_liquids.Mg2SiO4_liquid(),
          DKS_2013_liquids.Mg3Si2O7_liquid(),
          DKS_2013_liquids.MgSiO3_liquid(),
          DKS_2013_liquids.MgSi2O5_liquid(),
          DKS_2013_liquids.MgSi3O7_liquid(),
          DKS_2013_liquids.MgSi5O11_liquid(),
          DKS_2013_liquids.SiO2_liquid()]

R = burnman.constants.gas_constant
class FMS_solution_Wu():
    def __init__(self, molar_fractions=None):
        self.name = 'FeO-MgO-SiO2 solution'
        self.params = {'Z': 2.,
                       'b': np.array([0.6887, 0.6887, 1.3774]),
                       'omega': lambda Y: np.array([[0., 3347., - 17697.
                                                     - 38528.*Y[2]
                                                     + 842570*np.power(Y[2], 5.)
                                                     - 1549201*np.power(Y[2], 6.)
                                                     + 962015.*np.power(Y[2], 7.)],
                                                    [0., 0., -86090 + -48974.*Y[2] + 328109.*np.power(Y[2], 7.)],
                                                    [0., 0., 0.]]),
                       'eta': lambda Y: np.array([[0., 0., -16.736 + 62.76*np.power(Y[2], 7.)],
                                                  [0., 0., -37.656*Y[2] + 125.52*np.power(Y[2], 7.)],
                                                  [0., 0., 0.]])
        }
        

    # Simultaneous equations for asymmetric ternary system
    # The last member is the odd component
    def _asymmetric_pair_proportions(self, args, T):
        # A = FeO, B = MgO, C = SiO2
        nAA, nAB, nAC, nBB, nBC, nCC  = np.abs(args)
        nA, nB, nC = self.molar_fractions
        self.npr = np.array([ [nAA, nAB, nAC],
                                  [0., nBB, nBC],
                                  [0., 0., nCC] ])
        self.Xpr = self.npr/np.sum(self.npr)
        
    
        self.X = self.molar_fractions/np.sum(self.molar_fractions) # normalise molar_fractions (if necessary)
        self.Y = ( self.params['b']*self.molar_fractions /
                       np.sum(self.params['b']*self.molar_fractions) ) # equation 52 in PB1986
        
        
        self.omega = self.params['omega'](self.Y)
        self.eta = self.params['eta'](self.Y)
        self.eqm = 4.*np.exp(-(2.*(self.omega - self.eta * T ) /
                                   (self.params['Z']*R*T)))
    
        out = []

        out.append(2.*self.Xpr[0][0] + self.Xpr[0][1] + self.Xpr[0][2] - 2.*self.Y[0]) 
        out.append(2.*self.Xpr[1][1] + self.Xpr[1][2] + self.Xpr[0][1] - 2.*self.Y[1]) 
        out.append(2.*self.Xpr[2][2] + self.Xpr[0][2] + self.Xpr[1][2] - 2.*self.Y[2]) 
        
        out.append(self.eqm[0][1]*nAA*nBB - nAB*nAB)
        out.append(self.eqm[0][2]*nAA*nCC - nAC*nAC)
        out.append(self.eqm[1][2]*nBB*nCC - nBC*nBC)
        
        return out

    def _equilibrium_order(self, P, T, guess=None):
        Y_guess = ( self.params['b']*self.molar_fractions /
                    np.sum(self.params['b']*self.molar_fractions) )

        
        try:
            guesses = [self.pairs]
        except:
            guesses = []
        
        if guess != None:
            guesses.append(guess)

        g0 = np.array([Y_guess[0], 0.0, 0.0,
                       Y_guess[1], 0.0,
                       Y_guess[2]])
        g1 = np.array([0., Y_guess[0] + Y_guess[1] - Y_guess[2], Y_guess[0] - Y_guess[1] + Y_guess[2],
                       0., -Y_guess[0] + Y_guess[1] + Y_guess[2],
                       0.])


        fs = np.linspace(0., 1., 21)
        for f in fs:
            guesses.append(g0*f + g1*(1. - f))
            
        solution_found = False
        i=0
        while solution_found == False and i < len(guesses):
            sol =  fsolve(self._asymmetric_pair_proportions,
                          guesses[i], args=(T), full_output=True)

            if sol[2] == 1:
                solution_found = True
            else:
                i += 1
             
        if solution_found == False: 
            print sol[3]
            return 0
        else:
            self.pairs = sol[0]
            
            # Equations 3, 4, 5.
            # note modifications to each of the equations in the text
            self.delta_H = ( np.sum(self.params['b'] * self.X) *
                                 np.sum(self.Xpr * self.omega) ) / 2. 
        
            delta_Snc = ( np.sum(self.params['b'] * self.X) *
                          np.sum(self.Xpr * self.eta) ) / 2. 
        
        
            delta_Sc = -R *(np.sum([xi * np.log(xi) for xi in self.X if xi > 1.e-12]) +
                            self.params['Z'] / 2. *np.sum(self.params['b'] * self.X) *
                            ( np.sum( [ self.Xpr[i][i] * np.log(self.Xpr[i][i] /
                                                                    (self.Y[i]*self.Y[i]))
                                        for i in range(len(self.Y)) if self.Y[i] > 1.e-12] )
                              
                              + np.sum(np.sum( [ [ self.Xpr[i][j] * np.log(self.Xpr[i][j] /
                                                                               (2.*self.Y[i]*self.Y[j]))
                                                   for i in range(len(self.Y))
                                                   if j > i and self.Y[i] > 1.e-12 and self.Y[j] > 1.e-12 ]
                                                 for j in range(len(self.Y)) ] )) ) ) 
        
            self.delta_S = delta_Sc + delta_Snc
            self.delta_G = self.delta_H - T*self.delta_S
            return 1

    def _unit_vector_length(self, v):
        length = np.sqrt(np.sum([ vi*vi for vi in v ]))
        return v/length, length
    
    def set_state(self, P, T):

        molar_fractions = self.molar_fractions

        # Find partial gibbs
        # First, find vector towards MgO, FeO and SiO2
        dX = 0.001

        
        dA, XA = self._unit_vector_length(np.array([1., 0., 0.]) - self.molar_fractions)
        dA = dA*dX
        self.molar_fractions = molar_fractions + dA
        sol = self._equilibrium_order(P, T)
        GA = self.delta_G
        
        dB, XB = self._unit_vector_length(np.array([0., 1., 0.]) - self.molar_fractions)
        dB = dB*dX
        self.molar_fractions = molar_fractions + dB
        sol = self._equilibrium_order(P, T)
        GB = self.delta_G

        dC, XC = self._unit_vector_length(np.array([0., 0., 1.]) - self.molar_fractions)
        dC = dC*dX
        self.molar_fractions = molar_fractions + dC
        sol = self._equilibrium_order(P, T)
        GC = self.delta_G


        self.molar_fractions = molar_fractions
        sol = self._equilibrium_order(P, T)
        G0 = self.delta_G

        self.partial_gibbs_excesses = np.array( [ G0 + (GA - G0)/dX*XA,
                                                  G0 + (GB - G0)/dX*XB,
                                                  G0 + (GC - G0)/dX*XC ] )

        MgO_liq.set_state(P, T)
        SiO2_liq.set_state(P, T)

        self.partial_gibbs = ( np.array( [ 0.,
                                           MgO_liq.gibbs,
                                           SiO2_liq.gibbs] ) +
                               self.partial_gibbs_excesses )
        return sol
    




FMS_Wu = FMS_solution_Wu()

class FMS_solution_simple():
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


FMS = FMS_solution_simple()






# Gibbs figure
P =  24.e9
Simon_Glatzel = lambda Tr, Pr, A, C, P: Tr*np.power(1. + (P - Pr)/A, 1./C)
T = Simon_Glatzel(2605., 24., 29.892, 3.677, P/1.e9)

print P, T


for m in [MgO_liq, SiO2_liq, DKS_per, DKS_pv, DKS_stv, SLB_per, SLB_pv, SLB_stv]:
    m.set_state(P, T)
    
S = DKS_stv.gibbs - SiO2_liq.gibbs
M = DKS_per.gibbs - MgO_liq.gibbs
PV = 0.5*(S + M) + 0.5*(DKS_pv.gibbs - DKS_per.gibbs - DKS_stv.gibbs)

plt.plot([0.0, 0.5, 1.0], np.array([M, PV, S])/1.e3, label='de Koker et al. (2013) solids')

PV2 = 0.5*(S + M) + 0.5*(SLB_pv.gibbs - SLB_per.gibbs - SLB_stv.gibbs)
plt.plot([0.0, 0.5, 1.0], np.array([M, PV2, S])/1.e3, label='Stixrude and Lithgow-Bertelloni (2011) solids')


np.savetxt('Baron_deltaG_solids.dat', zip(*[[0.0, 0.5, 1.0], [M, PV, S], [M, PV2, S]]), header='X(SiO2), DKS2013, SLB2011') 




for P in [24.e9]:
    for m in [MgO_liq, SiO2_liq, DKS_per, DKS_pv, DKS_stv, SLB_per, SLB_pv, SLB_stv]:
        m.set_state(P, T)
    fSis=[]
    Gexs=[]
    for phase in phases:
        
        try:
            nSi = phase.params['formula']['Si']
        except:
            nSi = 0.
        try:
            nMg = phase.params['formula']['Mg']
        except:
            nMg = 0.
                
        sum_cations = nSi+nMg
        fSi=nSi/sum_cations
        
        phase.set_state(P, T)
        Gex = phase.gibbs/sum_cations - (fSi*SiO2_liq.gibbs + (1.-fSi)*MgO_liq.gibbs)
        Gexs.append(Gex)
        fSis.append(fSi)


    plt.plot(fSis, np.array(Gexs)/1.e3, marker='o', linestyle='None', label='de Koker et al. (2013) liquid')


    
np.savetxt('Baron_deltaG_DKS2013_liquids.dat', zip(*[fSis, Gexs]), header='X(SiO2), deltaG (J/K/mol)') 


Xs = np.linspace(0.0001, 0.9999, 101)
Gs = np.empty_like(Xs)
for i, X in enumerate(Xs):
    FMS_Wu.molar_fractions = [0., 1. - X, X]
    FMS_Wu.set_state(P, T)
    Gs[i] = FMS_Wu.delta_G

plt.plot(Xs, Gs/1.e3, label='Wu et al. (1993) liquid')

np.savetxt('Baron_deltaG_Wu_et_al_2013_liquid.dat', zip(*[Xs, Gs]), header='X(SiO2), deltaG (J/K/mol)') 


plt.legend(loc='lower right')
plt.xlabel('X SiO2 (mol %)')
plt.ylabel('Gibbs energy difference (kJ/mol)')
plt.title(str(P/1.e9)+' GPa; '+str(T)+' K')
plt.show()
