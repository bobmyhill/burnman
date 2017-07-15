import numpy as np
from scipy.optimize import fsolve, brentq
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
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

R = burnman.constants.gas_constant

f = 1.0
#f = 0.58 # to fit experimental data
class FMS_solution():
    def __init__(self, molar_fractions=None):
        self.name = 'FeO-MgO-SiO2 solution'
        self.params = {'Z': 2.,
                       'b': np.array([0.6887, 0.6887, 1.3774]),
                       'omega': lambda Y: np.array([[0., 3347., - 17697.
                                                     - 38528.*Y[2]
                                                     + 842570*np.power(Y[2], 5.)
                                                     - 1549201*np.power(Y[2], 6.)
                                                     + 962015.*np.power(Y[2], 7.)],
                                                    [0., 0., -86090 - 48974*Y[2] + 328109*np.power(Y[2], 7.)],
                                                    [0., 0., 0.]]),
                       'eta': lambda Y: np.array([[0., 0., -16.736 + 62.76*np.power(Y[2], 7.)],
                                                  [0., 0., (37.656*0.5 - 125.52*np.power(0.5, 7)) -37.656*Y[2] + 125.52*np.power(Y[2], 7.)],
                                                  [0., 0., 0.]]),
                       'zeta': lambda Y: np.array([[0., 0., 0.],
                                                  [0., 0., (-10180.)],
                                                  [0., 0., 0.]])/1.e9 # in Pa^-1
                   }
    # Simultaneous equations for asymmetric ternary system
    # The last member is the odd component
    def _asymmetric_pair_proportions(self, args, P, T):
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
        self.zeta = self.params['zeta'](self.Y)
        self.eqm = 4.*np.exp(-(2.*(self.omega - self.eta * T +self.zeta*P) /
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
                          guesses[i], args=(P, T), full_output=True)

            if sol[2] == 1:
                solution_found = True
            else:
                i += 1
             
        if solution_found == False: 
            print sol[3]
            return 0
        else:
            self.pairs = sol[0]


            self.delta_V = ( np.sum(self.params['b'] * self.X) *
                                 np.sum(self.Xpr * self.zeta ) ) / 2. 
            
            # Equations 3, 4, 5.
            # note modifications to each of the equations in the text
            
            self.delta_H = ( np.sum(self.params['b'] * self.X) *
                             np.sum(self.Xpr * (self.omega + self.zeta*P) ) ) / 2.
            
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


        self.partial_gibbs = ( np.array( [ 0.,
                                           0.,
                                           0. ] ) +
                               self.partial_gibbs_excesses )
        return sol




FMS = FMS_solution()



for P in [1.e9, 2.e9, 3.e9]:
    
    Ts = []
    G = []
    H = []
    S = []
    V = []
    V2 = []
    
    for T in np.linspace(1500., 5500., 101):
        y = 0.0001 # makes evaluation easier
        x = 1./3.
        FMS.molar_fractions = np.array([(1. - x)*y, (1. - x)*(1. - y),  x])
        sol = FMS.set_state(P, T)
        if sol != 1:
            print 'No solution found'
        else:
            Ts.append(T)
            G.append(FMS.delta_G)
            H.append(FMS.delta_H)
            S.append(FMS.delta_S)
            V.append(FMS.delta_V)
    plt.subplot(221)
    plt.xlabel('T (K)')
    plt.ylabel('Gmix (kJ/mol)')
    plt.plot(Ts, np.array(G)/1.e3, label=str(P/1.e9))
    plt.subplot(222)
    plt.xlabel('T (K)')
    plt.ylabel('Hmix (kJ/mol)')
    plt.plot(Ts, np.array(H)/1.e3, label=str(P/1.e9))
    plt.subplot(223)
    plt.xlabel('T (K)')
    plt.ylabel('Smix (J/K/mol)')
    plt.plot(Ts, S, label=str(P/1.e9))
    plt.subplot(224)
    plt.xlabel('T (K)')
    plt.ylabel('Vmix (cm^3/mol)')
    plt.plot(Ts, np.array(V)*1.e6, label=str(P/1.e9))
plt.legend(loc='upper right')
plt.show()
