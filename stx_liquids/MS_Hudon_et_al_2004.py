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

MgO_liq = DKS_2013_liquids.MgO_liquid()
SiO2_liq = DKS_2013_liquids.SiO2_liquid()


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
                                                  [0., 0., -37.656*Y[2] + 125.52*np.power(Y[2], 7.)],
                                                  [0., 0., 0.]]),
                       'zeta': lambda Y: np.array([[0., 0., 0.],
                                                  [0., 0., (-10180. - 10615.37*Y[2])*f + 8911.75*np.power(Y[2], 7.)],
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



xs = np.linspace(0.001, 0.999, 101)

for (P, T) in [(1.e5, 2000.)]:
    
    MgO_liq.set_state(P, T)
    SiO2_liq.set_state(P, T)
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
        Gex = phase.gibbs/sum_cations - (fSi*SiO2_gibbs + (1.-fSi)*MgO_gibbs)       
        Hex = phase.H/sum_cations - (fSi*SiO2_H + (1.-fSi)*MgO_H)

        Sex = phase.S/sum_cations - (fSi*SiO2_S + (1.-fSi)*MgO_S)

        Vex = phase.V/sum_cations - (fSi*SiO2_V + (1.-fSi)*MgO_V)

        K_T = phase.K_T
        K_Tex = (phase.K_T - (fSi*SiO2_K_T + (1.-fSi)*MgO_K_T))/K_T

        fSis.append(fSi)
        Gexs.append(Gex)
        Hexs.append(Hex)
        Sexs.append(Sex)
        Vexs.append(Vex)
        K_Ts.append(K_T)
        K_Texs.append(K_Tex)

        
    plt.subplot(221)
    plt.plot(fSis, np.array(Gexs)/1.e3, marker='o', linestyle='None', label='MgO-SiO2 (DKS)')
    plt.subplot(222)
    plt.plot(fSis, np.array(Hexs)/1.e3, marker='o', linestyle='None', label='MgO-SiO2 (DKS)')
    plt.subplot(223)
    plt.plot(fSis, Sexs, marker='o', linestyle='None', label='MgO-SiO2 (DKS)')
    plt.subplot(224)
    plt.plot(fSis, np.array(Vexs)*1.e6, marker='o', linestyle='None', label='MgO-SiO2 (DKS)')
    
    c = []
    G = []
    H = []
    S = []
    V = []
    V2 = []
    for i, x in enumerate(xs):
        y = 0.0001 # makes evaluation easier 
        FMS.molar_fractions = np.array([(1. - x)*y, (1. - x)*(1. - y),  x])
        sol = FMS.set_state(P, T)
        if sol != 1:
            print 'No solution found'
        else:
            c.append(x)
            G.append(FMS.delta_G)
            H.append(FMS.delta_H)
            S.append(FMS.delta_S)
            V.append(FMS.delta_V)
    plt.subplot(221)
    plt.xlabel('X SiO2 (mole fraction)')
    plt.ylabel('Gmix (kJ/mol)')
    plt.plot(c, np.array(G)/1.e3, label='MgO-SiO2 (Hudon et al. 2004)')
    plt.subplot(222)
    plt.xlabel('X SiO2 (mole fraction)')
    plt.ylabel('Hmix (kJ/mol)')
    plt.plot(c, np.array(H)/1.e3, label='MgO-SiO2 (Hudon et al. 2004)')
    plt.subplot(223)
    plt.xlabel('X SiO2 (mole fraction)')
    plt.ylabel('Smix (J/K/mol)')
    plt.plot(c, S, label='MgO-SiO2 (Hudon et al. 2004)')
    plt.subplot(224)
    plt.xlabel('X SiO2 (mole fraction)')
    plt.ylabel('Vmix (cm^3/mol)')
    plt.plot(c, np.array(V)*1.e6, label='MgO-SiO2 (Hudon et al. 2004)')

from burnman.minerals import HP_2011_ds62, SLB_2011, DKS_2013_liquids, DKS_2013_solids, RS_2014_liquids

P0 = 1.e5
T0 = 2000.


per = HP_2011_ds62.per()
wus = HP_2011_ds62.fper()
fo = HP_2011_ds62.fo()
fa = HP_2011_ds62.fa()
en = HP_2011_ds62.en()
crst = HP_2011_ds62.crst()

ms = [per, wus, fo, fa, en, crst]
for m in ms:
    m.set_state(P0, T0)

Mg_X = np.array([0.0, 1./3., 0.5, 1.0])
Mg_mod = np.array([5.0e-6, 3.4e-6, 5.0e-6, 0.e-6])
Mg_Y = np.array([per.V + Mg_mod[0],
                 (fo.V + Mg_mod[1])/3.,
                 (en.V + Mg_mod[2]*2.)/4.,
                 crst.V + Mg_mod[3]])
Mg_mix = Mg_Y - ((1. - Mg_X)*(per.V + Mg_mod[0]) + (Mg_X)*(crst.V + Mg_mod[3]))

Fe_X = np.array([0.0, 1./3., 1.0])
Fe_mod = np.array([2.2e-6, 4.92e-6, 0.e-6])
Fe_Y = np.array([wus.V + Fe_mod[0],
                 (fa.V + Fe_mod[1])/3.,
                 crst.V + Fe_mod[2]])
Fe_mix = Fe_Y - ((1. - Fe_X)*(wus.V + Fe_mod[0]) + (Fe_X)*(crst.V + Fe_mod[2]))

plt.plot(Mg_X, Mg_mix*1.e6, marker='o', linestyle=':', label='MgO-SiO2 (experimental)')
plt.plot(Fe_X, Fe_mix*1.e6, marker='o', linestyle=':', label='FeO-SiO2 (experimental)')
plt.legend(loc='lower right')
plt.title(str(P/1.e9)+' GPa; '+str(T)+' K')
plt.show()
