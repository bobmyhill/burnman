import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import HP_2011_ds62, RS_2014_liquids, DKS_2013_solids, DKS_2013_liquids, SLB_2011
from burnman import constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import fsolve, brentq

'''
# SiO2
fig1 = mpimg.imread('figures/hp_sio2_melting.png')
plt.imshow(fig1, extent=[0., 80., 1500., 5000.], aspect='auto')
fig2 = mpimg.imread('figures/sio2_melting.png')
plt.imshow(fig2, extent=[0., 15., 1673., 3273.], aspect='auto')


stv = DKS_2013_solids.stishovite()
liq = DKS_2013_liquids.SiO2_liquid()
pressures = np.linspace(13.e9, 250.e9, 51)
temperatures = np.empty_like(pressures)
guess = 3000.
for i, P in enumerate(pressures):
    print P/1.e9
    temperatures[i] = burnman.tools.equilibrium_temperature([stv, liq], [1.0, -1.0], P, max(3000., guess))
    guess = temperatures[i]
plt.plot(pressures/1.e9, temperatures)
plt.show()


fig1 = mpimg.imread('figures/Alfe_MgO_melting.png')
plt.imshow(fig1, extent=[0., 150., 2000., 9000.], aspect='auto')

per = DKS_2013_solids.periclase()
liq = DKS_2013_liquids.MgO_liquid()
pressures = np.linspace(1.e5, 150.e9, 51)
temperatures = np.empty_like(pressures)
guess = 3000.
for i, P in enumerate(pressures):
    print P/1.e9
    temperatures[i] = burnman.tools.equilibrium_temperature([per, liq], [1.0, -1.0], P, max(3000., guess))
    guess = temperatures[i]
plt.plot(pressures/1.e9, temperatures)
plt.show() 

exit()
'''



R = burnman.constants.gas_constant

# MgO melting curve
per = SLB_2011.periclase()
#per.params['F_0'] += -40036.738
per.params['q_0'] = 0.15
per.params['grueneisen_0'] = 1.5 # not a perfect match of both HP heat capacity (~1.4) and HP volume (~1.6), but not bad...

'''
per2 = HP_2011_ds62.per()
temperatures = np.linspace(100., 3000., 101)
C1 = np.empty_like(temperatures)
C2 = np.empty_like(temperatures)
V1 = np.empty_like(temperatures)
V2 = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    per.set_state(1.e5, T)
    per2.set_state(1.e5, T)

    C1[i] = per.heat_capacity_p
    C2[i] = per2.heat_capacity_p
    V1[i] = per.V
    V2[i] = per2.V

plt.plot(temperatures, V1)
plt.plot(temperatures, V2)
plt.show()
'''

MgO_liq = DKS_2013_liquids.MgO_liquid()
SiO2_liq = DKS_2013_liquids.SiO2_liquid()
stv = DKS_2013_solids.stishovite()
per = DKS_2013_solids.periclase()
mpv = DKS_2013_solids.perovskite()


# FeO
class FS_solution():
    def __init__(self, molar_fractions=None):
        self.name = 'FeO-SiO2 solution'
        self.params = {'Z': 2.,
                       'b': np.array([0.6887, 0.6887, 1.3774]),
                       'omega': lambda Y: np.array([[0., 3347., (- 17697.
                                                                 - 38528.*Y[2]
                                                                 + 842570*np.power(Y[2], 5.)
                                                                 - 1549201*np.power(Y[2], 6.)
                                                                 + 962015.*np.power(Y[2], 7.))],
                                                    [0., 0., (-71500. +
                                                              240000. * np.power(Y[2], 7.))],
                                                    [0., 0., 0.]]),
                       'eta': lambda Y: np.array([[0., 0., -16.736 + 62.76*np.power(Y[2], 7.)],
                                                  [0., 0., 80.*(np.power(Y[2], 8.) -
                                                                np.power(0.5, 8.))],
                                                  [0., 0., 0.]])
        }


        

    # Simultaneous equations for asymmetric ternary system
    # The last member is the odd component
    def _asymmetric_pair_proportions(self, args, T):
        # A = FeO, B = MgO, C = SiO2
        nAA, nAC, nCC  = np.abs(args)
        nA, nB, nC = self.molar_fractions
        self.npr = np.array([ [nAA, 0., nAC],
                                  [0., 0., 0.],
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
        out.append(2.*self.Xpr[2][2] + self.Xpr[0][2] + self.Xpr[1][2] - 2.*self.Y[2]) 
    
        out.append(self.eqm[0][2]*nAA*nCC - nAC*nAC)
        
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

        g0 = np.array([Y_guess[0], 0.0,
                       Y_guess[2]])
        g1 = np.array([0., Y_guess[0] + Y_guess[2],
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

        dC, XC = self._unit_vector_length(np.array([0., 0., 1.]) - self.molar_fractions)
        dC = dC*dX
        self.molar_fractions = molar_fractions + dC
        sol = self._equilibrium_order(P, T)
        GC = self.delta_G


        self.molar_fractions = molar_fractions
        sol = self._equilibrium_order(P, T)
        G0 = self.delta_G

        self.partial_gibbs_excesses = np.array( [ G0 + (GA - G0)/dX*XA,
                                                  G0 + (GC - G0)/dX*XC ] )

        #MgO_liq.set_state(P, T)
        #SiO2_liq.set_state(P, T)

        self.partial_gibbs = ( np.array( [ 0.,
                                           0. ] ) +
                               self.partial_gibbs_excesses )
        return sol

    



# FeO melting curve
wus = SLB_2011.wuestite()
Fe2SiO4_liq = RS_2014_liquids.Fe2SiO4_liquid()

# For FeO, use the model of Wu et al., 1993
# and Fe2SiO4 and SiO2 liquids (adjusting dE and dS to fit melting curves at 1 bar) 
# assuming that the volume of mixing is negligible,
# to calculate the gibbs free energy of FeO liquid.

FS = FS_solution()
FS.molar_fractions = [1./3., 0., 2./3.] # still include MgO


Pm = 1.e5
Tm = 1650.

wus.set_state(Pm, Tm)
FS.set_state(Pm, Tm)
Fe2SiO4_liq.set_state(Pm, Tm)
SiO2_liq.set_state(Pm, Tm)

S_tweak = 7.
delta_G = wus.gibbs - (((Fe2SiO4_liq.gibbs - 3.*FS.delta_G) - SiO2_liq.gibbs)/2.)
delta_S = wus.S - (((Fe2SiO4_liq.S - 3.*FS.delta_S) - SiO2_liq.S)/2.) + (171.990 - 157.409) + S_tweak
delta_E = delta_G + Tm*delta_S
delta_V = 3.e-7

class FeO_liquid():
    def __init__(self):
        self.name = 'FeO liquid'
        self.members = []
    def set_state(self, P, T):
        wus.set_state(P, T)
        FS.set_state(P, T)
        Fe2SiO4_liq.set_state(P, T)
        SiO2_liq.set_state(P, T)
        
        self.gibbs = ((Fe2SiO4_liq.gibbs - 3.*FS.delta_G) - SiO2_liq.gibbs)/2. + delta_E - T*delta_S + P*delta_V
        self.S = ((Fe2SiO4_liq.S - 3.*FS.delta_S) - SiO2_liq.S)/2. + delta_S
        self.V = ((Fe2SiO4_liq.V - 3.*0.) - SiO2_liq.V)/2. + delta_V



FeO_liq = FeO_liquid()

class FMS_solution():
    def __init__(self, molar_fractions=None):
        self.name = 'FeO-MgO-SiO2 solution'
        self.params = {'Z': 2.,
                       'b': np.array([0.6887, 0.6887, 1.3774]),
                       'omega': lambda Y: np.array([[0., 3347., (- 17697.
                                                                 - 38528.*Y[2]
                                                                 + 842570*np.power(Y[2], 5.)
                                                                 - 1549201*np.power(Y[2], 6.)
                                                                 + 962015.*np.power(Y[2], 7.))],
                                                    [0., 0., (-71500. +
                                                              240000. * np.power(Y[2], 7.))],
                                                    [0., 0., 0.]]),
                       'eta': lambda Y: np.array([[0., 0., -16.736 + 62.76*np.power(Y[2], 7.)],
                                                  [0., 0., 80.*(np.power(Y[2], 8.) -
                                                                np.power(0.5, 8.))],
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

        FeO_liq.set_state(P, T)
        MgO_liq.set_state(P, T)
        SiO2_liq.set_state(P, T)

        self.partial_gibbs = ( np.array( [ FeO_liq.gibbs,
                                           MgO_liq.gibbs,
                                           SiO2_liq.gibbs ] ) +
                               self.partial_gibbs_excesses )
        return sol


FMS = FMS_solution()

if __name__ == '__main__':
    # FeO
    temperatures = np.linspace(500., 2000., 101)
    Ss = np.empty_like(temperatures)
    Ss2 = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        FeO_liq.set_state(1.e5, T)
        wus.set_state(1.e5, T)
        Ss[i] = FeO_liq.S
        Ss2[i] = wus.S
        
    plt.plot(temperatures, Ss)
    plt.plot(temperatures, Ss2)
    plt.show()
    
    pressures = np.linspace(1.e5, 200.e9, 21)
    temperatures = np.empty_like(pressures)
    guess = 1650.
    for i, P in enumerate(pressures):
        print P/1.e9
        temperatures[i] = burnman.tools.equilibrium_temperature([wus, FeO_liq], [1.0, -1.0], P, max(2000., guess))
        guess = temperatures[i]


    fig1 = mpimg.imread('figures/Ozawa_2011_FeO_0_350_500_7000.png')
    plt.imshow(fig1, extent=[0., 350., 500., 7000.], aspect='auto')
    
    #fig1 = mpimg.imread('figures/Fischer_Campbell_2010_FeO_melting_0_80_1500_3500.png')
    fig1 = mpimg.imread('figures/Knittle_Jeanloz_1991_FeO_melting_0_120_0_7000.png')
    #plt.imshow(fig1, extent=[0., 80., 1500., 3500.], aspect='auto')
    plt.imshow(fig1, extent=[0., 120., 0., 7000.], aspect='auto')
    
    ZF2008 = np.loadtxt('data/FeO_melting_Zhang_Fei_2008.dat', unpack=True)
    plt.plot(ZF2008[0], ZF2008[2], marker='o', linestyle='None')
    KJ1991 = np.loadtxt('data/FeO_melting_Knittle_Jeanloz_1991.dat', unpack=True)
    mask = [i for (i, v) in enumerate(KJ1991[3]) if v==3]
    plt.plot(KJ1991[0][mask], KJ1991[1][mask], marker='o', linestyle='None')
    
    plt.plot(pressures/1.e9, temperatures)
    plt.show()    
    

    # MgO
    pressures = np.linspace(1.e5, 120.e9, 21)
    temperatures = np.empty_like(pressures)
    guess = 3098.
    for i, P in enumerate(pressures):
        print P/1.e9
        temperatures[i] = burnman.tools.equilibrium_temperature([per, MgO_liq], [1.0, -1.0], P, max(3000., guess))
        guess = temperatures[i] 
        
    fig1 = mpimg.imread('figures/Alfe_MgO_melting.png')
    plt.imshow(fig1, extent=[0., 150., 2000., 9000.], aspect='auto')
    plt.plot(pressures/1.e9, temperatures)
    plt.show()    
    


    # SiO2
    fig1 = mpimg.imread('figures/hp_sio2_melting.png')
    plt.imshow(fig1, extent=[0., 80., 1500., 5000.], aspect='auto')
    fig2 = mpimg.imread('figures/sio2_melting.png')
    plt.imshow(fig2, extent=[0., 15., 1673., 3273.], aspect='auto')
    
    
    temperatures = np.linspace(1500., 3000., 10)
    pressures = np.empty_like(temperatures)
    guess = 3000.
    for i, T in enumerate(temperatures):
        print T
        pressures[i] = burnman.tools.equilibrium_pressure([coe, stv], [1.0, -1.0], T, max(5.e9, guess))
        guess = pressures[i]
    plt.plot(pressures/1.e9, temperatures)
    
    
    pressures = np.linspace(0.5e9, 5.e9, 10)
    temperatures = np.empty_like(pressures)
    guess = 3000.
    for i, P in enumerate(pressures):
        print P/1.e9
        temperatures[i] = burnman.tools.equilibrium_temperature([qtz, SiO2_liq], [1.0, -1.0], P, max(3000., guess))
        guess = temperatures[i]
    plt.plot(pressures/1.e9, temperatures)
    
    pressures = np.linspace(4.5e9, 13.e9, 21)
    temperatures = np.empty_like(pressures)
    guess = 3000.
    for i, P in enumerate(pressures):
        print P/1.e9
        temperatures[i] = burnman.tools.equilibrium_temperature([coe, SiO2_liq], [1.0, -1.0], P, max(3000., guess))
        guess = temperatures[i]
    plt.plot(pressures/1.e9, temperatures)

    pressures = np.linspace(13.e9, 80.e9, 21)
    temperatures = np.empty_like(pressures)
    guess = 3000.
    for i, P in enumerate(pressures):
        print P/1.e9
        temperatures[i] = burnman.tools.equilibrium_temperature([stv, SiO2_liq], [1.0, -1.0], P, max(3000., guess))
        guess = temperatures[i]
    plt.plot(pressures/1.e9, temperatures)
    plt.show()    

