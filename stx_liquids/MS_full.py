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
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

phases = [DKS_2013_liquids.MgO_liquid(),
          DKS_2013_liquids.Mg5SiO7_liquid(),
          DKS_2013_liquids.Mg2SiO4_liquid(),
          DKS_2013_liquids.Mg3Si2O7_liquid(),
          DKS_2013_liquids.MgSiO3_liquid(),
          DKS_2013_liquids.MgSi2O5_liquid(),
          DKS_2013_liquids.MgSi3O7_liquid(),
          DKS_2013_liquids.MgSi5O11_liquid(),
          DKS_2013_liquids.SiO2_liquid()]

MgO_liq = DKS_2013_liquids.MgO_liquid()
SiO2_liq = DKS_2013_liquids.SiO2_liquid()


per = SLB_2011.periclase()
per.params['F_0'] += -40036.738
per.params['q_0'] = 0.15
fo = HP_2011_ds62.fo()
en = HP_2011_ds62.en()
crst = HP_2011_ds62.crst()

Tm = 3098.
per.set_state(1.e5, Tm)
MgO_liq.set_state(1.e5, Tm)
dS = 169.595 - 144.532 # JANAF
MgO_liq.property_modifiers = [['linear', {'delta_E': (per.gibbs - MgO_liq.gibbs) + Tm*(per.S - MgO_liq.S + dS), 'delta_S': (per.S - MgO_liq.S + dS), 'delta_V': 0.}]]

Hm = 8900.
Tm = 1999.
Sm = Hm/Tm
crst = HP_2011_ds62.crst()
crst.set_state(1.e5, Tm)
SiO2_liq.set_state(1.e5, Tm)
SiO2_liq.property_modifiers = [['linear', {'delta_E': (crst.gibbs - SiO2_liq.gibbs) + Tm*(crst.S - SiO2_liq.S + Sm), 'delta_S': (crst.S - SiO2_liq.S + Sm), 'delta_V': 0.}]]

def heat_capacity(T, prm):
    return ( prm['a'][0] +
             prm['a'][1]*T +
             prm['a'][2]/(T*T) +
             prm['a'][3]/np.sqrt(T) +
             prm['a'][4]/(T*T*T) )

def enthalpy(T, prm):
    return prm['A'] + ( prm['a'][0]*( T - 298.15 ) +
                     0.5  * prm['a'][1]*( T*T - 298.15*298.15 ) +
                     -1.0 * prm['a'][2]*( 1./T - 1./298.15 ) +
                     2.0  * prm['a'][3]*(np.sqrt(T) - np.sqrt(298.15) ) +
                     -0.5 * prm['a'][4]*(1./(T*T) - 1./(298.15*298.15) ) )

def entropy(T, prm):
    return prm['B'] + ( prm['a'][0]*(np.log(T/298.15)) +
                     prm['a'][1]*(T - 298.15) +
                     -0.5 * prm['a'][2]*(1./(T*T) - 1./(298.15*298.15)) +
                     -2.0 * prm['a'][3]*(1./np.sqrt(T) - 1./np.sqrt(298.15)) +
                     -1./3. * prm['a'][4]*(1./(T*T*T) - 1./(298.15*298.15*298.15) ) )

'''
class SiO2_liquid(): # modified to fit melting curve, melting corresponds to HP_2011_ds62.crst at 1999.
    def __init__(self):
        self.prm = {'A': -890687.,
                     'B': 65.,
                     'a': np.array([7.177548e+01,   1.55168e-02,   2.489344e+05,
                                    -7.9021e+02, 0.])
                    }
        
    def set_state(self, P, T):
        self.gibbs = (enthalpy(T, self.prm) - T*entropy(T, self.prm))
        self.H = (enthalpy(T, self.prm))
        self.S = (entropy(T, self.prm))
        self.heat_capacity_p = (heat_capacity(T, self.prm))

class MgO_liquid():
    def __init__(self):
        self.prm = {'A': -129346.,
                    'B': 9.04,
                    'a': [1.29253257e+01,   9.44364230e-04,  -3.72469663e+05, 7.91765434e+00, 0.]
                    }
    def set_state(self, P, T):
        self.gibbs = 4.184*(enthalpy(T, self.prm) - T*entropy(T, self.prm))
        self.H = 4.184*(enthalpy(T, self.prm))
        self.S = 4.184*(entropy(T, self.prm))
        self.heat_capacity_p = 4.184*(heat_capacity(T, self.prm))
        
        
SiO2_liq = SiO2_liquid()
MgO_liq = MgO_liquid()
'''

class MgO_liquid_tweak(): 
    def __init__(self):
        self.prm0 = {'A': 54178.83752719708, 'a': [2.839602347217558, 0.00341484389174643, -1502552.5191544946, -922.37318553906005, 0.0], 'B': 66.993850170202407}
        self.prm1 = {'A': 58562.842558220582, 'a': [5.7165061030308122, -1.0221081120981808e-05, -6269356.7755687768, -722.99967198621596, 0.0], 'B': 79.347606239344714}
        
    def set_state(self, P, T):
        if T < 1750.:
            self.prm = self.prm0
        else:
            self.prm = self.prm1
        self.gibbs = (enthalpy(T, self.prm) - T*entropy(T, self.prm))
        self.H = (enthalpy(T, self.prm))
        self.S = (entropy(T, self.prm))
        self.heat_capacity_p = (heat_capacity(T, self.prm))
        self.gibbs = 0.
        self.H = 0.
        self.S = 0.
        self.heat_capacity_p = 0.


class SiO2_liquid_tweak(): 
    def __init__(self):
        self.prm0 = {'A': 29317.598670186722, 'a': [3.14355762533431, 0.0010122323779954451, -17272.057503375127, -666.69586046751851, 0.0], 'B': 39.790628572419884}
        self.prm1 = {'A': -75643.03446637874, 'a': [138.45503432848142, -0.018434113923524766, 64483238.833861053, -5613.4139101984401, 0.0], 'B': -187.76074293402641}
        
    def set_state(self, P, T):
        if T < 2250.:
            self.prm = self.prm0
        else:
            self.prm = self.prm1
        self.gibbs = (enthalpy(T, self.prm) - T*entropy(T, self.prm))
        self.H = (enthalpy(T, self.prm))
        self.S = (entropy(T, self.prm))
        self.heat_capacity_p = (heat_capacity(T, self.prm))
        self.gibbs = 0.
        self.H = 0.
        self.S = 0.
        self.heat_capacity_p = 0.

MgO_tweak = MgO_liquid_tweak()
SiO2_tweak = SiO2_liquid_tweak()


R = burnman.constants.gas_constant

class FMS_solution():
    def __init__(self, molar_fractions=None):
        self.name = 'FeO-MgO-SiO2 solution'

        z = 2.
        r = 0.38
        b1 = -(r*np.log(r)/(1. - r) + np.log(1. - r))/(z*np.log(2.))
        b2 = b1*(1. - r)/r
        
        '''
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
        '''
        self.params = {'Z': z,
                       'b': np.array([b1, b1, b2]),
                       'omega': lambda Y: np.array([[0., 3347., (- 17697.
                                                                 - 38528.*Y[2]
                                                                 + 842570*np.power(Y[2], 5.)
                                                                 - 1549201*np.power(Y[2], 6.)
                                                                 + 962015.*np.power(Y[2], 7.))],
                                                    [0., 0., (-63000. +
                                                              -32000. * np.power(Y[1], 3.) + 
                                                              22000. * np.power(Y[2], 3.) +
                                                              205000. * np.power(Y[2], 7.))],
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

        MgO_liq.set_state(P, T)
        SiO2_liq.set_state(P, T)
        MgO_tweak.set_state(P, T)
        SiO2_tweak.set_state(P, T)

        self.partial_gibbs = ( np.array( [ 0.,
                                           MgO_liq.gibbs + MgO_tweak.gibbs,
                                           SiO2_liq.gibbs + SiO2_tweak.gibbs ] ) +
                               self.partial_gibbs_excesses )
        return sol
    




FMS = FMS_solution()
T=1600. + 273.
xs = np.linspace(0.001, 0.999, 101)

'''
plt.subplot(222)
fig1 = mpimg.imread('figures/Harvey_et_al_2015_MS_enthalpy.png')
plt.imshow(fig1, extent=[0, 1, -40000, 50000], aspect='auto')
plt.subplot(223)
fig1 = mpimg.imread('figures/Harvey_et_al_2015_MS_entropy.png')
plt.imshow(fig1, extent=[0, 1, -2., 10.], aspect='auto')
'''


stv = SLB_2011.stishovite()
stv.set_state(13.7e9, 3023.)
SiO2_liq.set_state(13.7e9, 3023.)
stv.params['F_0'] += SiO2_liq.gibbs - stv.gibbs


per_DKS = DKS_2013_solids.periclase()
stv_DKS =  DKS_2013_solids.stishovite()
MgO_liq_DKS = DKS_2013_liquids.MgO_liquid()
SiO2_liq_DKS = DKS_2013_liquids.SiO2_liquid()

sol = [0., 0., 0.]

P = 1.e5
temperatures = [1800., 2163.]
for T in temperatures:
    
    print T
    MgO_liq.set_state(P, T)
    per.set_state(P, T)
    SiO2_liq.set_state(P, T)
    stv.set_state(P, T)
    
    plt.subplot(221)

    
    min_set = [[SLB_2011.periclase(), SLB_2011.stishovite(),
                SLB_2011.mg_perovskite(), SLB_2011.forsterite(),
                SLB_2011.enstatite()],
               [HP_2011_ds62.per(), HP_2011_ds62.stv(),
                HP_2011_ds62.mpv(), HP_2011_ds62.fo(),
                HP_2011_ds62.en()]]
    for (per2, stv2, mpv2, fo2, en2) in min_set:
        per2.set_state(P, T)
        stv2.set_state(P, T)
        mpv2.set_state(P, T)
        fo2.set_state(P, T)
        en2.set_state(P, T)
        per_DKS.set_state(P, T)
        stv_DKS.set_state(P, T)
        MgO_liq_DKS.set_state(P, T)
        SiO2_liq_DKS.set_state(P, T)
        dG_pv = mpv2.gibbs - per2.gibbs - stv2.gibbs
        dG_fo = fo2.gibbs - 2.*per2.gibbs - stv2.gibbs
        dG_en = en2.gibbs - 2.*per2.gibbs - 2.*stv2.gibbs
        
        #plt.plot([0., 0.5, 1.], [per_DKS.gibbs - MgO_liq_DKS.gibbs,
        #                         0.5*(per_DKS.gibbs + stv_DKS.gibbs) -
        #                         0.5*(MgO_liq_DKS.gibbs + SiO2_liq_DKS.gibbs)
        #                         + 0.5*dG_pv,
        #                         stv_DKS.gibbs - SiO2_liq_DKS.gibbs], label=str(T)+' K') 
        plt.plot([0., 1./3., 1./2.], [per_DKS.gibbs - MgO_liq_DKS.gibbs,
                                      (2./3.*per_DKS.gibbs + 1./3.*stv_DKS.gibbs) -
                                      (2./3.*MgO_liq_DKS.gibbs + 1./3.*SiO2_liq_DKS.gibbs)
                                      + 1./3.*dG_fo,
                                      0.5*(per_DKS.gibbs + stv_DKS.gibbs) -
                                      0.5*(MgO_liq_DKS.gibbs + SiO2_liq_DKS.gibbs)
                                      + 1./4.*dG_en], label=str(T)+' K') 
        
    phases[0].set_state(P, T)
    phases[-1].set_state(P, T)
    MgO_gibbs = phases[0].gibbs
    SiO2_gibbs = phases[-1].gibbs

    MgO_H = phases[0].H
    SiO2_H = phases[-1].H

    MgO_S = phases[0].S
    SiO2_S = phases[-1].S

    MgO_V = phases[0].V
    SiO2_V = phases[-1].V

    MgO_K_T = phases[0].K_T
    SiO2_K_T = phases[-1].K_T

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
    plt.title('Excess Gibbs') 
    plt.plot(fSis, Gexs, marker='o', linestyle='None', label=str(P/1.e9)+' GPa, '+str(T)+' K')
    plt.subplot(222)
    plt.title('Excess Enthalpies') 
    plt.plot(fSis, Hexs, marker='o', linestyle='None', label=str(P/1.e9)+' GPa, '+str(T)+' K')
    plt.subplot(223)
    plt.title('Excess Entropies') 
    plt.plot(fSis, Sexs, marker='o', linestyle='None', label=str(P/1.e9)+' GPa, '+str(T)+' K')
    plt.subplot(224)
    plt.title('SiO2 activities')
    plt.plot([0., 0.], [1., 1.])

    
    for y in [0.001]:
        c = []
        G = []
        H = []
        aSiO2 = []
        S = []
        for i, x in enumerate(xs):
            FMS.molar_fractions = np.array([(1. - x)*y, (1. - x)*(1. - y),  x])
            sol = FMS.set_state(P, T)
            if sol != 1:
                print 'No solution found'
            else:
                c.append(x)
                G.append(FMS.delta_G)
                S.append(FMS.delta_S)
                H.append(FMS.delta_H)
                aSiO2.append(np.exp(FMS.partial_gibbs_excesses[2]/(R*T)))
            
        c = np.array(c)
        G = np.array(G)
        H = np.array(H)
        S = np.array(S)
        aSiO2 = np.array(aSiO2)
        plt.subplot(221)
        plt.plot(c, G, label=str(T)+'; Fe/(Mg+Fe) = '+str(y))
        plt.subplot(222)
        plt.plot(c, H, label=str(T)+'; Fe/(Mg+Fe) = '+str(y))
        plt.subplot(223)
        plt.plot(c, S, label=str(T)+'; Fe/(Mg+Fe) = '+str(y))
        plt.subplot(224)
        plt.plot(c, aSiO2, label=str(T)+'; Fe/(Mg+Fe) = '+str(y))
 
plt.subplot(221)       
plt.legend(loc='lower right')
plt.xlabel('X Si')
plt.show()



def MS_liquidus_temperature(temperature, pressure, solid, n_cations_solid, SiO2_fraction_solid, SiO2_fraction):
    # Find liquidus curves from stoichiometric phases
    
    solid.set_state(pressure, temperature)
    FMS.molar_fractions = np.array([0.001, (1. - SiO2_fraction),  SiO2_fraction - 0.001])
    FMS.set_state(pressure, temperature)
    mu_phase_liq = ( FMS.partial_gibbs[1]*( 1. - SiO2_fraction_solid ) +
                     FMS.partial_gibbs[2]*SiO2_fraction_solid ) * n_cations_solid
    return solid.gibbs - mu_phase_liq
    

# Plot MS phase diagram at 1 bar
fig1 = mpimg.imread('figures/Harvey_et_al_2015_MS_1bar.png')
plt.imshow(fig1, extent=[0, 1, 1500, 2000], aspect='auto')

curves = [[per, 1., 0., 0.001, 0.33, 11],
          [fo, 3., 1./3., 0.29, 0.55, 11],
          [en, 4., 1./2., 0.50, 0.61, 11],
          [crst, 1., 1., 0.57, 0.64, 6]]

for P in [1.e5]:
    for curve in curves:
        solid, nc_solid, c_solid, X_min, X_max, n = curve
        X_SiO2 = np.linspace(X_min, X_max, n)
    
        temperatures = np.empty_like(X_SiO2)
        
    
        Tmin = 1400.
        Tmax = 3200. + P/1.e9*80.
                
        for i, X in enumerate(X_SiO2):
            sol = brentq(MS_liquidus_temperature, Tmin, Tmax, args=(P, solid, nc_solid, c_solid, X), full_output=True)
            if sol[1].converged == True:
                temperatures[i] = sol[0]
                Tmin = temperatures[i] - 300.
                Tmax = temperatures[i] + 300.
                print X, temperatures[i]
            else:
                temperatures[i] = 2000.
        plt.plot(X_SiO2, temperatures-273.15)
plt.ylabel('Temperature (C)')
plt.xlabel('X SiO2')
plt.show()
