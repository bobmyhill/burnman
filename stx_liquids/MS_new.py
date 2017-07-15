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

    def _equilibrium_order(self, T, guess):
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
    
    def set_state(self, T, guess=None):

        molar_fractions = self.molar_fractions
        

        # Find partial gibbs
        # First, find vector towards MgO, FeO and SiO2
        dX = 0.001

        
        dA, XA = self._unit_vector_length(np.array([1., 0., 0.]) - self.molar_fractions)
        dA = dA*dX
        self.molar_fractions = molar_fractions + dA
        sol = self._equilibrium_order(T, guess)
        GA = self.delta_G
        
        dB, XB = self._unit_vector_length(np.array([0., 1., 0.]) - self.molar_fractions)
        dB = dB*dX
        self.molar_fractions = molar_fractions + dB
        sol = self._equilibrium_order(T, guess)
        GB = self.delta_G

        dC, XC = self._unit_vector_length(np.array([0., 0., 1.]) - self.molar_fractions)
        dC = dC*dX
        self.molar_fractions = molar_fractions + dC
        sol = self._equilibrium_order(T, guess)
        GC = self.delta_G


        self.molar_fractions = molar_fractions
        sol = self._equilibrium_order(T, guess)
        G0 = self.delta_G

        self.partial_gibbs_excesses = np.array( [ G0 + (GA - G0)/dX*XA,
                                                  G0 + (GB - G0)/dX*XB,
                                                  G0 + (GC - G0)/dX*XC ] )
        
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

P = 1.e5
sol = [0., 0., 0.]
for T in [3000.]:
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

    
    for y in [0.001, 0.5, 0.999]:
        c = []
        G = []
        H = []
        aSiO2 = []
        S = []
        for i, x in enumerate(xs):
            FMS.molar_fractions = np.array([(1. - x)*y, (1. - x)*(1. - y),  x])
            sol = FMS.set_state(T)
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



def enthalpy(T, prm):
    return prm.A + ( prm.a[0]*( T - 298.15 ) +
                     0.5  * prm.a[1]*( T*T - 298.15*298.15 ) +
                     -1.0 * prm.a[2]*( 1./T - 1./298.15 ) +
                     2.0  * prm.a[3]*(np.sqrt(T) - np.sqrt(298.15) ) +
                     -0.5 * prm.a[4]*(1./(T*T) - 1./(298.15*298.15) ) )

def entropy(T, prm):
    return prm.B + ( prm.a[0]*(np.log(T/298.15)) +
                     prm.a[1]*(T - 298.15) +
                     -0.5 * prm.a[2]*(1./(T*T) - 1./(298.15*298.15)) +
                     -2.0 * prm.a[3]*(1./np.sqrt(T) - 1./np.sqrt(298.15)) +
                     -1./3. * prm.a[4]*(1./(T*T*T) - 1./(298.15*298.15*298.15) ) )

class params():
    def __init__(self):
        self.A = 0.
        self.B = 0.
        self.a = [0., 0., 0., 0., 0.]

    
def G_SiO2_liquid(T):
    prm = params()
    if T < 1996.:
        prm.A = -214339.36
        prm.B = 12.148448
        prm.a = [19.960229, 0.e-3, -5.8684512e5, -89.553776, 0.66938861e8]
    else:
        prm.A = -221471.21
        prm.B = 2.3702523
        prm.a = [20.50000, 0., 0., 0., 0.]

    return 4.184*(enthalpy(T, prm) - T*entropy(T, prm))

def G_SiO2_liquid(T): # modified to fit melting curve...
    prm = params()
    prm.A = -2.20401092e+05
    prm.B = 4.78243403e+00
    prm.a = [1.82964211e+01,   6.76081909e-04, 0., 0., 0.]
    return 4.184*(enthalpy(T, prm) - T*entropy(T, prm))

def G_crst(T):
    prm = params()
    prm.A = -216629.36
    prm.B = 11.001147
    prm.a = [19.960229, 0.e-3, -5.8684512e5, -89.553776, 0.66938861e8]

    return 4.184*(enthalpy(T, prm) - T*entropy(T, prm))

def G_MgO_liquid(T):
    prm = params()
    prm.A = -130340.58
    prm.B = 6.4541207
    prm.a = [17.398557, -0.751e-3, 1.2494063e5, -70.793260, 0.013968958e8]

    return 4.184*(enthalpy(T, prm) - T*entropy(T, prm))

def G_per(T):
    prm = params()
    prm.A = -143761.95
    prm.B = 6.4415388
    prm.a = [14.605557, 0.e-3, -1.4845937e5, -70.793260, 0.013968958e8]

    return 4.184*(enthalpy(T, prm) - T*entropy(T, prm))


def G_fo(T):
    prm = params()
    prm.A = -520482.62
    prm.B = 22.468913
    prm.a = [57.036654, 0.e-3, 0.e5, -478.31286, -0.27782811e8]

    return 4.184*(enthalpy(T, prm) - T*entropy(T, prm))/3. # (1 cation basis)

def G_en(T):
    prm = params()
    prm.A = -368906.91
    prm.B = 16.117975
    prm.a = [39.813456, 0.e-3, -5.426786e5, -286.94742, 0.66718532e8]

    return 4.184*(enthalpy(T, prm) - T*entropy(T, prm))/2. # (1 cation basis)



# Find liquidus curves from stoichiometric phases
def MS_liquidus_temperature(temperature, G_solid, SiO2_fraction_solid, SiO2_fraction):
    T = temperature
    FMS.molar_fractions = np.array([0., (1. - SiO2_fraction),  SiO2_fraction])
    sol = FMS.set_state(T)
    G0 = FMS.delta_G

    
    G_mechanical = G_SiO2_liquid(T)*SiO2_fraction_solid + G_MgO_liquid(T)*(1. - SiO2_fraction_solid)
    
    dX = 1.e-2
    FMS.molar_fractions = np.array([0., (1. - SiO2_fraction - dX),  SiO2_fraction + dX])
    sol = FMS.set_state(T)
    G1 = FMS.delta_G

    dGxsdX = (G1 - G0)/dX
    Delta_X = SiO2_fraction_solid - SiO2_fraction
    
    G_xs = G0 + Delta_X*dGxsdX
    mu_phase_liq = G_mechanical + G_xs

    return G_solid(T) - mu_phase_liq

'''
def immiscibility(args, T):
    SiO2_fraction_0, SiO2_fraction_1 = args
    #print (SiO2_fraction_0, SiO2_fraction_1)
    dX = 1.e-2
    
    FMS.molar_fractions = np.array([0., (1. - SiO2_fraction_0),  SiO2_fraction_0])
    sol = FMS.set_state(T)
    G00 = FMS.delta_G
    FMS.molar_fractions = np.array([0., (1. - SiO2_fraction_0 - dX),  SiO2_fraction_0 + dX])
    sol = FMS.set_state(T)
    G10 = FMS.delta_G

    dGxsdX0 = (G10 - G00)/dX


    FMS.molar_fractions = np.array([0., (1. - SiO2_fraction_1),  SiO2_fraction_1])
    sol = FMS.set_state(T)
    G01 = FMS.delta_G
    FMS.molar_fractions = np.array([0., (1. - SiO2_fraction_1 - dX),  SiO2_fraction_1 + dX])
    sol = FMS.set_state(T)
    G11 = FMS.delta_G

    dGxsdX1 = (G11 - G01)/dX

    #print [dGxsdX1 - dGxsdX0,
    #       (G01 - G00) - dGxsdX0*(SiO2_fraction_1 - SiO2_fraction_0)]

    if SiO2_fraction_1 - SiO2_fraction_0 < 0.001:
        return [10000., 10000.]
    else:
        return [dGxsdX1 - dGxsdX0,
                (G01 - G00)/(SiO2_fraction_1 - SiO2_fraction_0) - dGxsdX0]

for temperature in np.linspace(1800., 2300., 11):
    print temperature - 273.15, root(immiscibility, [0.4, 0.99], args=(temperature), method='broyden2')
'''
    

# Plot MS phase diagram at 1 bar
fig1 = mpimg.imread('figures/Harvey_et_al_2015_MS_1bar.png')
plt.imshow(fig1, extent=[0, 1, 1500, 2000], aspect='auto')

curves = [[G_per, 0., 0.000001, 0.33, 21],
          [G_fo, 1./3., 0.29, 0.55, 21],
          [G_en, 0.5, 0.50, 0.61, 21],
          [G_crst, 1., 0.57, 0.64, 21]]

for curve in curves:
    G_solid, c_solid, X_min, X_max, n = curve
    X_SiO2 = np.linspace(X_min, X_max, n)
    
    temperatures = np.empty_like(X_SiO2)

    
    Tmin = 500.
    Tmax = 3680.

    if np.abs(c_solid - 0.5) < 1.e-5:
        Tmin = 1600.
        Tmax = 2000.
    if np.abs(c_solid - 1.) < 1.e-5:
        Tmin = 1300.
        Tmax = 2800.
        
    for i, X in enumerate(X_SiO2):
        sol = brentq(MS_liquidus_temperature, Tmin, Tmax, args=(G_solid, c_solid, X), full_output=True)
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
