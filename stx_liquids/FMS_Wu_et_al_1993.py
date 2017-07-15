import numpy as np
from scipy.optimize import fsolve, brentq
import matplotlib.pyplot as plt


import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    DKS_2013_liquids, \
    DKS_2013_solids, \
    SLB_2011
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

P=1.e5
T=1600. + 273.15
MgO_liq.set_state(P, T)
SiO2_liq.set_state(P, T)
MgO_gibbs = MgO_liq.gibbs
SiO2_gibbs = SiO2_liq.gibbs
MgO_S = MgO_liq.S
SiO2_S = SiO2_liq.S

fSis=[]
Gexs=[]
Sexs=[]
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
    Gex = phase.gibbs/sum_cations - (fSi*SiO2_gibbs + (1.-fSi)*MgO_gibbs) 
    Sex = phase.S/sum_cations - (fSi*SiO2_S + (1.-fSi)*MgO_S)  
    
    fSis.append(fSi)
    Gexs.append(Gex)
    Sexs.append(Sex)
    

#plt.plot(fSis, Sexs, marker='o', linestyle='None', label='DKS2013')


R = burnman.constants.gas_constant

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
                                                  [0., 0., 0.]])
                                            
        }

# Simultaneous equations for asymmetric ternary system
# The last member is the odd component
def asymmetric_pair_proportions(args, solution, T):
    # A = FeO, B = MgO, C = SiO2
    nAA, nAB, nAC, nBB, nBC, nCC  = np.abs(args)
    nA, nB, nC = solution.molar_fractions
    solution.npr = np.array([ [nAA, nAB, nAC],
                              [0., nBB, nBC],
                              [0., 0., nCC] ])
    solution.Xpr = solution.npr/np.sum(solution.npr)

    
    solution.X = solution.molar_fractions/np.sum(solution.molar_fractions) # normalise molar_fractions (if necessary)
    solution.Y = ( solution.params['b']*solution.molar_fractions /
                   np.sum(solution.params['b']*solution.molar_fractions) ) # equation 52 in PB1986


    solution.omega = solution.params['omega'](solution.Y)
    solution.eta = solution.params['eta'](solution.Y)
    solution.eqm = 4.*np.exp(-(2.*(solution.omega - solution.eta * T ) /
                               (solution.params['Z']*R*T)))
    
    out = []

    out.append(2.*solution.Xpr[0][0] + solution.Xpr[0][1] + solution.Xpr[0][2] - 2.*solution.Y[0]) 
    out.append(2.*solution.Xpr[1][1] + solution.Xpr[1][2] + solution.Xpr[0][1] - 2.*solution.Y[1]) 
    out.append(2.*solution.Xpr[2][2] + solution.Xpr[0][2] + solution.Xpr[1][2] - 2.*solution.Y[2]) 
        
    out.append(solution.eqm[0][1]*nAA*nBB - nAB*nAB)
    out.append(solution.eqm[0][2]*nAA*nCC - nAC*nAC)
    out.append(solution.eqm[1][2]*nBB*nCC - nBC*nBC)

    return out




# The following few lines plots the state of
# short range order in the subsystem MgO-FeO
# at a composition of Mg0.5Fe0.5O between 20 and 3000 K
'''
FMS = FMS_solution()
x = 0.9
FMS.molar_fractions = np.array([1. - x, x, 0.])

temperatures = np.linspace(20., 3000., 101)
Ts = []
FF = []
FM = []
MM = []

for i, T in enumerate(temperatures):
    sol =  fsolve(asymmetric_pair_proportions,
                  [FMS.molar_fractions[0], 1.e-10, 0.0,
                   FMS.molar_fractions[1], 1.e-10,
                   FMS.molar_fractions[2]],
                  args=(FMS, T), full_output=True)

    if sol[2] == 1:
        Ts.append(T)
        FF.append(sol[0][0])
        FM.append(sol[0][1])
        MM.append(sol[0][3])

plt.plot(Ts, FF, label='Fe-Fe')
plt.plot(Ts, FM, label='Fe-Mg')
plt.plot(Ts, MM, label='Mg-Mg')
plt.legend(loc='lower right')
plt.show()
'''


def set_state(solution, T, guess=None):

    Y_guess = ( solution.params['b']*solution.molar_fractions /
                np.sum(solution.params['b']*solution.molar_fractions) )

    guesses = []
    if guess != None:
        guesses = [guess]

    g0 = np.array([Y_guess[0], 0.0, 0.0,
                   Y_guess[1], 0.0,
                   Y_guess[2]])
    g1 = np.array([0., Y_guess[0] + Y_guess[1] - Y_guess[2], Y_guess[0] - Y_guess[1] + Y_guess[2],
                   0., -Y_guess[0] + Y_guess[1] + Y_guess[2],
                   0.])

    fs = np.linspace(0., 1., 21)
    for f in fs:
        guesses.append(g0*f + g1*(1. - f))
    
    try:
        guesses.append(solution.pairs)
    except:
        i=0
        
    solution_found = False
    i=0
    while solution_found == False and i < len(guesses):
        sol =  fsolve(asymmetric_pair_proportions,
                      guesses[i], args=(solution, T), full_output=True)

        if sol[2] == 1:
            solution_found = True
        else:
            i += 1
             
    if solution_found == False: 
        print sol[3]
        return 0
    else:
        solution.pairs = sol[0]

        # Equations 3, 4, 5.
        # note modifications to each of the equations in the text
        solution.delta_H = np.sum(solution.params['b'] * solution.X) * ( np.sum(solution.Xpr * solution.omega) ) / 2.
    
        delta_Sc = -R *(np.sum([xi * np.log(xi) for xi in solution.X if xi > 1.e-12]) +
                        solution.params['Z'] / 2. * np.sum(solution.params['b'] * solution.X) *
                        ( np.sum( [ solution.Xpr[i][i] * np.log(solution.Xpr[i][i] /
                                                                (solution.Y[i]*solution.Y[i]))
                                    for i in range(len(solution.Y)) if solution.Y[i] > 1.e-12] )
                          
                          + np.sum(np.sum( [ [ solution.Xpr[i][j] * np.log(solution.Xpr[i][j] /
                                                                           (2.*solution.Y[i]*solution.Y[j]))
                                               for i in range(len(solution.Y))
                                               if j > i and solution.Y[i] > 1.e-12 and solution.Y[j] > 1.e-12 ]
                                             for j in range(len(solution.Y)) ] )) ) )
        
        delta_Snc = ( np.sum(solution.params['b'] * solution.X) *
                      np.sum(solution.Xpr * solution.eta) ) / 2.
        
        
        solution.delta_S = delta_Sc + delta_Snc
        solution.delta_G = solution.delta_H - T*solution.delta_S
        return 1




FMS = FMS_solution()
T=1600. + 273.
xs = np.linspace(0.001, 0.999, 101)

#fig1 = mpimg.imread('figures/MgO_SiO2_gibbs_mixing_-10_0_kcal_Pelton_Blander_1986.png')
#plt.imshow(fig1, extent=[0, 1, -41840, 0], aspect='auto')


sol = [0., 0., 0.]
for T in [1000.]:
    for y in [1.0]:
        c = []
        G = []
        S = []
        for i, x in enumerate(xs):
            FMS.molar_fractions = np.array([(1. - x)*y, (1. - x)*(1. - y),  x])
            sol = set_state(FMS, T)
            if sol != 1:
                print 'No solution found'
            else:
                G0 = FMS.delta_G
                S0 = FMS.delta_S
                c.append(x)
                G.append(G0)
                S.append(S0)
            
        c = np.array(c)
        G = np.array(G)
        S = np.array(S)
        plt.plot(c, S, label=str(T)+'; Fe/(Mg+Fe) = '+str(y))
        
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



             

# Find liquidus curves from stoichiometric phases
def MS_liquidus_temperature(temperature, G_solid, SiO2_fraction_solid, SiO2_fraction):
    T = temperature
    FMS.molar_fractions = np.array([0., (1. - SiO2_fraction),  SiO2_fraction])
    sol = set_state(FMS, T)
    G0 = FMS.delta_G

    
    G_mechanical = G_SiO2_liquid(T)*SiO2_fraction_solid + G_MgO_liquid(T)*(1. - SiO2_fraction_solid)
    
    dX = 1.e-2
    FMS.molar_fractions = np.array([0., (1. - SiO2_fraction - dX),  SiO2_fraction + dX])
    sol = set_state(FMS, T)
    G1 = FMS.delta_G

    dGxsdX = (G1 - G0)/dX
    Delta_X = SiO2_fraction_solid - SiO2_fraction
    
    G_xs = G0 + Delta_X*dGxsdX
    mu_phase_liq = G_mechanical + G_xs

    return G_solid(T) - mu_phase_liq



X_SiO2 = np.linspace(0.01, 0.3, 101)
temperatures = np.empty_like(X_SiO2)

Tguess = 3080.
for i, X in enumerate(X_SiO2):
    sol = brentq(MS_liquidus_temperature, Tguess - 200., Tguess + 200., args=(G_per, 0., X), full_output=True)
    if sol[1].converged == True:
        temperatures[i] = sol[0]
        Tguess = temperatures[i]
        print Tguess
    else:
        temperatures[i] = 2000.
plt.plot(X_SiO2, temperatures)
plt.show()
