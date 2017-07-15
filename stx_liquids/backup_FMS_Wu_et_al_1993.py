import numpy as np
from scipy.optimize import fsolve
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

'''
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
T=1400.
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
    

plt.plot(fSis, Sexs, marker='o', linestyle='None', label='DKS2013')
'''





R = burnman.constants.gas_constant

w_FM = lambda Y: 3347.
eta_FM = lambda Y: 0.

w_MS = lambda Y: -86090 - 48974*Y + 328109*np.power(Y, 7.) # Y = Y_SiO2
eta_MS = lambda Y: -37.656*Y + 125.52*np.power(Y, 7.) # Y = Y_SiO2


# Alternative expression from Blander and Pelton, 1986 (reported in cal)
w_MS = lambda Y: (-33976. + 53760.*np.power(Y, 3.)
                  - 107429.*np.power(Y, 5.)
                  + 126025.*np.power(Y, 7.))*4.148 # Y = Y_SiO2
eta_MS = lambda Y: (-6. + 20.*np.power(Y, 7.))*4.148 # Y = Y_SiO2


w_FS = lambda Y: (- 17697.
                  - 38528.*Y
                  + 842570*np.power(Y, 5.)
                  - 1549201*np.power(Y, 6.)
                  + 962015.*np.power(Y, 7.)) # Y = Y_SiO2
eta_FS = lambda Y: -16.736 + 62.76*np.power(Y, 7.) # Y = Y_SiO2


'''
f = np.array([1., 1., 2.])
b = 1.3774/2.*f

# Simultaneous equations for binary system
def find_binary_pair_proportions(args, n, Z, eqm_FM):
    nAA, nBB, nAB = args
    nA, nB = n

    out = []
    out.append(2.*nAA + nAB - Z*nA)
    out.append(2.*nBB + nAB - Z*nB)
    out.append(eqm_FM*nAA*nBB - nAB*nAB)

    return out

temperatures = np.linspace(1., 5000., 101)
X_ijs = []

for i, T in enumerate(temperatures):
    eqm_FM = 4.*np.exp(-(2.*(w_FM(0.) - eta_FM(0.)*T)/(Z_FM*R*T)))
    pair_amounts = fsolve(find_binary_pair_proportions, [1., 1., 1.], args=([1., 1.], Z_FM, eqm_FM))
    X_ijs.append(pair_amounts/np.sum(pair_amounts))


plt.plot(temperatures, zip(*X_ijs)[0], label='Fe-Fe (or Mg-Mg)')
plt.plot(temperatures, zip(*X_ijs)[2], label='Fe-Mg')
plt.legend(loc='lower right')
plt.show()
'''

class FMS_solution():
    def __init__(self, molar_fractions=None):
        self.name = 'FeO-MgO-SiO2 solution'
        self.params = {'Z': np.array([2., 2., 2.]),
                       'b': np.array([0.6887, 0.6887, 1.3774])}
    

# Simultaneous equations for asymmetric ternary system
# The last member is the odd component
def asymmetric_pair_proportions(args, solution, T):
    # A = FeO, B = MgO, C = SiO2
    nAA, nAB, nAC, nBB, nBC, nCC  = np.abs(args)
    nA, nB, nC = solution.molar_fractions
    Z_A, Z_B, Z_C = solution.params['Z']

    solution.npr = np.array([ [nAA, nAB, nAC],
                              [0., nBB, nBC],
                              [0., 0., nCC] ])
    solution.Xpr = solution.npr/np.sum(solution.npr)

    
    solution.X = solution.molar_fractions/np.sum(solution.molar_fractions) # normalise molar_fractions (if necessary)
    solution.Y = ( solution.params['b']*solution.molar_fractions /
                   np.sum(solution.params['b']*solution.molar_fractions) ) # equation 52 in PB1986

    omega = [ w_FM(0.), w_FS(solution.Y[2]), w_MS(solution.Y[2]) ]
    eta = [ eta_FM(0.), eta_FS(solution.Y[2]), eta_MS(solution.Y[2]) ]

    solution.omega = [ [ 0., omega[0], omega[1] ],
                       [ 0., 0., omega[2] ],
                       [ 0., 0., 0. ] ]
    
    solution.eta = [ [ 0., eta[0], eta[1] ],
                     [ 0., 0., eta[2] ],
                     [ 0., 0., 0. ] ]

    z = 2.
    solution.eqm = [ 4.*np.exp(-(2.*(omega[i] - eta[i] * T ) /
                                 (z*R*T))) for i in range(len(omega)) ] 
    
    out = []
    #out.append(2.*nAA + nAB + nAC - Z_A*nA)
    #out.append(nAB + 2.*nBB + nBC - Z_B*nB)
    #out.append(nAC + nBC + 2.*nCC - Z_C*nC)

    out.append(2.*solution.Xpr[0][0] + solution.Xpr[0][1] + solution.Xpr[0][2] - 2.*solution.Y[0]) 
    out.append(2.*solution.Xpr[1][1] + solution.Xpr[1][2] + solution.Xpr[0][1] - 2.*solution.Y[1]) 
    out.append(2.*solution.Xpr[2][2] + solution.Xpr[0][2] + solution.Xpr[1][2] - 2.*solution.Y[2]) 
        
    out.append(solution.eqm[0]*nAA*nBB - nAB*nAB)
    out.append(solution.eqm[1]*nAA*nCC - nAC*nAC)
    out.append(solution.eqm[2]*nBB*nCC - nBC*nBC)

    return out


FMS = FMS_solution()


# The following few lines plots the state of
# short range order in the subsystem MgO-FeO
# at a composition of Mg0.5Fe0.5O between 20 and 3000 K
'''
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

T=1600. + 273.
xs = np.linspace(0.001, 0.999, 101)

fig1 = mpimg.imread('figures/MgO_SiO2_gibbs_mixing_-10_0_kcal_Pelton_Blander_1986.png')
plt.imshow(fig1, extent=[0, 1, -41840, 0], aspect='auto')

sol = [0., 0., 0.]
for y in [0.0, 0.25, 0.5, 0.75, 1.0]:
    c = []
    G = []
    for i, x in enumerate(xs):
        FMS.molar_fractions = np.array([(1. - x)*y, (1. - x)*(1. - y),  x])
        if sol[2] == 1:
            guess = sol[0]
        else:
            guess = [FMS.molar_fractions[0], 0.0, 0.0,
                     FMS.molar_fractions[1], 0.0,
                     FMS.molar_fractions[2]]
            
        sol =  fsolve(asymmetric_pair_proportions,
                      guess, args=(FMS, T), full_output=True)

        if sol[2] != 1:
            print sol[3]
        else:
            
            # Equations 3, 4, 5.
            # note modifications to each of the equations in the text
            FMS.delta_H = np.sum(FMS.params['b'] * FMS.X) * ( np.sum(FMS.Xpr * FMS.omega) ) / 2.
            
            z = 2.
            delta_Sc = -R *(np.sum([xi * np.log(xi) for xi in FMS.X if xi > 1.e-12]) +
                            z / 2. * np.sum(FMS.params['b'] * FMS.X) *
                            ( np.sum( [ FMS.Xpr[i][i] * np.log(FMS.Xpr[i][i] /
                                                               (FMS.Y[i]*FMS.Y[i]))
                                        for i in range(len(FMS.Y)) if FMS.Y[i] > 1.e-12] )
                              
                              + np.sum(np.sum( [ [ FMS.Xpr[i][j] * np.log(FMS.Xpr[i][j] /
                                                                          (2.*FMS.Y[i]*FMS.Y[j]))
                                                   for i in range(len(FMS.Y))
                                                   if j > i and FMS.Y[i] > 1.e-12 and FMS.Y[j] > 1.e-12 ]
                                                 for j in range(len(FMS.Y)) ] )) ) )
            
            delta_Snc = ( np.sum(FMS.params['b'] * FMS.X) *
                          np.sum(FMS.Xpr * FMS.eta) ) / 2.

            
                
            FMS.delta_S = delta_Sc + delta_Snc
            FMS.delta_G = FMS.delta_H - T*FMS.delta_S
            c.append(x)
            G.append(FMS.delta_G)

            #G.append((np.array(FMS.omega) - T*np.array(FMS.eta))[1][2]/4.184/1000.)

    c = np.array(c)
    G = np.array(G)
    plt.plot(c, G, label=str(T)+'; Fe/(Mg+Fe) = '+str(y))
    
plt.legend(loc='lower right')
plt.xlabel('X Si')
plt.show()

