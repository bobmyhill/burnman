import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os, sys
sys.path.insert(1,os.path.abspath('..'))

R = 8.31446

class KMgF_solution():
    def g_KF(self, T):
        return 0.

    def g_MgF2(self, T):
        return 0.

    def g_SiO2(self, T):
        return 0.
        
    def __init__(self, molar_fractions=None):
        self.name = 'KF-MgF2-LiF liquid solution'
        self.params = {
            'asymmetric_systems': [(0, 1, 2)], # last component is the odd one out i.e. SiO2 is the asymmetric component in the system FeII, FeIII, O
            'g_std': lambda T: np.array( [self.g_KF(T), self.g_MgF2(T), self.g_SiO2(T)] ),
            'Z': np.array( [ [6., 3., 2.],
                             [6., 6., 2.],
                             [2., 2., 2.] ] ),
            'g': lambda T, chi: np.array( [ [ 0., (-23974.4 + 6.8813*T +
                                                   (1730.7 - 3.2421*T)*chi[0][1] +
                                                   (-35478.0 + 2.000*T)*0. +
                                                   18875.0*chi[0][1]*chi[0][1]), 0.],
                                            [ 0., 0., 0.],
                                            [ 0., 0., 0. ] ] ),
            'ternary_q': lambda T, Y, chi: [ [0., 0., 0.],
                                             [0., 0., 0.],
                                             [0., 0., 0.] ]
        }





def pair_proportions(list_npr, solution, T):
    # The list of pair proportions should be in the order
    # n00, n01, n02, ... n11, n12, ... 
    # i.e. upper triangular form
    
    n_mbrs = len(solution.molar_fractions)
    solution.npr = np.zeros((n_mbrs, n_mbrs))
    i = 0
    j = 0
    for n in np.abs(list_npr):
        solution.npr[i][j] = n
        solution.npr[j][i] = n
        i += 1
        if i == n_mbrs:
            j += 1
            i = j
            
    solution.Xpr = solution.npr/np.sum(np.abs(list_npr))
    
    #solution.Z = np.zeros((n_mbrs))
    solution.Z = np.array([solution.params['Z'][i][i] for i in range(n_mbrs)]) # initialize Zs
    for i in range(n_mbrs):
        sumn = solution.npr[i][i] + np.sum( [solution.npr[i][j] for j in range(n_mbrs)] )
        if i == 0:
            solution.sumn = sumn # bug finding only
        if sumn != 0.:
            solution.Z[i] = 1. / ( 1. / sumn *
                                   ( 2.*solution.npr[i][i]/solution.params['Z'][i][i] +
                                     np.sum( [solution.npr[i][j]/solution.params['Z'][i][j]
                                              for j in range(n_mbrs)
                                              if i != j] ) ) ) # eqns 12,13 of PC2001
                       
    solution.n = np.array( [ ( 2.*solution.npr[i][i]/solution.params['Z'][i][i] +
                               np.sum( [solution.npr[i][j]/solution.params['Z'][i][j]
                                        for j in range(n_mbrs)
                                        if i != j] ) )
                             for i in range(n_mbrs)] ) # eqn 14 of PC2001    
    
    solution.X = solution.n/np.sum(solution.n) # eqn 4 of PC2001
    solution.Y = solution.Z * solution.n / np.sum(solution.Z * solution.n) # eqn 5 of PC2001

    solution.chi = np.zeros((n_mbrs, n_mbrs))
    for i in range(n_mbrs): 
        for j in range(n_mbrs):
            if i != j:
                k = [[a, b] for (a, b, c) in solution.params['asymmetric_systems']
                     if (c == j) and (a == i or b == i)]
                k.append([i, i])
                k = list(set([item for sublist in k for item in sublist])) # flatten and find unique values
                l = [[a, b] for (a, b, c) in solution.params['asymmetric_systems']
                     if (c == i) and (a == j or b == j)]
                l.append([j, j])
                l = list(set([item for sublist in l for item in sublist])) # flatten and find unique values
                
                kl = list(set([item for sublist in [k, l] for item in sublist]))
                num = ( np.sum( [ np.sum( [ solution.Xpr[m][n]
                                            for m in k if m >= n] )
                              for n in k ] ) )
                denom = ( np.sum( [ np.sum( [ solution.Xpr[m][n]
                                              for m in kl if m >= n] )
                                    for n in kl ] ) )

                #print i, j, k, l, num, denom
                if denom == 0.:
                    solution.chi[i][j] = 0. # only happens if the component doesn't exist
                else:
                    solution.chi[i][j] = num / denom # eqns 27,29 of PC2001

    #print solution.npr
    #print solution.chi
    
    #exit()
    #print 'WARNING still need to sort out ternary terms (eqn 30 in PC2001)'
    solution.delta_g = ( solution.params['g'](T, solution.chi) +
                         solution.params['ternary_q'](T, solution.Y, solution.chi) ) # eqns 30,24 of PC2001
    
    out = list(solution.molar_fractions - solution.n )
    solution.eqm = np.zeros((n_mbrs, n_mbrs))
    for i in range(n_mbrs):
        for j in range(i+1, n_mbrs):
            solution.eqm[i][j] = 4. * np.exp( -solution.delta_g[i][j] /
                                              (R*T) ) # eqn 13 of Petal2000
                                  
            out.append( ( solution.eqm[i][j]*solution.Xpr[i][i]*solution.Xpr[j][j] -
                          solution.Xpr[i][j]*solution.Xpr[i][j] ) )

    return out


def set_state(solution, T, guesses = None):
    if guesses == None:
        try:
            guesses = [solution.pairs]
        except:
            guesses = []
    else:
        guesses = [guesses]
            
    guesses.append([solution.molar_fractions[0], 0.0, 0.0,
                  solution.molar_fractions[1], 0.0,
                  solution.molar_fractions[2]])
    solution_found = False
    i = 0 
    while solution_found == False and i < len(guesses):
        sol =  fsolve(pair_proportions,
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
        delta_Sc = -R *(np.sum([solution.n[i] * np.log(xi)
                                for i, xi in enumerate(solution.X) if xi > 1.e-12]) +
                        np.sum([solution.npr[i][i] * np.log(solution.Xpr[i][i] /
                                                            (solution.Y[i]*solution.Y[i]))
                                for i in range(len(solution.Y)) if solution.Y[i] > 1.e-12]) +
                        np.sum(np.sum( [ [ solution.npr[i][j] * np.log(solution.Xpr[i][j] /
                                                                       (2.*solution.Y[i]*solution.Y[j]))
                                           for j in range(i+1, len(solution.Y))
                                           if solution.Y[i] > 1.e-12
                                           and solution.Y[j] > 1.e-12 ]
                                         for i in range(len(solution.Y)) ] ))) # typo in last sum of eqn 8 of PC2001?
        solution.Gnonconf = ( np.sum(solution.n*solution.params['g_std'](T)) +
                             np.sum( np.sum( [ [ solution.npr[i][j] *
                                                 ( solution.delta_g[i][j] / 2. )
                                                 for j in range(i+1, len(solution.Y))
                                                 if solution.Y[i] > 1.e-12
                                                 and solution.Y[j] > 1.e-12]
                                               for i in range(len(solution.Y)) ] ))) # typo in last sum of eqn 7 of PC2001?
        solution.delta_G = solution.Gnonconf - T*delta_Sc 
        
        
        return 1 # successful convergence


print 'WARNING still need to sort out ternary terms (eqn 30 in PC2001)'


plot = True
if plot == True:

    fig1 = mpimg.imread('figures/deltaH_KCl_MgCl2.png')
    plt.imshow(fig1, extent=[0., 1., -20000., 0.], aspect='auto')
    
    sol = KMgF_solution()
    T = 800. + 273.15
    all_xs = np.linspace(0.0, 1.0, 101)

    xs = []
    Gs = []
    Hs = []
    Ss = []
    for i, x in enumerate(all_xs):
        sol.molar_fractions = np.array([x, 1. - x, 0.])
        guess = [2.*x, 0., 1. - x, 0., 0., 0.]
        status1 = set_state(sol, T, guess)
        G = sol.delta_G
        
        status2 = set_state(sol, T + 1.)
        S = -(sol.delta_G - G)
        
        if status1 == 1 and status2 == 1:
            xs.append(1. - x)
            
            Gs.append(G)
            Ss.append(S)

            Hs.append(G + T*S)
            

        
    plt.plot(xs, np.array(Hs), label=str(T)+' K')

    plt.xlabel('Mole fraction MgF2')
    plt.legend(loc='lower right')
    plt.show()    
    
