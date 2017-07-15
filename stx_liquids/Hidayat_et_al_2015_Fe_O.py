import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os, sys
sys.path.insert(1,os.path.abspath('..'))

R = 8.31446

class FMS_solution():
    def g_FeO(self, T):
        return 0.

    def g_MgO(self, T):
        return 0.

    def g_SiO2(self, T):
        return 0.
        
    def __init__(self, molar_fractions=None):
        self.name = 'FeO-MgO-SiO2 liquid solution'
        self.params = {
            'asymmetric_systems': [(0, 1, 2)], # last component is the odd one out i.e. SiO2 is the asymmetric component in the system FeII, FeIII, O
            'g_std': lambda T: np.array( [self.g_FeO(T), self.g_MgO(T), self.g_SiO2(T)] ),
            'Z': np.array( [ [2., 2., 2.],
                             [2., 2., 2.],
                             [2., 2., 2.] ] ),
            'g': lambda T, chi: np.array( [ [ 0., 3347., (- 17697.
                                                          - 38528.*chi[0][2]
                                                          + 842570*np.power(chi[0][2], 5.)
                                                          - 1549201*np.power(chi[0][2], 6.)
                                                          + 962015.*np.power(chi[0][2], 7.) + 
                                                          (-16.736 + 62.76*np.power(chi[0][2], 7.))*T)],
                                            [ 0., 0., (-86090 - 48974*chi[0][2] + 328109*np.power(chi[1][2], 7.) +
                                                       (-37.656*chi[0][2] + 125.52*np.power(chi[1][2], 7.))*T)],
                                            [ 0., 0., 0. ] ] ),
            'ternary_q': lambda T, Y, chi: [ [0., 0., 0.],
                                             [0., 0., 0.],
                                             [0., 0., 0.] ]
        }


R = 8.31446

class dummy_solution(): # to fit figure 1 of Pelton et al., 2000
    def g_A(self, T):
        return 0.

    def g_B(self, T):
        return 0.

    def g_C(self, T):
        return 0.
        
    def __init__(self, molar_fractions=None):
        self.name = 'dummy liquid solution'
        self.params = {
            'asymmetric_systems': [(0, 1, 2)], # last component is the odd one out i.e. C is the asymmetric component in the system FeII, FeIII, O
            'g_std': lambda T: np.array( [self.g_A(T), self.g_B(T), self.g_C(T)] ),
            'Z': np.array( [ [2., 2., 2.],
                             [2., 2., 2.],
                             [2., 2., 2.] ] ),
            'g': lambda T, chi: np.array( [ [ 0., -84.e3, 0.],
                                            [ 0., 0., 0.],
                                            [ 0., 0., 0.] ] ),
            'ternary_q': lambda T, Y, chi: [ [0., 0., 0.],
                                             [0., 0., 0.],
                                             [0., 0., 0.] ]
        }

class KCl_MgCl2_solution(): # to fit figure 1 of Pelton et al., 2000
    def g_KCl(self, T):
        return 0.

    def g_MgCl2(self, T):
        return 0.

    def g_C(self, T):
        return 0.
        
    def __init__(self, molar_fractions=None):
        self.name = 'KCl MgCl2 liquid solution'
        self.params = {
            'asymmetric_systems': [(0, 1, 2)], # last component is the odd one out i.e. C is the asymmetric component in the system FeII, FeIII, O
            'g_std': lambda T: np.array( [self.g_KCl(T), self.g_MgCl2(T), self.g_C(T)] ),
            'Z': np.array( [ [6., 3., 2.],
                             [6., 6., 2.],
                             [2., 2., 2.] ] ),
            'g': lambda T, chi: np.array( [ [ 0., (-17497. -
                                                   1026.*chi[0][1] -
                                                   14801.*chi[1][0]), 0.],
                                            [ 0., 0., 0.],
                                            [ 0., 0., 0.] ] ),
            'ternary_q': lambda T, Y, chi: [ [0., 0., 0.],
                                             [0., 0., 0.],
                                             [0., 0., 0.] ]
        }

class Fe_O_solution():
    def g_FeII(self, T):
        if T > 298. and T < 1811.:
            return 13265.9 + 117.5756 * T - 23.5143 * T * np.log(T) - 0.00439752 * T * T - 5.892698e-8 * T * T * T + 77358.5 / T - 3.6751551e-21*np.power(T, 7)
        elif T < 6000.:
            return 10838.8 + 291.3020 * T - 46.0000 * T * np.log(T)
        else:
            raise Exception("Temperature outside range for FeII")

    def g_FeIII(self, T):
        return self.g_FeII(T) + 6276.0

    def g_O(self, T):
        if T > 298. and T < 2990.:
            return 121184.8 + 136.0406 * T - 24.50000 * T * np.log(T) - 9.8420e-4 * T * T - 0.12938e-6 * T * T * T + 322517. / T
        else:
            raise Exception("Temperature outside range for oxygen")
        
    def __init__(self, molar_fractions=None):
        self.name = 'Fe-O solution, FeII, FeIII, O'
        self.params = {
            'asymmetric_systems': [(0, 1, 2)], # last component is the odd one out i.e. O is the asymmetric component in the system FeII, FeIII, O
            'g_std': lambda T: np.array( [self.g_FeII(T), self.g_FeIII(T), self.g_O(T)] ),
            'Z': np.array( [ [6., 6., 2.],
                             [6., 6., 2.],
                             [2., 3., 6.] ] ),
            'g': lambda T, chi: np.array( [ [ 0., 83680., -391580.56 + (129778.63 - 30.3340*T) * chi[0][2] ],
                                          [ 0., 0., (-394551.2 + 12.5520*T) + (83680.00) * np.power(chi[1][2], 2.) ],
                                          [ 0., 0., 0. ] ] ),
            'ternary_q': lambda T, Y, chi: [ [ 0., (30543.20 - 44.0041*T) * Y[2], 0. ],
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

'''
T = 2000.
Fe_O = Fe_O_solution()
Fe_O.molar_fractions = np.array([0.0, 1.0, 0.0])
guesses = [0.0, 0.0, 0.0, 3.0, 0.0, 0.0]
print pair_proportions(guesses, Fe_O, T)
print 'Xpr', Fe_O.Xpr
print '' 
print 'chi', Fe_O.chi
exit()
'''

'''
T = 2000.
Fe_O = Fe_O_solution()
Fe_O.molar_fractions = np.array([1.0, 0.0, 1.0])
guesses = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0]
print pair_proportions(guesses, Fe_O, T)
print 'Z', Fe_O.Z
exit()
'''
plot_1 = False
if plot_1 == True:
    solution = dummy_solution()
    T = 1273.
    all_xs = np.linspace(0.0, 1.0, 101)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    
    for g in [0., -21e3, -42e3, -84e3]:
        solution.params['g'] =  lambda T, chi: np.array( [ [ 0., g, g],
                                                           [ 0., 0., g],
                                                           [ 0., 0., 0.] ] )
        
        xs = []
        Gs = []
        Hs = []
        Ss = []
        for i, x in enumerate(all_xs):
            solution.molar_fractions = np.array([1. - x, x, 0.])
            guess = [1. - x, 0., 0., x, 0., 0.]
            status = set_state(solution, T, guess)
            if status == 1:
                xs.append(x)
                G = solution.delta_G
                
                status = set_state(solution, T+1., guess)
                S = -(solution.delta_G - G)
                
                Gs.append(G)
                Ss.append(S)
                Hs.append(G + T*S)
                
            
        ax1.plot(xs, Hs, label=str(g/1.e3)+' kJ/mol')
        ax2.plot(xs, Ss, label=str(g/1.e3)+' kJ/mol')
    plt.legend(loc='lower right')
    plt.show()

plot = False
if plot == True:

    fig1 = mpimg.imread('figures/deltaH_KCl_MgCl2.png')
    plt.imshow(fig1, extent=[0., 1., -20000., 0.], aspect='auto')
    
    sol = KCl_MgCl2_solution()
    T = 800. + 273.15
    all_xs = np.linspace(0.0, 1.0, 101)

    xs = []
    Gs = []
    Hs = []
    Ss = []
    for i, x in enumerate(all_xs):
        sol.molar_fractions = np.array([x, 1. - x, 0.])
        guess = [x, 1. - x, 0., 0., 0., 0.]
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

    plt.ylim(-16000., 0.)
    plt.xlabel('Mole fraction MgCl2')
    plt.legend(loc='lower right')
    plt.show()    
    
plot_2 = False
if plot_2 == True:
    print 'WARNING: Probably not correct, as Blander and Pelton (1987) use'
    print 'a different solution model scheme'
    FMS = FMS_solution()
    all_xs = np.linspace(0.0, 1.0, 101)

    fig1 = mpimg.imread('figures/MgO_SiO2_gibbs_mixing_-10_0_kcal_Pelton_Blander_1986.png')
    plt.imshow(fig1, extent=[0, 1, -41840, 0], aspect='auto')

    for T in [1873.]:
        print T
        xs = []
        Gs = []
        Hs = []
        Ss = []
        for i, x in enumerate(all_xs):
            FMS.molar_fractions = np.array([0., 1. - x, x])
            guess = [0., 0., 0., 1. - x, 0., x]
            status = set_state(FMS, T, guess)
            
            if status == 1:
                xs.append(x)
                Gs.append(FMS.delta_G)

            
        plt.plot(xs, np.array(Gs), label=str(T)+' K', marker='.', linestyle='None')

    plt.xlabel('X(SiO2)')
    plt.legend(loc='lower right')
    plt.show()





Fe_O = Fe_O_solution()
T = 1873.
all_xs = np.linspace(0.0, 1.0, 101)


for y in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    print y
    xs = []
    Gs = []
    for i, x in enumerate(all_xs):
        Fe_O.molar_fractions = np.array([(1. - x)*(1. - y), x*(1. - y), y])

        status = set_state(Fe_O, T)
        if status == 1:
            xs.append(x)
            Gs.append(Fe_O.delta_G)
            
    plt.plot(xs, Gs, label=str(y))
plt.legend(loc='lower right')
plt.show()
    
