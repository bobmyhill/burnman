import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import minimize

from quasichemical_functions import *
from models import *


test_binary = False
test_asymmetric_binary = False
test_ternary = True
test_quaternary = False

# Test binary
if test_binary:
    pressure = 1.e5
    temperature = 1273.15
    liq = test_binary_liquid()
    
    xs = np.linspace(1.e-12, 1. - 1.e-12, 51)
    Gs = np.empty_like(xs)
    Ss = np.empty_like(xs)
    
    deltaT = 1.
    
    fig = plt.figure()
    ax = [fig.add_subplot(1, 2, i+1) for i in range(2)]
    
    figH = mpimg.imread('binary_42kJ.png')
    figS = mpimg.imread('binary_42kJ_entropy.png')
    
    ax[0].imshow(figH, extent=[0., 1., -41000., 0.], aspect='auto')
    ax[1].imshow(figS, extent=[0., 1., 0., 7.25], aspect='auto')
    
    for dG in [0., -21000., -42000., -84000.]:
        liq.deltaG_0_pairs = lambda P, T: np.array([[0., dG], 
                                                    [0., 0.]]) 
        for i, x in enumerate(xs):
            bulk_composition = {'A': 1.-x,
                                'B': x}
            Gs1 = equilibrate_liquid(liq, pressure, temperature-0.5*deltaT, bulk_composition)
            Gs2 = equilibrate_liquid(liq, pressure, temperature+0.5*deltaT, bulk_composition)
            Gs[i] = (Gs1 + Gs2)/2.
            Ss[i] = -(Gs2 - Gs1)/deltaT
            Hs = Gs + temperature*Ss

        ax[0].plot(xs, Hs, label='{0} kJ/mol'.format(dG/1000.))
        ax[1].plot(xs, Ss, label='{0} kJ/mol'.format(dG/1000.))


    ax[1].plot(xs, -8.31446*(xs*np.log(xs) + (1. - xs)*np.log(1. - xs)), linestyle='--', linewidth=2., label='ideal')

    for i in range(2):
        ax[i].legend(loc='best')
    plt.show()


# Test asymmetric binary
if test_asymmetric_binary:
    pressure = 1.e5
    temperature = 1073.15
    liq = KCl_MgCl2_liquid()
    
    figH = mpimg.imread('KCl_MgCl2_enthalpy.png')

    fig = plt.figure()
    ax = [fig.add_subplot(1, 2, i+1) for i in range(2)]
    ax[0].imshow(figH, extent=[-0.003, 0.995, -20000., 0.], aspect='auto')
    
    deltaT = 1.
    xs = np.linspace(0., 1., 51)
    Gs = np.empty_like(xs)
    Ss = np.empty_like(xs)

    for temperature in [673.15, 1073.15, 1473.15]:
        for i, x in enumerate(xs):
            bulk_composition = {'K': 1.-x,
                                'Mg': x,
                                'Cl': 1. + x}
            Gs1 = equilibrate_liquid(liq, pressure, temperature-0.5*deltaT, bulk_composition)
            Gs2 = equilibrate_liquid(liq, pressure, temperature+0.5*deltaT, bulk_composition)
            Gs[i] = (Gs1 + Gs2)/2.
            Ss[i] = -(Gs2 - Gs1)/deltaT
            
        Hs = Gs + temperature*Ss
        ax[0].plot(xs, Hs, label='{0} K'.format(temperature))
        ax[1].plot(xs, Ss, label='{0} K'.format(temperature))


    
    ax[0].set_xlim(0., 1.)
    ax[1].set_xlim(0., 1.)
    ax[1].plot(xs,
               -8.31446*(xs*np.log(xs) +
                         (1. - xs)*np.log(1. - xs)),
               linestyle='--', linewidth=2., label='ideal')
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    plt.show()
    

# FeII-FeIII-O system
if test_ternary:
    
    '''
  0.6 Fe  +  0.4 O  =
 
    0          mol   gas_ideal
            (1200 C, 1 atm,     a=1.2745E-06)
            ( 1.2739E-06      Fe
            + 5.7752E-10      FeO
            + 5.1388E-13      O
            + 3.4181E-14      O2
            + 1.4404E-29      O3)
 
  + 1.0000     mol   Liq(Matte/Metal)
    (39.907 gram, 1.0000 mol)
            (1200 C, 1 atm,     a=1.0000)
            ( 0.57280         Fe
            + 0.40000         O
            + 2.7198E-02      Fe3+)
 
            Mole fraction of quadruplets:
            Fe-Fe-Va-Va                        0.43759
            Fe3+-Fe3+-Va-Va                    3.3605E-03
            O-O-Va-Va                          3.0850E-06

            Fe-Fe3+-Va-Va                      2.5191E-03
            Fe-O-Va-Va                         0.50439

            O-Fe3+-Va-Va                       5.2141E-02
            Total amount/mol                   1.4375
 
            System component         Amount/mol    Amount/gram   Mole fraction  Mass fraction
            Fe                       0.60000        33.507         0.60000        0.83963
            O                        0.40000        6.3998         0.40000        0.16037
 
  Cut-off limit for gaseous fractions/phase activities = 1.00E-75
 
  ********************************************************************
        H             G             V             S            Cp
       (J)           (J)         (litre)        (J/K)         (J/K)
  ********************************************************************
  -3.54516E+04  -1.71807E+05   0.00000E+00   9.25603E+01   3.54530E+01
    '''

    pressure = 1.e5
    temperature = 1473.15
    liq = Fe_O_liquid()

    bulk_composition = {'Fe': 0.6,
                        'O': 0.4}
    
    # Order = FeII, FeIII, O
    factsage_n_component = np.array([0.57280, 2.7198E-02, 0.40000])
    
    factsage_nij =  1.4375*np.array([0.43759, 3.3605E-03, 3.0850E-06, 
                                     2.5191E-03, 0.50439,
                                     5.2141E-02])

    factsage_gibbs = -1.71807E+05

    # First, let's test the pair proportions given by FactSage
    G = gibbs(factsage_nij, pressure, temperature, liq)
    print(G)
    print('FactSage: {0}'.format(factsage_n_component))
    print('This model: {0}'.format(liq.n_component))
    print('diff: {0}'.format(liq.n_component - factsage_n_component))
    print('diff Gibbs: {0} J/mol'.format(G - factsage_gibbs))
    print('')

    # Next, let's try minimizing the free energy ourselves
    G = equilibrate_liquid(liq, pressure, temperature, bulk_composition, guess=factsage_nij)

    print('FactSage: {0}'.format(factsage_n_component))
    print('This model: {0}'.format(liq.n_component))
    print('diff: {0}'.format(liq.n_component - factsage_n_component))
    print('diff Gibbs: {0} J/mol'.format(G - factsage_gibbs))
    print('')

    '''
    0.43 Fe  +  0.57 O  =
 
    0          mol   gas_ideal
            (1200 C, 1 atm,     a=6.2499E-06)
            ( 6.2429E-06      O2
            + 6.9448E-09      O
            + 8.4898E-11      FeO
            + 1.3857E-11      Fe
            + 3.5553E-17      O3)
 
  + 1.0000     mol   Liq(Matte/Metal)
    (33.133 gram, 1.0000 mol)
            (1200 C, 1 atm,     a=1.0000)
            ( 0.14491         Fe
            + 0.57000         O
            + 0.28509         Fe3+)
 
            Mole fraction of quadruplets:
            Fe-Fe-Va-Va                        9.1326E-04
            O-O-Va-Va                          8.7636E-05
            Fe3+-Fe3+-Va-Va                    3.7874E-03
            Fe-O-Va-Va                         0.25235
            Fe-Fe3+-Va-Va                      1.2217E-04
            O-Fe3+-Va-Va                       0.74274
            Total amount/mol                   1.1456
 
            System component         Amount/mol    Amount/gram   Mole fraction  Mass fraction
            Fe                       0.43000        24.013         0.43000        0.72476
            O                        0.57000        9.1197         0.57000        0.27524
 
  Cut-off limit for gaseous fractions/phase activities = 1.00E-75
 
  ********************************************************************
        H             G             V             S            Cp
       (J)           (J)         (litre)        (J/K)         (J/K)
  ********************************************************************
  -1.05095E+05  -2.18804E+05   0.00000E+00   7.71879E+01   3.32729E+01
    '''


    bulk_composition = {'Fe': 0.43,
                        'O': 0.57}
    
    # Order = FeII, FeIII, O
    factsage_n_component = np.array([0.14491, 0.28509, 0.57000])
    
    factsage_nij =  1.1456*np.array([9.1326E-04, 3.7874E-03, 8.7636E-05, 
                                     1.2217E-04, 0.25235,
                                     0.74274])

    factsage_gibbs = -2.18804E+05

    # First, let's test the pair proportions given by FactSage
    G = gibbs(factsage_nij, pressure, temperature, liq)
    
    print('FactSage: {0}'.format(factsage_n_component))
    print('This model: {0}'.format(liq.n_component))
    print('diff: {0}'.format(liq.n_component - factsage_n_component))
    print('diff Gibbs: {0} J/mol'.format(G - factsage_gibbs))
    print('')

    # Next, let's try minimizing the free energy ourselves
    G = equilibrate_liquid(liq, pressure, temperature, bulk_composition, guess=factsage_nij)

    print('FactSage: {0}'.format(factsage_n_component))
    print('This model: {0}'.format(liq.n_component))
    print('diff: {0}'.format(liq.n_component - factsage_n_component))
    print('diff Gibbs: {0} J/mol'.format(G - factsage_gibbs))
    print('')    


if test_quaternary:
    '''
    0.90 Fe +  0.05 O +  0.05 S =

               (1200.00 C, 1 atm,     a=1.0000)
           ( 5.0000E-02      S                                         UQPY
           + 0.90000         Fe                                        UQPY
           + 5.0000E-02      O                                         UQPY
           + 1.5621E-07      Fe3+                                      UQPY)

    #           Mole fraction of quadruplets:
    #           Fe-Fe-Va-Va                        0.92317
    #           Fe3+-Fe3+-Va-Va                    1.9692E-12
    #           O-O-Va-Va                          6.0832E-08
    #           S-S-Va-Va                          5.1020E-05

    #           Fe-Fe3+-Va-Va                      8.8572E-08
    #           Fe-O-Va-Va                         3.8403E-02
    #           Fe-S-Va-Va                         3.8369E-02

    #           Fe3+-O-Va-Va                       1.1314E-07
    #           Fe3+-S-Va-Va                       2.2542E-08

    #           O-S-Va-Va                          3.5234E-06

    #           Total amount/mol                   2.6038

    #*****************************************************************
    #       H            G            V            S           Cp
    #      (J)          (J)        (litre)       (J/K)        (J/K)
    # *****************************************************************
    #  4.61261E+04 -9.50916E+04  0.00000E+00  9.58610E+01  3.85991E+01
    ''' 

    # order of pairs [FeII, FeIII, O, S]
    # [(0, 0), (1, 1), (2, 2), (3, 3),
    #  (0, 1), (0, 2), (0, 3),
    #  (1, 2), (1, 3),
    #  (2, 3)]

    pressure = 1.e5
    temperature = 1473.15
    liq = Fe_O_S_liquid()

    bulk_composition = {'Fe': 0.9,
                        'O': 0.05,
                        'S': 0.05}
    
    # Order = FeII, FeIII, O, S
    factsage_n_component = np.array([0.9, 1.5621E-07, 0.05, 0.05])
    
    factsage_nij =  2.6038*np.array([0.92317, 1.9692E-12, 6.0832E-08, 5.1020E-05,
                                     8.8572E-08, 3.8403E-02, 3.8369E-02,
                                     1.1314E-07, 2.2542E-08,
                                     3.5234E-06])

    factsage_gibbs = -9.50916E+04

    # First, let's test the pair proportions given by FactSage
    gibbs = gibbs(factsage_nij, pressure, temperature, liq)
    
    print('FactSage: {0}'.format(factsage_n_component))
    print('This model: {0}'.format(liq.n_component))
    print('diff: {0}'.format(factsage_n_component - liq.n_component))
    print('diff Gibbs: {0} J/mol'.format(factsage_gibbs - gibbs))
    print('')

    bulk_composition = {'Fe': liq.n_component[0] + liq.n_component[1],
                        'O': liq.n_component[2],
                        'S': liq.n_component[3]}
    # Next, let's try minimizing the free energy ourselves, with the (incorrect) bulk composition
    gibbs = equilibrate_liquid(liq, pressure, temperature, bulk_composition, guess=factsage_nij)

    print('FactSage: {0}'.format(factsage_n_component))
    print('This model: {0}'.format(liq.n_component))
    print('diff: {0}'.format(factsage_n_component - liq.n_component))
    print('diff Gibbs: {0} J/mol'.format(factsage_gibbs - gibbs))
    print('')



"""
# user input
pressure = 1.e9
temperature = 1000.
liq = Fe_O_liquid() # Fe_O_S_liquid()

xs = np.linspace(0.0, 1.0, 101)
Gs = np.empty_like(xs)
Gs2 = np.empty_like(xs)
x_FeII = np.empty_like(xs)
x_FeIII = np.empty_like(xs)
x_O = np.empty_like(xs)
Sconf = np.empty_like(xs)


fig = plt.figure()
ax = [fig.add_subplot(1, 1, i+1) for i in range(1)]
fig1 = mpimg.imread('Hidayat_2015_Fe_O_Fig3.png')
fig2 = mpimg.imread('FeIII_FeII_ratio.png')

#ax[0].imshow(fig1, extent=[0., 1., -250000., 0.], aspect='auto')
ax[0].imshow(fig2, extent=[0., 1., 0., 1.], aspect='auto')


#for temperature in np.linspace(1000., 3000., 5):
for temperature in [1873.]:
    guess = []
    print(temperature)
    for i, x in enumerate(xs):
        print(i)
        bulk_composition = {'Fe': (1. - x),
                            'O': x}
        Gs[i] = equilibrate_liquid(liq, pressure, temperature, bulk_composition) #, guess)
        x_FeII[i] = liq.n_component[0]
        x_FeIII[i] = liq.n_component[1]
        x_O[i] = liq.n_component[2]
        Sconf[i] = liq.configurational_entropy
        guess = liq.nij
        
    #ax[0].plot(xs, Gs - xs*Gs[-1] - (1. - xs)*Gs[0], label='{0} K'.format(temperature))
    #ax[1].plot(xs, Sconf, label='{0} K'.format(temperature))
    ax[0].plot(xs, x_FeIII/(x_FeII + x_FeIII), label='{0} K'.format(temperature))
    #ax[3].plot(xs, x_O, label='{0} K'.format(temperature))
    
    n_Os = np.linspace(0.001, 0.50, 101)
    Gs2 = np.empty_like(n_Os)
    Gs3 = np.empty_like(n_Os)
    for i, n_O in enumerate(n_Os):
        #Gs2[i] = minimize(binary_excess_gibbs, [1.e-12],
        #                 args=(n_O, temperature), bounds=((0., 2.*n_O),)).fun
        Gs3[i] = simple_excess_enthalpy(n_O, temperature)
    #ax[0].plot(n_Os, Gs2, label='2: {0} K'.format(temperature))
    #ax[0].plot(n_Os, Gs3, label='3: {0} K'.format(temperature))

    
plt.legend(loc='best')
plt.show()


exit()
"""
'''
xs = np.linspace(0.2, 0.5, 31)
Gs = np.empty_like(xs)
x_FeII = np.empty_like(xs)
x_FeIII = np.empty_like(xs)
x_O = np.empty_like(xs)

temperature = 1873.


proportions = np.array([1., 1.e-12, 1.e-12, 1.e-12])
GFe = gibbs_components_constrained(pressure, temperature, proportions, liq)

proportions = np.array([0.5, 1.e-12, 0.5, 1.e-12])
GFeO = gibbs_components_constrained(pressure, temperature, proportions, liq)

#for temperature in np.linspace(1000., 3000., 5):
for f in [1.e-12, 0.025, 0.05, 0.075, 0.1]:
    print(f)
    for i, x in enumerate(xs):
        print(i)
        proportions = np.array([(1. - x)*(1. - f), (1. - x)*f, x, 1.e-12])
        Gs[i] = gibbs_components_constrained(pressure, temperature, proportions, liq)
        x_FeII[i] = liq.n_component[0]
        x_FeIII[i] = liq.n_component[1]
        x_O[i] = liq.n_component[2]
        

    plt.plot(xs, Gs - xs*2.*GFeO - (1. - xs*2.)*GFe, label='{0}'.format(f))
    #plt.plot(xs, x_FeIII/(x_FeII + x_FeIII), label='{0}'.format(f))
plt.legend(loc='best')
plt.show()


# number of moles of m-n pairs, n_mn, upper triangular, diagonals nonzero
#n_pairs = np.array([[1., 1., 1., 1.],
#                    [0., 1., 1., 1.],
#                    [0., 0., 1., 1.],
#                    [0., 0., 0., 1.]])

'''
'''
n = 4

Qij = np.zeros((n*n - n)/2) # order parameters given in the order (01, 02, ..., 12, 13,...)

A = np.identity((n*n + n)/2)

ij = [(i, j) for i in range(n) for j in range(i+1, n)]

for k, (i, j) in enumerate(ij):
    A[n+k,i] -= Qij[k]
    A[n+k,j] -= Qij[k]

    A[i,n+k] += 1 
    if i==j:
        A[i,n+k] += 1
    

b = np.zeros((n*n + n)/2)
b[:n] = proportions

pairs = np.linalg.solve(A, b)
n_pairs = np.zeros((n, n))
for i in range(n):
    n_pairs[i][i] = pairs[i]*liq.ZZ[i][i]

for k, (i, j) in enumerate(ij):
    n_pairs[i][j] = pairs[n+k]*liq.ZZ[i][j]

print(A)
print(gibbs(n_pairs, pressure, temperature, liq))
'''
