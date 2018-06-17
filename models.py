import numpy as np

def prep_class(params):
    
    params.n = len(params.components)
    params.ij = [(i, i) for i in range(params.n)]
    params.ij.extend([(i, j) for i in range(params.n) for j in range(i+1, params.n)])

    params.ternary_asymmetry_indices = [[[[], []] for i in range(params.n)]
                                        for j in range(params.n)]
    for i in range(params.n):
        for j in range(params.n):
            if i != j:
                params.ternary_asymmetry_indices[i][j][0].append(i)
                params.ternary_asymmetry_indices[i][j][1].extend([i, j])
                
                for a in params.ternary_asymmetries:
                    if i==a[0] and j==a[2]: # j is asymmetric
                        params.ternary_asymmetry_indices[i][j][0].append(a[1])
                        params.ternary_asymmetry_indices[i][j][1].append(a[1])
                    elif i==a[1] and j==a[2]: # j is asymmetric
                        params.ternary_asymmetry_indices[i][j][0].append(a[0])
                        params.ternary_asymmetry_indices[i][j][1].append(a[0])
                    elif j==a[0] and i==a[2]: # i is asymmetric
                        params.ternary_asymmetry_indices[i][j][1].append(a[1])
                    elif j==a[1] and i==a[2]: # i is asymmetric
                        params.ternary_asymmetry_indices[i][j][1].append(a[0])

class Fe_O_liquid(object):
    # Parameters are 
    def __init__(self):
        # Component names
        self.components = ['Fe_II', 'Fe_III', 'O']
        self.component_formulae = [{'Fe': 1.},
                                   {'Fe': 1.},
                                   {'O': 1.}]
        
        # Asymmetries in the ternary subsystems
        # indices start from 0. the asymmetric component goes last
        self.ternary_asymmetries = [[0, 1, 2]] 

        # Coordination of component m in n, Z_mn.
        # Note swapped values for FeIII-O
        # compared with Hidayat et al. and Shishin et al.,
        # consistent with logic laid out in Hidayat et al., 2015:
        # "in order to set the composition of maximum short-range
        # ordering at the molar ratio n_A / n_B = 2 , one can set the ratio
        # Z^B_BA / Z^A_AB = 2." Maximum ordering in FeIIIO is expected when
        # n_FeIII/n_O = 2/3, so Z^O_OFeIII / Z^FeIII_FeIIIO = 2/3
        self.ZZ = np.array([[6., 6., 2.],
                            [6., 6., 3.],
                            [2., 2., 6.]]) 


        self.g_Fe = lambda P, T: (13265.9 + 117.5756*T - 23.5143*T*np.log(T) -
                                  0.00439752*T*T - 5.89269e-8*T*T*T + 77358.5/T -
                                  3.6751551e-21*np.power(T, 7.)
                                  if T<1811. else -10838.8 + 291.3020*T - 46.*T*np.log(T))
        
        self.g_O = lambda P, T: (121184.8 + 136.0406*T - 24.5*T*np.log(T) -
                                 9.8420e-4*T*T - 0.12938e-6*T*T*T + 322517.2/T
                                 if T < 2990. else
                                 102133.3 + 231.9844*T - 36.2*T*np.log(T))

        self.g_component = lambda P, T: np.array([self.g_Fe(P, T),
                                                  self.g_Fe(P, T) + 6276.0,
                                                  self.g_O(P, T)])  # gibbs free energies of pure components

        self.deltaG_0_pairs = lambda P, T: np.array([[0., 83680., -391580.56], 
                                                     [0., 0., -394551.20 + 12.5520*T],
                                                     [0., 0., 0.]]) # Eq. 9; upper triangular, diagonal elements zeros
        
        
        # FeII, FeIII, O
        # first n columns give the pair (or triplet),
        # the next n give the raised indices,
        # and the last the value.
        self.g_binary = lambda P, T: [[0, 2, 1, 0, 129778.63 - 30.3340*T],
                                      [1, 2, 2, 0, 83680.]] # note swapped Fe3+ and O
        
        self.q_ternary = lambda P, T: [[0, 1, 2, 0, 0, 1, 30543.20 - 44.00414*T]] 

        self.g_ternary = lambda P, T: []

        prep_class(self)
        

# model parameters
class Fe_O_S_liquid(object):
    # Parameters are 
    def __init__(self):
        # Component names
        self.components = ['Fe_II', 'Fe_III', 'O', 'S']
        self.component_formulae = [{'Fe': 1.},
                                   {'Fe': 1.},
                                   {'O': 1.},
                                   {'S': 1.}]
        
        # Asymmetries in the ternary subsystems
        # indices start from 0. the asymmetric component goes last
        self.ternary_asymmetries = [[0, 1, 2],
                                    [0, 1, 3],
                                    [2, 3, 0],
                                    [2, 3, 1]] 

        # Coordination of component m in n, Z_mn.
        # Note swapped values for FeIII-O and FeIII-S
        # compared with Hidayat et al. and Shishin et al.,
        # consistent with logic laid out in Hidayat et al., 2015:
        # "in order to set the composition of maximum short-range
        # ordering at the molar ratio n_A / n_B = 2 , one can set the ratio
        # Z^B_BA / Z^A_AB = 2." Maximum ordering in FeIIIO is expected when
        # n_FeIII/n_O = 2/3, so Z^O_OFeIII / Z^FeIII_FeIIIO = 2/3
        self.ZZ = np.array([[6., 6., 2., 2.],
                            [6., 6., 3., 3.],
                            [2., 2., 6., 6.],
                            [2., 2., 6., 6.]]) 


        self.g_Fe = lambda P, T: (13265.9 + 117.5756*T - 23.5143*T*np.log(T) -
                                  0.00439752*T*T - 5.89269e-8*T*T*T + 77358.5/T -
                                  3.6751551e-21*np.power(T, 7.)
                                  if T<1811. else -10838.8 + 291.3020*T - 46.*T*np.log(T))
        
        self.g_O = lambda P, T: (121184.8 + 136.0406*T - 24.5*T*np.log(T) -
                                 9.8420e-4*T*T - 0.12938e-6*T*T*T + 322517./T
                                 if T < 2990. else
                                 102133.3 + 231.9844*T - 36.2*T*np.log(T))
        
        self.g_S = lambda P, T: (-4001.5 + 77.9057*T - 15.504*T*np.log(T) -
                                 0.0186290*T*T - 0.24942e-6*T*T*T - 113945/T
                                 if T < 388.36 else
                                 -5285183.2 + 118449.6004*T - 19762.4000*T*np.log(T) + 32.79275100*T*T -
                                 10221.417e-6*T*T*T + 264673500./T
                                 if T < 428.15 else
                                 - 8174994.8 + 319914.0872*T - 57607.2990*T*np.log(T) + 135.3045000*T*T - 52997.333e-6*T*T*T
                                 if T < 432.25 else
                                 - 219408.8 + 7758.8558*T - 1371.8500*T*np.log(T) + 2.8450351*T*T - 1013.8033e-6*T*T*T
                                 if T < 453.15 else
                                 92539.8 - 1336.3502*T + 202.9580*T*np.log(T) - 0.2531915*T*T + 51.8835e-6*T*T*T - 8202200/T
                                 if T < 717. else
                                 -6890. + 176.3709*T - 32.*T*np.log(T))

        self.g_component = lambda P, T: np.array([self.g_Fe(P, T),
                                                  self.g_Fe(P, T) + 6276.0,
                                                  self.g_O(P, T),
                                                  self.g_S(P, T)])  # gibbs free energies of pure components



        # WARNING: # FeIIIS and OS pairs unknown, set as arbitrary small number (1.e-12) here
        self.deltaG_0_pairs = lambda P, T: np.array([[0., 83680., -391580.56,
                                                      -122334.14 + 112.4659*T - 13.7582*T*np.log(T)], # gFeIIS was -104888.10 + 0.3388*T in ref [2]
                                                     [0., 0., -394551.20 + 12.5520*T, 1.e-12],
                                                     [0., 0., 0., 1.e-12],
                                                     [0., 0., 0., 0.]]) # Eq. 9; upper triangular, diagonal elements zeros
        
        # FeII, FeIII, O, S
        # first n columns give the pair (or triplet),
        # the next n give the raised indices,
        # and the last the value.
        self.g_binary = lambda P, T: [[0, 2, 1, 0, 129778.63 - 30.3340*T],
                                      [0, 3, 1, 0, 35043.32 - 9.880*T], # note opposite index order in the two papers... (listed as g_FeIIS^{01, 02, 03, 10, 20, 40} in Shishin, but g_FeIIS^{10, 20, 30, 01, 02, 04} in Waldner and Pelton, 2005)
                                      [0, 3, 2, 0, 23972.27], # same here...
                                      [0, 3, 3, 0, 30436.82], # and so on...
                                      [0, 3, 0, 1, 8626.26], #
                                      [0, 3, 0, 2, 72954.29 - 26.178*T], #
                                      [0, 3, 0, 4, 25106.], # to here.
                                      [1, 2, 2, 0, 83680.]]
        

        self.q_ternary = lambda P, T: [[0, 1, 2, 0, 0, 1, 30543.20 - 44.00414*T],
                                       [2, 3, 0, 0, 1, 2, 292567.68 - 204.2804*T]] # note i, j in paper is SO (i=3, j=2), not OS... so 0<->1 and 3<->4 switched here because i<j in Eq. 7..
        
        self.g_ternary = lambda P, T: [[0, 3, 2, 0, 0, 1, 22044.34 - 17.80426*T],
                                       [0, 3, 2, 0, 0, 2, 27764.62 - 22.25532*T],
                                       [0, 2, 3, 1, 0, 1, -23012.00],
                                       [0, 2, 3, 9, 0, 1, -41553.40 + 41.84*T],
                                       [1, 2, 3, 0, 0, 2, -21756.8]]

        prep_class(self)



class Fe_O_liquid(object):
    # Parameters are 
    def __init__(self):
        # Component names
        self.components = ['Fe_II', 'Fe_III', 'O']
        self.component_formulae = [{'Fe': 1.},
                                   {'Fe': 1.},
                                   {'O': 1.}]
        
        # Asymmetries in the ternary subsystems
        # indices start from 0. the asymmetric component goes last
        self.ternary_asymmetries = [[0, 1, 2]] 

        # Coordination of component m in n, Z_mn.
        # Note swapped values for FeIII-O
        # compared with Hidayat et al. and Shishin et al.,
        # consistent with logic laid out in Hidayat et al., 2015:
        # "in order to set the composition of maximum short-range
        # ordering at the molar ratio n_A / n_B = 2 , one can set the ratio
        # Z^B_BA / Z^A_AB = 2." Maximum ordering in FeIIIO is expected when
        # n_FeIII/n_O = 2/3, so Z^O_OFeIII / Z^FeIII_FeIIIO = 2/3
        self.ZZ = np.array([[6., 6., 2.],
                            [6., 6., 3.],
                            [2., 2., 6.]]) 


        self.g_Fe = lambda P, T: (13265.9 + 117.5756*T - 23.5143*T*np.log(T) -
                                  0.00439752*T*T - 5.89269e-8*T*T*T + 77358.5/T -
                                  3.6751551e-21*np.power(T, 7.)
                                  if T<1811. else -10838.8 + 291.3020*T - 46.*T*np.log(T))
        
        self.g_O = lambda P, T: (121184.8 + 136.0406*T - 24.5*T*np.log(T) -
                                 9.8420e-4*T*T - 0.12938e-6*T*T*T + 322517.2/T
                                 if T < 2990. else
                                 102133.3 + 231.9844*T - 36.2*T*np.log(T))

        self.g_component = lambda P, T: np.array([self.g_Fe(P, T),
                                                  self.g_Fe(P, T) + 6276.0,
                                                  self.g_O(P, T)])  # gibbs free energies of pure components

        self.deltaG_0_pairs = lambda P, T: np.array([[0., 83680., -391580.56], 
                                                     [0., 0., -394551.20 + 12.5520*T],
                                                     [0., 0., 0.]]) # Eq. 9; upper triangular, diagonal elements zeros
        
        
        # FeII, FeIII, O
        # first n columns give the pair (or triplet),
        # the next n give the raised indices,
        # and the last the value.
        self.g_binary = lambda P, T: [[0, 2, 1, 0, 129778.63 - 30.3340*T],
                                      [1, 2, 2, 0, 83680.]]
        
        self.q_ternary = lambda P, T: [[0, 1, 2, 0, 0, 1, 30543.20 - 44.00414*T]] 

        self.g_ternary = lambda P, T: []

        prep_class(self)
        

# model parameters
class KCl_MgCl2_liquid(object):
    # Parameters are 
    def __init__(self):
        # Component names
        self.components = ['KCl', 'MgCl2']
        self.component_formulae = [{'K': 1., 'Cl': 1.},
                                   {'Mg': 1., 'Cl': 2.}]
        
        # Asymmetries in the ternary subsystems
        # indices start from 0. the asymmetric component goes last
        self.ternary_asymmetries = [] 

        # Coordination of component m in n, Z_mn.
        # Note swapped values for FeIII-O and FeIII-S
        # compared with Hidayat et al. and Shishin et al.,
        # consistent with logic laid out in Hidayat et al., 2015:
        # "in order to set the composition of maximum short-range
        # ordering at the molar ratio n_A / n_B = 2 , one can set the ratio
        # Z^B_BA / Z^A_AB = 2." Maximum ordering in FeIIIO is expected when
        # n_FeIII/n_O = 2/3, so Z^O_OFeIII / Z^FeIII_FeIIIO = 2/3
        self.ZZ = np.array([[6., 3.],
                            [6., 6.]]) 


        self.g_KCl = lambda P, T: 0.
        
        self.g_MgCl2 = lambda P, T: 0.

        # gibbs free energies of pure components
        self.g_component = lambda P, T: np.array([self.g_KCl(P, T),
                                                  self.g_MgCl2(P, T)])  


        # Eq. 9; upper triangular, diagonal elements zeros
        self.deltaG_0_pairs = lambda P, T: np.array([[0., -17497.], 
                                                     [0., 0.]]) 
        
        # KCl, MgCl2
        # first n columns give the pair (or triplet),
        # the next n give the raised indices,
        # and the last the value.
        self.g_binary = lambda P, T: [[0, 1, 1, 0, -1026.],
                                      [0, 1, 0, 1, -14801.]]

        self.q_ternary = lambda P, T: [] 
        self.g_ternary = lambda P, T: []

        prep_class(self)


# model parameters
class test_binary_liquid(object):
    # Parameters are 
    def __init__(self):
        # Component names
        self.components = ['A', 'B']
        self.component_formulae = [{'A': 1.},
                                   {'B': 1.}]
        
        # Asymmetries in the ternary subsystems
        # indices start from 0. the asymmetric component goes last
        self.ternary_asymmetries = [] 

        # Coordination of component m in n, Z_mn.
        # Note swapped values for FeIII-O and FeIII-S
        # compared with Hidayat et al. and Shishin et al.,
        # consistent with logic laid out in Hidayat et al., 2015:
        # "in order to set the composition of maximum short-range
        # ordering at the molar ratio n_A / n_B = 2 , one can set the ratio
        # Z^B_BA / Z^A_AB = 2." Maximum ordering in FeIIIO is expected when
        # n_FeIII/n_O = 2/3, so Z^O_OFeIII / Z^FeIII_FeIIIO = 2/3
        self.ZZ = np.array([[2., 2.],
                            [2., 2.]]) 


        self.g_A = lambda P, T: 0.
        
        self.g_B = lambda P, T: 0.

        # gibbs free energies of pure components
        self.g_component = lambda P, T: np.array([self.g_A(P, T),
                                                  self.g_B(P, T)])  


        # Eq. 9; upper triangular, diagonal elements zeros
        self.deltaG_0_pairs = lambda P, T: np.array([[0., -42000.], 
                                                     [0., 0.]]) 
        
        # A, B
        # first n columns give the pair (or triplet),
        # the next n give the raised indices,
        # and the last the value.
        self.g_binary = lambda P, T: []
        self.q_ternary = lambda P, T: [] 
        self.g_ternary = lambda P, T: []

        prep_class(self)
