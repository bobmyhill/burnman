import numpy as np

# Asymmetries in the ternary subsystems
# indices start from 0. the asymmetric component goes last
ternary_asymmetries = [[1, 2, 0],
                       [0, 1, 4],
                       [0, 3, 2],
                       [2, 3, 1],
                       [1, 4, 2],
                       [1, 3, 4]] 

n = 5

chi = np.zeros((n, n))
ksi = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            num_indices = [i+1]
            denom_indices = [i+1, j+1]
            for a in ternary_asymmetries:
                if i==a[0] and j==a[2]:
                    num_indices.append(a[1]+1)
                    denom_indices.append(a[1]+1)
                elif i==a[1] and j==a[2]:
                    num_indices.append(a[0]+1)
                    denom_indices.append(a[0]+1)
                elif j==a[0] and i==a[2]:
                    denom_indices.append(a[1]+1)
                elif j==a[1] and i==a[2]:
                    denom_indices.append(a[0]+1)

            print(i+1, j+1, num_indices, denom_indices)

            # NOTE: NUMBERS INCREMENTED BY 1 FOR PURPOSES OF COMPARISON WITH PAPER ONLY!!!
            
            '''
            with warnings.catch_warnings():
                warnings.simplefilter('error')

                num = np.sum(p_pairs[np.ix_(num_indices,num_indices)])
                denom = np.sum(p_pairs[np.ix_(denom_indices,denom_indices)])
                try:
                    chi[i][j] = num / denom # Eq. 27.
                except:
                    if np.abs(num) < 1.e-12 and np.abs(denom) < 1.e-12:
                        chi[i][j] = 0.
                    
            # Only single-counting when i!=j for the example in eq 29.
            # p_pairs must be strictly upper triangular for this implementation to work
            # as shown in the example (eq. 29),
            # where p_pairs[i,j] is counted only once if i!=j.
            
            ksi[i][j] = np.sum(p_coord[num_indices]) # eq. 22
            '''
