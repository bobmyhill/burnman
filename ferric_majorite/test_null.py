import numpy as np
import scipy.optimize as opt
from scipy.linalg import pinv
from scipy.optimize import minimize
from sympy import Matrix, nsimplify
from sympy.matrices import GramSchmidt

from scipy.linalg import sqrtm

import os
import sys

def simplify_matrix(arr):
    def f(i,j):
        return nsimplify(arr[i][j])
    return Matrix( len(arr), len(arr[0]), f )


# Mg, Fe, Ca, Al, Si (O not measured directly)
S = np.array([[3., 0., 0., 2., 3.], # py
              [0., 3., 0., 2., 3.], # alm
              [0., 0., 3., 2., 3.], # gr
              [4., 0., 0., 0., 4.]]) # sk

#     MgX, FeX, CaX, AlY, FeY, MgY, SiY 
E = np.array([[1., 0., 0., 1., 0., 0., 0.],
              [0., 1., 0., 1., 0., 0., 0.],
              [0., 0., 1., 1., 0., 0., 0.],
              [1., 0., 0., 0., 0., 0.5, 0.5]])
             

b = np.array([2.5, 0.29, 0.65, 0.88, 3.6])
sigma_b = np.array([0.04, 0.02, 0.09, 0.07, 0.1])



# Here's a bit of prep work to convert from endmember space to site space
A = S.T.dot(pinv(E.T))
null = np.array(Matrix(E).nullspace()).astype(float)

# First, let's make a guess using nnls
Aprime = np.vstack((A, null))
bprime = np.concatenate((b, np.zeros(len(null))))
xprime_guess, residual = opt.nnls(Aprime, bprime)


# Now, let's do a constrained minimization
# Define minimisation function
fn = lambda x, A, b: np.linalg.norm(A.dot(x) - b)

# Define constraints and bounds
# We need some python magic to do this dynamically
endmember_constraints = lambda nullspace: [{'type': 'eq', 'fun': lambda x, eq=eq: eq.dot(x)}
                                           for eq in nullspace]

cons = endmember_constraints(null)
bounds = [[0., None]]*len(E.T)

#Call minimisation subject to these values
sol = minimize(fn, xprime_guess, args=(A, b), method='SLSQP',bounds=bounds,constraints=cons)

# We're done! Convert back to endmember space
endmember_amounts = pinv(E.T).dot(sol.x)
res = A.dot(sol.x) - b

print(endmember_amounts)
n_mbrs = len(S)

covar_comp = np.diag([s*s for s in sigma_b])
covar_endmembers = pinv(S).T.dot(covar_comp).dot(pinv(S))

# This is *NOT* endmember uncertainties normalised to one!!
# To do this, we need to remove the first endmember

# i.e. p_py = x_py/sum(x_i)
# dp_i/dx_i  = (1 - p_i)/sum(x)
# dp_j/dx_i = - p_j/sum(x)

dpdx = np.eye(n_mbrs) - np.array([endmember_amounts for i in range(n_mbrs)])
covar_endmember_proportions = dpdx.dot(covar_endmembers).dot(dpdx.T)
print(covar_endmember_proportions)


# We want to ask what the uncertainty on the endmember chemical potentials are,
# given the uncertainty on the endmember proportions.

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))
import burnman

gt = burnman.minerals.JH_2015.garnet()
print(gt.endmember_names)


dc = 0.00001
c0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
gt.set_composition(c0)
gt.set_state(1.e5, 1000.)
print(gt.partial_gibbs[0])



c1 = c0 - (np.array([1., 0., 0., 0., 0.]) - c0)*dc/2.
gt.set_composition(c1)
G1 = gt.gibbs*(1. - dc/2.)

p1 = gt.partial_gibbs

c2 = c0 + (np.array([1., 0., 0., 0., 0.]) - c0)*dc/2.
gt.set_composition(c2)

p2 = gt.partial_gibbs

G2 = gt.gibbs*(1. + dc/2.)
print((G2 - G1)/dc)

k=1
l=1
for i in range(4):
    for j in range(i+1, 4):
        if j==k:
            if i==l:
                print('(1 - p{0})W{0}{1}'.format(i, j))
            else:
                print('- p{0}W{0}{1}'.format(i, j))
        if i==k:
            if j==l:
                print('(1 - p{1})W{0}{1}'.format(i, j))
            else:
                print('- p{1}W{0}{1}'.format(i, j))



print(p2 - p1)

#print(

#[[4.e3, 35.e3, 91.e3, 2.e3],
# [4.e3, 60.e3, 6.e3],
# [2.e3, 47.e3],
# [101.e3]]
exit()
                
            


"""
# no unique way to obtain proportions or uncertainties.
# can attempt to maximize resolution:
# expected variability between endmembers?
S = np.array([[3., 0., 0., 2.], # py
              [0., 3., 0., 2.], # alm
              [0., 0., 3., 2.], # gr
              [4., 0., 0., 0.]]) # sk

#     MgX, FeX, CaX, AlY, FeY, MgY, SiY 
E = np.array([[1., 0., 0., 1., 0., 0., 0.],
              [0., 1., 0., 1., 0., 0., 0.],
              [0., 0., 1., 1., 0., 0., 0.],
              [1., 0., 0., 0., 0., 0.5, 0.5]])
             

b = np.array([2.5, 0.29, 0.65, 0.88])
sigma_b = np.array([0.04, 0.02, 0.09, 0.07])

covar_comp = np.diag([s*s for s in sigma_b])
covar_endmembers = pinv(S).T.dot(covar_comp).dot(pinv(S))
print(covar_endmembers)

"""
