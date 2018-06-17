import numpy as np
from sympy import Matrix, nsimplify
import warnings
from scipy.optimize import minimize, nnls

R = 8.31446

def logish(x, eps=1.e-5):
    """
    2nd order series expansion of log(x) about eps: log(eps) - sum_k=1^infty (f_eps)^k / k
    Prevents infinities at x=0
    """
    f_eps = 1. - x/eps
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        mask = x>eps
    ln = np.where(x<=eps, np.log(eps) - f_eps - f_eps*f_eps/2., 0.)
    ln[mask] = np.log(x[mask])
    return ln

def logishf(x, eps=1.e-5):
    """
    2nd order series expansion of log(x) about eps: log(eps) - sum_k=1^infty (f_eps)^k / k
    Prevents infinities at x=0
    """
    f_eps = 1. - x/eps
    if x < eps:
        return np.log(eps) - f_eps - f_eps*f_eps/2.
    else:
        return np.log(x)

def inverseish(x, eps=1.e-5):
    """
    1st order series expansion of 1/x about eps: 2/eps - x/eps/eps
    Prevents infinities at x=0
    """
    mask = x>eps
    oneoverx = np.where(x<=eps, 2./eps - x/eps/eps, 0.)
    oneoverx[mask] = 1./x[mask]
    return oneoverx


# The following formulation is taken
# from the one-sublattice multicomponent quasichemical formalism of
# Pelton and Chartrand (2001)
# (METALLURGICAL AND MATERIALS TRANSACTIONS A; VOLUME 32A, 1355-1360)

# In particular, this implementation uses the *second* formulation of the
# Gibbs free energy excesses
# (i.e. equation 11, rather than equation 9, and
#  Part III.B.3, rather than Part III.A.3 for the interpolation formulae)


def unflatten_pairs(nij, n, ij, ZZ):
    n_pairs = np.zeros((n, n))
    for k, (i, j) in enumerate(ij):
        n_pairs[i][j] = nij[k]
    return n_pairs
    
def gibbs(nij, pressure, temperature, params):
    n_pairs = unflatten_pairs(nij, params.n, params.ij, params.ZZ)
    params.n_pairs = n_pairs
    
    n = params.n

    # coordination state for each component
    invZ = np.array([(np.sum(n_pairs[i,:]/params.ZZ[i,:]) +
                      np.sum(n_pairs[:,i]/params.ZZ[i,:])) /
                     (np.sum(n_pairs[i,:]) + np.sum(n_pairs[:,i]) + 1.e-16)
                     for i in range(n)]) # coordination of component m, Z_m, eqs 12, 13

    Zn_component = np.array([np.sum(n_pairs[i,:]) + np.sum(n_pairs[:,i])
                            for i in range(n)]) # eq. 2
    n_component = Zn_component*invZ
    params.n_component = n_component
    
    p_pairs = n_pairs/np.sum(n_pairs) # Eq. 3; pair fractions, Xmn in PC2001
    p_component = n_component/np.sum(n_component) # Eq. 4; site fractions, Xm in PC2001
    p_coord = Zn_component/np.sum(Zn_component) # Eq. 5; coordination equivalent fractions; Ym in PC2001

    chi = np.zeros((n, n))
    ksi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                num_indices, denom_indices = params.ternary_asymmetry_indices[i][j]
                
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
                
    deltaG_pairs = params.deltaG_0_pairs(pressure, temperature)
    
    # Binary g, eq. 30
    for (i, j, ii, jj, gijkl) in params.g_binary(pressure, temperature):
        deltaG_pairs[i, j] += np.power(chi[i,j], ii)*np.power(chi[j, i], jj)*gijkl
        #deltaG_pairs[i, j] += np.power(p_pairs[i,i], ii)*np.power(p_pairs[j,j], jj)*gijkl
        
    # Ternary q, eq. 24
    for (i, j, k, ii, jj, kk, q) in params.q_ternary(pressure, temperature):
        # in the ijk ternary, which component (if any) is asymmetric?
        asymmetric_component = [a[2]
                                for a in params.ternary_asymmetries
                                if (i in a and j in a and k in a)]

        if asymmetric_component == [] or asymmetric_component == [k]: # part of "l" sum
            deltaG_pairs[i, j] += (np.power(ksi[i][j]/(ksi[i][j] + ksi[j][i]), ii) *
                                   np.power(ksi[j][i]/(ksi[i][j] + ksi[j][i]), jj) *
                                   q*p_coord[k]*np.power(1. - ksi[i][j] - ksi[j][i], kk-1.))

        elif asymmetric_component == [i]: # part of "m" sum
            deltaG_pairs[i, j] += (np.power(ksi[i][j]/(ksi[i][j] + ksi[j][i]), ii) *
                                   np.power(ksi[j][i]/(ksi[i][j] + ksi[j][i]), jj) *
                                   q*p_coord[k] / ksi[j][i] *
                                   np.power(1. - p_coord[j]/ksi[j][i], kk-1.))
            
        elif asymmetric_component == [j]: # part of "n" sum
            deltaG_pairs[i, j] += (np.power(ksi[i][j]/(ksi[i][j] + ksi[j][i]), ii) *
                                   np.power(ksi[j][i]/(ksi[i][j] + ksi[j][i]), jj) *
                                   q*p_coord[k] / ksi[i][j] *
                                   np.power(1. - p_coord[i]/ksi[i][j], kk-1.))
    
    # Ternary g, eq. 30
    for (i, j, k, ii, jj, kk, g) in params.g_ternary(pressure, temperature):
        # in the ijk ternary, which component (if any) is asymmetric?
        asymmetric_component = [a[2]
                                for a in params.ternary_asymmetries
                                if (i in a and j in a and k in a)]
        
        if asymmetric_component == [] or asymmetric_component == [k]: # part of "l" sum
            deltaG_pairs[i, j] += (np.power(chi[i][j], ii) * np.power(chi[j][i], jj) *
                                   g*p_coord[k]*np.power(1. - ksi[i][j] - ksi[j][i], kk-1.))
            
        elif asymmetric_component == [i]: # part of "m" sum
            deltaG_pairs[i, j] += (np.power(chi[i][j], ii) * np.power(chi[j][i], jj) *
                                   g*p_coord[k] / ksi[j][i] *
                                   np.power(1. - p_coord[j]/ksi[j][i], kk-1.))
      
        elif asymmetric_component == [j]: # part of "n" sum
            deltaG_pairs[i, j] += (np.power(chi[i][j], ii) * np.power(chi[j][i], jj) *
                                   g*p_coord[k] / ksi[i][j] *
                                   np.power(1. - p_coord[i]/ksi[i][j], kk-1.))
    

    ij = [(i, j) for i in xrange(n) for j in xrange(i+1, n)]

    deltaS_conf = -R*(np.sum(n_component*logish(p_component)) +
                      np.sum([n_pairs[i][i]*logish(p_pairs[i][i]/(p_coord[i]*p_coord[i]))
                              for i in range(len(p_coord)) if p_coord[i] > 0.]) +
                      np.sum([n_pairs[i][j]*logish(p_pairs[i][j]/(2.*p_coord[i]*p_coord[j]))
                              for i, j in ij if p_coord[i] > 0. and p_coord[j] > 0.])) # Eq. 8
    params.configurational_entropy = deltaS_conf
    params.gibbs = (np.sum(n_component*params.g_component(pressure, temperature))
                    - temperature*deltaS_conf
                    + np.sum(n_pairs*(deltaG_pairs/2.))) # Eq. 7
    return params.gibbs


def gibbs_components_constrained(pressure, temperature, component_proportions, params):
    n = len(component_proportions) # number of components
    npr = (n*n + n)/2 # number of unique pairs
    
    pair_bounds = [(0., None) for i in range(npr)] # all pairs must be > 0.
    
    bulk_constraints = []
    for k in range(n):
        def f(x, k = k, proportions=component_proportions, ij=params.ij, ZZ=params.ZZ):
            return ( np.sum([x[m]/ZZ[ij[m][0], ij[m][1]] for m in range(npr)
                             if ij[m][0] == k]) +
                     np.sum([x[m]/ZZ[ij[m][1], ij[m][0]] for m in range(npr)
                             if ij[m][1] == k]) - proportions[k] )
        bulk_constraints.append({'type': 'eq', 'fun' : f})

    # one possible guess is for *only* i-i pairs to have any non-zero concentration
    guess = np.zeros(npr)
    guess[:n] = [0.5*component_proportions[i]*params.ZZ[i,i] for i in range(n)]
    guess_gibbs = gibbs(guess, pressure, temperature, params)

    scaling = 1.e4 # improve scaling for solver
    gibbs_func = lambda nij, pressure, temperature, params: (gibbs(nij,
                                                                   pressure,
                                                                   temperature,
                                                                   params) -
                                                             guess_gibbs)/scaling
    
    sol = minimize(gibbs_func, guess, args=(pressure, temperature, params),
                   bounds=pair_bounds, constraints=bulk_constraints)

    return sol.fun*scaling + guess_gibbs


def stoichiometric_matrix(elements, formulae):
    # Populate the stoichiometric matrix
    def f(i,j):
        e = elements[i]
        if e in formulae[j]:
            return nsimplify(formulae[j][e])
        else:
            return 0
    return Matrix( len(elements), len(formulae), f )
    

def equilibrate_liquid(liquid, pressure, temperature, bulk_composition, guess=[]):
    # in the FeOS ternary, there is FeII and FeIII, which both
    # correspond to an Fe atom (which is strictly either II-fold or III-fold coordinated)
    # Charge balance is implicit, but bulk compositional constraints must be modified
    # to take into account 

    n_elements = len(bulk_composition)

    elements = list(set(bulk_composition.keys()))
    bulk_composition_vector = np.array([bulk_composition[e] for e in elements])
    
    M = stoichiometric_matrix(elements, liquid.component_formulae)
    _, inds = M.T.rref() # linearly independent rows
    M = np.array(M)
    n = liquid.n # number of components
    npr = (n*n + n)/2 # number of unique pairs
    
    pair_bounds = [(0., None) for i in range(npr)] # all pairs must be > 0.

    
    bulk_constraints = []
    for k in inds: #range(n_elements):
        def f(x, k=k, proportion=bulk_composition_vector, ij=liquid.ij, M=M, ZZ=liquid.ZZ):
            return ( np.sum([pl*(np.sum([x[m]/ZZ[ij[m][0], ij[m][1]] for m in range(npr)
                                         if ij[m][0] == l]) +
                                 np.sum([x[m]/ZZ[ij[m][1], ij[m][0]] for m in range(npr)
                                         if ij[m][1] == l]))
                             for l, pl in enumerate(M[k])]) - proportion[k])
        bulk_constraints.append({'type': 'eq', 'fun' : f})


    if guess == []:
        # possible_proportions:
        guess_proportions = nnls(M, bulk_composition_vector)[0]
        # one possible guess is for *only* i-i pairs to have any non-zero concentration
        guess = np.zeros(npr)
        guess[:n] = [0.5*guess_proportions[i]*liquid.ZZ[i][i] for i in range(n)]

    guess_gibbs = gibbs(guess, pressure, temperature, liquid)
    
    scaling = 1.e4 # improve scaling for solver
    gibbs_func = lambda nij, pressure, temperature, params: (gibbs(nij,
                                                                   pressure,
                                                                   temperature,
                                                                   params) -
                                                             guess_gibbs)/scaling
    
    sol = minimize(gibbs_func, guess, args=(pressure, temperature, liquid),
                   bounds=pair_bounds, constraints=bulk_constraints)

    liquid.nij = sol.x
    
    return sol.fun*scaling + guess_gibbs
