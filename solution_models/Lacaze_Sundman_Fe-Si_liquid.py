import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

L = [lambda T: -164434.6 + 41.9773*T,
     lambda T: -21.523*T,
     lambda T: -18821.542 + 22.07*T,
     lambda T: 9695.8]

def gibbs_excess(x_Si, T):
    x_Fe = 1. - x_Si
    gibbs = 0
    for i in xrange(4):
        gibbs = gibbs + x_Si*x_Fe*np.power((x_Fe - x_Si), i)*L[i](T)
    return gibbs


def margules(data, a, b):
    g = []
    for datum in data:
        x_Si, T = datum
        W = a - b*T
        x_Fe = 1. - x_Si
    
        g.append(x_Si*x_Fe*W)
    return g

compositions = np.linspace(0.0001, 1., 101) # 1. = Fe0.5Si0.5
gibbs = np.empty_like(compositions)

cTs = []
gs = []
for T in [1473., 1673., 1873.]:
    for i, c in enumerate(compositions):
        gibbs[i] = gibbs_excess(c/2., T) - c*gibbs_excess(0.5, T)

        cTs.append([c, T])
        gs.append(gibbs[i])
        
    plt.plot(compositions, gibbs, label=str(T)+' K')


popt, pcov = curve_fit(margules, cTs, gs)
print popt

for T in [1473., 1673., 1873.]:
    cT = zip(*[compositions, compositions*0. + T])
    plt.plot(compositions, margules(cT, *popt))

plt.legend(loc='lower left')
plt.show()

# Now let's look at the total excesses

def total_gibbs_excess(x_Si, T):
    x_Fe = 1. - x_Si
    gibbs = 8.31446*T*(x_Fe*np.log(x_Fe) + x_Si*np.log(x_Si))
    for i in xrange(4):
        gibbs = gibbs + x_Si*x_Fe*np.power((x_Fe - x_Si), i)*L[i](T)
    return gibbs

x_FeSis = np.empty_like(compositions)
cTs = []
gs = []
for T in [1000., 1500., 2000, 3000, 4000]:
    for i, c in enumerate(compositions):
        gibbs[i] = total_gibbs_excess(c/2., T) - c*total_gibbs_excess(0.5, T)

        x_Si = c/2.
        x_Fe = 1. - c/2.
        n_FeSi = x_Si
        n_Fe = x_Fe - x_Si
        x_FeSis[i] = n_FeSi/(n_FeSi + n_Fe)
        
        cTs.append([x_FeSis[i], T])
        gs.append(gibbs[i])
        
    plt.plot(compositions, gibbs, label=str(T)+' K')

plt.show()



'''
# Two entropy formalisms
def entropy_FeSi(x_Si):
    x_Fe = 1. - x_Si
    return x_Fe*np.log(x_Fe) + x_Si*np.log(x_Si)




compositions = np.linspace(0.0001, 0.499999, 101) # 0.5 = Fe0.5Si0.5
S1 = np.empty_like(compositions)
S2 = np.empty_like(compositions)
for i, x_Si in enumerate(compositions):
    # 1
    S1[i] = entropy_FeSi(x_Si) #- 2.*x_Si*entropy_FeSi(0.5)

    # 2
    x_Fe = 1. - x_Si
    n_FeSi = x_Si
    n_Fe = x_Fe - n_FeSi
    f_FeSi = n_FeSi/(n_Fe + n_FeSi)

    n_total = n_Fe + n_FeSi
    print f_FeSi, n_total
    S2[i] = entropy_FeSi(f_FeSi) * n_total
    

plt.plot(compositions, S1)
plt.plot(compositions, S2)
plt.show()
'''
