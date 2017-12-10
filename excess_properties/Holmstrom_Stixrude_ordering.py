from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

R = 8.31445
eps = np.finfo(float).eps


    

class solution(object):
    def __init__(self, m, n, deltaH, Wab, Wao, Wbo):
        self.m = m
        self.n = n
        self.deltaH = deltaH
        self.Wab = Wab
        self.Wao = Wao
        self.Wbo = Wbo

    def _delta_gibbs(self, Q, X, T):
        pa = 1 - X - self.m/(self.m+self.n)*Q
        pb = X - self.n/(self.m+self.n)*Q
        
        A = self.deltaH + (self.m+self.n)*self.Wao - self.n*self.Wab
        B = 2./(self.m+self.n) * (-self.m*self.m*self.Wao +
                                  self.m*self.n*(self.Wab - self.Wao - self.Wbo) -
                                  self.n*self.n*self.Wbo)
        C = self.m*(self.Wbo - self.Wao - self.Wab) + self.n*(self.Wbo - self.Wao + self.Wab)
        
        Kd = (1 - pa)*(1 - pb)/(pa*pb)
        
        return A + B*Q + C*X + self.m*self.n*R*T*np.log(Kd)
    
    def order(self, X, T):
        try:
            max_order = np.min((self.m+self.n)*np.array([X/self.m, (1. - X)/self.n]))
            return brentq(self._delta_gibbs, 100.*eps, max_order-100.*eps, args=(X, T))
        except:
            return 0.
        
jd_di = solution(1., 1., -6000., 26000., 16000., 16000.)
temperatures = np.linspace(300., 1400., 201)
Qs = np.empty_like(temperatures)

# Jadeite-Diopside example from Holland and Powell (1996)
'''
for X in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for i, T in enumerate(temperatures):
        Qs[i] = jd_di.order(X, T)
    plt.plot(temperatures, Qs, label='{0:.1f}'.format(X))
plt.legend(loc='best')
plt.show()
'''

# For convergent order
deltaH_O = -28000. # for reaction FeO + FeO -> 2Fe0.5Fe0.5O
W_HSLS_elastic = 3200.*4. # computed from ferropericlase_elastic_ternary.py 


FeO = solution(0.5, 0.5, deltaH_O,
               2.*(deltaH_O + W_HSLS_elastic/4.),
               W_HSLS_elastic/4., W_HSLS_elastic/4.)

temperatures = np.linspace(300., 4000., 201)
Qs = np.empty_like(temperatures)




for X in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for i, T in enumerate(temperatures):
        Qs[i] = FeO.order(X, T)
    plt.plot(temperatures, Qs, label='{0:.1f}'.format(X))
plt.legend(loc='best')





data = [[2000., [0.39, 0.53], [0.59, 0.91]],
        [3000., [0.65, 0.47], [0.44, 0.81]],
        [4000., [0.76, 0.49], [0.06, 0.31]]]

for T, f, Q in data:
    plt.scatter(T, Q[1], color='black')
    plt.scatter(T, Q[0], color='red')

plt.show()
