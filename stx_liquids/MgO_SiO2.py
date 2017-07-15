import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


b = [0.6887, 1.3774] # in text between equations 9 and 10
R = 8.31446 # gas_constant

class values():
    name = 'Model variable storage'



def pair_proportions(nn, T, n, v):
    # nAA, nAB, nBB
    v.npr = np.abs(np.array( [ [nn[0], nn[1]],
                               [nn[1], nn[2]] ]))
    v.Xpr = v.npr/np.sum(nn)
    
    v.Y = [b[0]*n[0] / (b[0]*n[0] + b[1]*n[1]),
           b[1]*n[1] / (b[0]*n[0] + b[1]*n[1])] # eqn 3
    
    v.delta_G = (-33976. + 6.0*T +
                 53760.*np.power(v.Y[1], 3.) -
                 107429.*np.power(v.Y[1], 5.) +
                 (126025. - 20.*T)*np.power(v.Y[1], 7.))*4.184 # value converted to J, eqn 20 BP1987
    
    out = []
    out.append(v.Y[0] - (v.Xpr[0][0] + v.Xpr[0][1]/2.)) # eqn 7
    out.append(v.Y[1] - (v.Xpr[1][1] + v.Xpr[0][1]/2.)) # eqn 8
    out.append(4.*np.exp(-v.delta_G/(R*T))*v.Xpr[0][0]*v.Xpr[1][1] - v.Xpr[0][1]*v.Xpr[0][1]) # eqn 9
    return out




T = 1600. + 273.15 # temperature used in figures 3 and 4 

v = values()
xs_all = np.linspace(0.001, 0.999, 1001)


xs = []
delta_Gs = []
Gs = []

for i, x in enumerate(xs_all):
    
    n = np.array([1. - x, x])
    
    X = n/np.sum(n)

    dT = 0.01
    
    pr0 = fsolve(pair_proportions, [n[0], 0.0, n[1]], args=(T, n, v), full_output=True)#
    if pr0[2] == 1:
        Sconf0 = -R * (X[0] * np.log(X[0]) + X[1]*np.log(X[1]) +
                       (b[0]*X[0] + b[1]*X[1]) *
                       (v.Xpr[0][0] * np.log(v.Xpr[0][0] / (v.Y[0]*v.Y[0]) ) +
                        v.Xpr[1][1] * np.log(v.Xpr[1][1] / (v.Y[1]*v.Y[1]) ) +
                        v.Xpr[0][1] * np.log(v.Xpr[0][1] / (2.*v.Y[0]*v.Y[1])) )) # first two terms eqn 5
    
        G0 = -T*Sconf0 +  (b[0]*X[0] + b[1]*X[1])*v.Xpr[0][1]/2.*v.delta_G # eqn 4, last term eqn 5
        # note typo in equation 4 (no normalisation for Xij)

        xs.append(x)
        delta_Gs.append(v.delta_G/1000./4.184)
        Gs.append(G0/4.184)

fig1 = mpimg.imread('figures/MO_SiO2_deltaG_-80_20kcal.png') # figure 4
plt.imshow(fig1, extent=[0, 1, -80, 20], aspect='auto')
plt.plot(xs, delta_Gs, marker='o')
plt.show()
    
fig1 = mpimg.imread('figures/MgO_SiO2_gibbs_mixing_-10_0_kJ_Pelton_Blander_1986.png') # figure 3
plt.imshow(fig1, extent=[0, 1, -10000, 0], aspect='auto')
plt.plot(xs, Gs, marker='o')
plt.show()
