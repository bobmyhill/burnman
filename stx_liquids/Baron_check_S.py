import numpy as np
import matplotlib.pyplot as plt

lmda = 1.43
WA = 70. # decent fit at 80 GPa, 3600 K
WB = 25. # decent fit at 80 GPa, 3600 K

Xs = np.linspace(0., 1., 101)
delta_Sc = np.empty_like(Xs)
delta_Snc = np.empty_like(Xs)
delta_S = np.empty_like(Xs)

for i, X in enumerate(Xs):    
    Y = X/(X + lmda*(1. - X))
    delta_Sc[i] = -8.31446*(X * np.log(X) + (1. - X) * np.log(1. - X))
    delta_Snc[i] = WA*Y*Y*(1. - Y) + WB*Y*(1. - Y)*(1. - Y)
    delta_S[i] = delta_Sc[i] + delta_Snc[i]

plt.plot(Xs, delta_S)
plt.show()
