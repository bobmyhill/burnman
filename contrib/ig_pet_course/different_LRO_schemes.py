from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

R = 8.31446

def S_std(Q, T, f):
    x = 0.5*(1. - Q)
    f2 = f*x*(1. - x)
    return -R*(1. - f2)*(x*np.log(x) + (1. - x)*np.log(1. - x))

def G_std(Q, T, f, W2, W4, W6):
    S_ideal = S_std(Q, T, f)
    return W6*np.power(Q, 6.) + W4*np.power(Q, 4.) + W2*np.power(Q, 2.) - S_ideal*T

temperatures = np.linspace(10., 5000., 1001)
Gs = np.empty_like(temperatures)
Ss = np.empty_like(temperatures)
Cps = np.empty_like(temperatures)
Qs = np.empty_like(temperatures)
G2s = np.empty_like(temperatures)
S2s = np.empty_like(temperatures)
Cp2s = np.empty_like(temperatures)
Q2s = np.empty_like(temperatures)

dat = np.loadtxt('/home/bob/projects/burnman_broken/order_disorder/S_conf_maj_Vinograd.dat')

Qtests = np.linspace(0., 0.999999, 10001)
for i, T in enumerate(temperatures):
    """
    Gtests = G_std2(Qtests, T)
    idx = np.argmin(Gtests)
    Gs[i] = Gtests[idx]
    Qs[i] = Qtests[idx]
    Ss[i] = -(Gs[i] - W4*np.power(Qs[i], 4.) - W2*np.power(Qs[i], 2.))/T
    """
    f = 0.
    W2 = -16.e3
    W4 = 0.
    W6 = 0.
    sol = minimize(G_std, [0.0001], args=(T, f, W2, W4, W6), bounds=[[0., 0.999999]])
    Gs[i] = sol.fun
    Qs[i] = sol.x
    Ss[i] = S_std(Qs[i], T, f)
    
    f = 0.95
    W2 = -8.e3
    W6 = -6.e3
    #W6 = -1.83e3 # for f = 0
    sol = minimize(G_std, [0.0001], args=(T, f, W2, W4, W6), bounds=[[0., 0.999999]])
    G2s[i] = sol.fun
    Q2s[i] = sol.x
    S2s[i] = S_std(Q2s[i], T, f)

Cps = temperatures*np.gradient(Ss, temperatures, edge_order=2)
Cp2s = temperatures*np.gradient(S2s, temperatures, edge_order=2)

fig = plt.figure()
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

ax[0].scatter(dat[:, 0], dat[:, 1])
ax[0].plot(temperatures, Ss)
ax[0].plot(temperatures, S2s)
ax[1].plot(temperatures, Cps)
ax[1].plot(temperatures, Cp2s)
ax[1].set_ylim(0., 20.)
plt.show()
