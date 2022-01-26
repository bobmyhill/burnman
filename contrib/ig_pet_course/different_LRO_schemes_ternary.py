from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

R = 8.31446

# 1 site total
def S_std(x_Mg1, x_Al, T, f):
    x_Al1 = x_Al
    x_Al2 = x_Al
    x_Si2 = x_Mg1 
    x_Si1 = 1. - x_Mg1 - x_Al1
    x_Mg2 = 1. - x_Si2 - x_Al2
    
    f2 = f*x_Mg2*x_Mg1
    return -0.5*R*(1. - f2)*(x_Al1*np.log(x_Al1)
                             + x_Mg1*np.log(x_Mg1)
                             + x_Si1*np.log(x_Si1)
                             + x_Al2*np.log(x_Al2)
                             + x_Mg2*np.log(x_Mg2)
                             + x_Si2*np.log(x_Si2))

def G_std(x_Mg1, x_Al, T, f, W21, W22, W23, W4, W6):
    x_Mg2 = 1. - x_Mg1 - x_Al
    Q = (x_Mg1 - x_Mg2)
    S_ideal = S_std(x_Mg1, x_Al, T, f)
    return (W6*np.power(Q, 6.)*(1. - x_Al) 
            + W4*np.power(Q, 4.)*(1. - x_Al) 
            + W21*np.power(Q, 2.)*(1. - x_Al) 
            + W22*np.power(Q, 2.)*(x_Al) 
            + (1. - x_Al)*x_Al*W23
            - S_ideal*T)

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

if False:
    fig = plt.figure()
    ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

    ax[0].scatter(dat[:, 0], dat[:, 1])

    for x_Al in [1.e-10, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(x_Al)
        for i, T in enumerate(temperatures):

            f = 0.
            W21 = -8.e3
            W22 = -8.e3
            W4 = 0.
            W6 = 0.
            sol = minimize(G_std, [0.0001], args=(x_Al, T, f, W21, W22, 0., W4, W6), bounds=[[1.e-10, 1. - x_Al - 1.e-5]])
            Gs[i] = sol.fun
            Qs[i] = sol.x
            Ss[i] = S_std(Qs[i], x_Al, T, f)

            f = 0.95
            W21 = -8.e3
            W22 = -16.e3
            W6 = -6.e3
            #W6 = -1.83e3 # for f = 0
            sol = minimize(G_std, [0.0001], args=(x_Al, T, f, W21, W22, 0., W4, W6), bounds=[[1.e-10, 1. - x_Al - 1.e-5]])
            G2s[i] = sol.fun
            Q2s[i] = sol.x
            S2s[i] = S_std(Q2s[i], x_Al, T, f)

        Cps = temperatures*np.gradient(Ss, temperatures, edge_order=2)
        Cp2s = temperatures*np.gradient(S2s, temperatures, edge_order=2)

        ax[0].plot(temperatures, Ss)
        ax[0].plot(temperatures, S2s)
        ax[1].plot(temperatures, Cps)
        ax[1].plot(temperatures, Cp2s)
    ax[1].set_ylim(0., 20.)
    plt.show()


x_Als = np.linspace(1.e-5, 1. - 1.e-5, 101)
Gs = np.empty_like(x_Als)
Ss = np.empty_like(x_Als)
Cps = np.empty_like(x_Als)
Qs = np.empty_like(x_Als)
G2s = np.empty_like(x_Als)
S2s = np.empty_like(x_Als)
Cp2s = np.empty_like(x_Als)
Q2s = np.empty_like(x_Als)

fig = plt.figure(figsize=(6, 6))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

Ts = np.linspace(1073., 3673., 14)
for T in Ts:
    print(T)
    for i, x_Al in enumerate(x_Als):

        f = 0.
        W21 = -16.e3
        W22 = -16.e3
        W23 = 0.e3
        W4 = 0.
        W6 = 0.
        sol = minimize(G_std, [0.0001], args=(x_Al, T, f, W21, W22, W23, W4, W6), bounds=[[1.e-10, 1. - x_Al - 1.e-10]])
        Gs[i] = sol.fun
        Qs[i] = sol.x
        Ss[i] = S_std(Qs[i], x_Al, T, f)

        f = 0.95
        W21 = -8.e3
        W22 = -20.e3
        W23 = 6.e3
        W6 = -6.e3
        #W6 = -1.83e3 # for f = 0
        sol = minimize(G_std, [0.0001], args=(x_Al, T, f, W21, W22, W23, W4, W6), bounds=[[1.e-10, 1. - x_Al - 1.e-10]])
        G2s[i] = sol.fun
        Q2s[i] = sol.x
        S2s[i] = S_std(Q2s[i], x_Al, T, f)

    #Cps = temperatures*np.gradient(Ss, temperatures, edge_order=2)
    #Cp2s = temperatures*np.gradient(S2s, temperatures, edge_order=2)

    ax[0].plot(1.-x_Als, Ss)
    if T > 3800.:
        ax[1].plot(1.-x_Als, Ss, linestyle=':')
    ax[1].plot(1.-x_Als, S2s)
    ax[2].plot(1.-x_Als, Gs - (1.-x_Als)*Gs[0])
    ax[3].plot(1.-x_Als, G2s - (1.-x_Als)*G2s[0])
for i in range(4):
    ax[i].set_xlim(0., 1.)
plt.show()