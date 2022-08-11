import numpy as np
import matplotlib.pyplot as plt


def gibbs(P, T, Q):
    Tc0 = 847.
    VD = 1.36e-6
    SD = 5.76
    Tc = Tc0 + VD/SD*P
    return SD*(T - Tc)*(Q*Q - 1) + 1./3.*Tc0*(Q*Q*Q*Q*Q*Q - 1.)


for P in np.linspace(0., 2.e9, 5):
    Qs_eqm = []
    Gs_eqm = []

    temperatures = np.linspace(0., 2000., 1001)
    Qs = np.linspace(0., 5., 1001)
    for T in temperatures:
        Gs = gibbs(P, T, Qs)
        # plt.plot(Qs, Gs)
        i = np.argmin(Gs)
        Qs_eqm.append(Qs[i])
        Gs_eqm.append(Gs[i])

    plt.plot(Qs_eqm, Gs_eqm)

plt.show()