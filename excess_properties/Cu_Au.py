import numpy as np
import matplotlib.pyplot as plt
# Elastic contribution (G(x)) using constant Omega

Xs = np.array([0., 0.25, 0.5, 0.75, 1.0])
Vs = np.array([7.112, 8.294, 8.767, 9.506, 10.218])
Bs = np.array([138, 148, 163, 0., 171])

plt.scatter(Xs, Vs - Xs*Vs[-1] - (1. - Xs)*Vs[0])
plt.show()
plt.scatter(Xs, Bs - Xs*Bs[-1] - (1. - Xs)*Bs[0])
plt.ylim(0., 30.)
plt.show()


# Chemical contribution (Eq. 4.6)
eps = [0., -4.269, -5.585, -3.849, 0.] # kcal/mol

def G(x):
    Omega = 3.6*4. # kcal/mol approx
    return Omega*x*(1. - x)


#Eq 2.13


# N_A = number A atoms
# N_B = number B atoms
# sigma = state of order

# if site i is occupied by A, S^{(i)} = -1, eta_1^{(i)} = 0, eta_0^{(i)} = 1
# if site i is occupied by B, S^{(i)} = 1,  eta_1^{(i)} = 1, eta_0^{(i)} = 0


# \Delta E = \sum_n \ksi_n(\sigma)*\var_epsilon^{(n)} + G(x) # eq. 3.9



# \xsi_m(n) = \krondelta_{nm}
