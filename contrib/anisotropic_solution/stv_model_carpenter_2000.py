import numpy as np
import matplotlib.pyplot as plt

# Values from Table 2


def C11(P): return 578 + 5.38*P
def C33(P): return 776 + 4.94*P
def C12(P): return 86 + 5.38*P
def C13(P): return 191 + 2.72*P
def C44(P): return 252 + 1.88*P
def C66(P): return 323 + 3.10*P


Pastc = 49
Pc = 49. + 50.7

l1 = -8.
l2 = 24.62
l3 = 17.
a = -0.04856
b = 10.94

# Values from text
l4 = 20.
l6 = 20.


def gibbs(P):
    # Equation 1
    if P < Pastc:
        return 0.
    else:
        return 0.5*a*(P - Pastc)*Qsqr(P) + 0.25*bast(P)*Qsqr(P)*Qsqr(P)


def Qsqr(P):
    # Equation 3
    return a / bast(P) * (Pastc - P)


def bast(P):
    # Equation 5
    return b - 2.*((l3*l3*(C11(P) + C12(P)) +
                    2.*l1*l1*C33(P) - 4.*l1*l3*C13(P)) /
                   ((C11(P) + C12(P))*C33(P) - 2.*C13(P)*C13(P)))


def chi(P):
    # Equations 9, 10
    if P < Pastc:
        return 1./(a*(P - Pc))
    else:
        return 1./(2.*a*b/bast(P)*(Pastc - P) + a*(Pastc - Pc))


def CN(P, mode='eqm'):
    # Expressions in Table 1
    C = np.zeros((7, 7))

    if mode == 'eqm':
        X = chi(P)
        Q = np.sqrt(Qsqr(P))
    else:
        X = 1.
        Q = 1.

    if (P < Pastc and mode == 'eqm') or mode == 'tet':
        C[1, 1] = C11(P) - l2*l2*X
        C[2, 2] = C[1, 1]
        C[3, 3] = C33(P)
        C[1, 2] = C12(P) + l2*l2*X
        C[1, 3] = C13(P)
        C[2, 3] = C[1, 3]
        C[4, 4] = C44(P)
        C[5, 5] = C44(P)
        C[6, 6] = C66(P)
    else:
        C[1, 1] = C11(P) - (4.*l1*l1*Q*Q + l2*l2 + 4.*l1*l2*Q)*X
        C[2, 2] = C11(P) - (4.*l1*l1*Q*Q + l2*l2 - 4.*l1*l2*Q)*X
        C[3, 3] = C33(P) - 4.*l3*l3*Q*Q*X
        C[1, 2] = C12(P) - (4.*l1*l1*Q*Q - l2*l2)*X
        C[1, 3] = C13(P) - (4*l1*l3*Q*Q + 2.*l2*l3*Q)*X
        C[2, 3] = C13(P) - (4*l1*l3*Q*Q - 2.*l2*l3*Q)*X
        C[4, 4] = C44(P) + 2.*l4*Q
        C[5, 5] = C44(P) - 2.*l4*Q
        C[6, 6] = C66(P) + 2.*l6*Q*Q

    C[2, 1] = C[1, 2]
    C[3, 1] = C[1, 3]
    C[3, 2] = C[2, 3]

    return C


pressures = np.linspace(0., 100., 101)
Ceqm = np.zeros((101, 7, 7))
Ctet = np.zeros((101, 7, 7))
Cort = np.zeros((101, 7, 7))
S = np.zeros((101, 7, 7))
KS = np.zeros(101)

for i, P in enumerate(pressures):
    Ceqm[i] = CN(P, mode='eqm')
    Ctet[i] = CN(P, mode='tet')
    Cort[i] = CN(P, mode='ort')

    S[i, 1:, 1:] = np.linalg.inv(Ceqm[i, 1:, 1:])
    KS[i] = 1./np.sum(S[i, 1:4, 1:4])

for (i, j) in [(1, 1),
               (2, 2),
               (3, 3),
               (4, 4),
               (5, 5),
               (6, 6),
               (1, 2),
               (1, 3),
               (2, 3)]:
    ln, =  plt.plot(pressures, Ceqm[:, i, j], label=f'$C_{{{i}{j}}}$')
    # plt.plot(pressures, Cort[:, i, j],
    #         label=f'$C_{{{i}{j}}}$', linestyle=':', c=ln.get_color())

plt.plot(pressures, KS, label='$K_S$')
plt.legend()
plt.show()

"""
pressures = np.linspace(48., 52., 101)
G = np.array([gibbs(P) for P in pressures])
Vrel = np.gradient(G, pressures, edge_order=2)
plt.plot(pressures, Vrel)
plt.show()
"""
