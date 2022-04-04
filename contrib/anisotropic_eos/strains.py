import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
# from burnman.tools.eos import check_anisotropic_eos_consistency
# from burnman.minerals.SLB_2011 import stishovite
# from burnman.minerals.HP_2011_ds62 import stv as stishovite_HP

Pcstar = 49.
Pc = 99.7
l1 = -8.
l2 = 24.62
l3 = 17.
l4 = 20.
l6 = 20.
b = 10.94
a = -0.04856


def Cij0(pressure):
    C110 = 578 + 5.38 * pressure
    C330 = 776 + 4.94 * pressure
    C120 = 86 + 5.38 * pressure
    C130 = 191 + 2.72 * pressure
    C440 = 252 + 1.88 * pressure
    C660 = 323 + 3.10 * pressure

    return np.array([C110, C110, C330, C120, C130, C130, C440, C440, C660])


def strains(Q, pressure):
    C110, _, C330, C120, C130, _, C440, _, C660 = Cij0(pressure)

    ea = l2 / (0.5*(C110 - C120))*Q/2.
    eb = l1/(0.5*(C110 - C120))*Q*Q/2.
    e1 = - ea - eb
    e2 = ea - eb
    e3 = - l3/C330*Q*Q

    return np.array([ea, eb, e1, e2, e3])


Qs = np.linspace(-1, 1., 101)
for pressure in [0., 100.]:
    es = strains(Qs, pressure)
    Fs = np.array([np.diag(e[2:] + 1.) for e in es.T])

    """
    dets = np.linalg.det(Fs)
    print(Fs.shape, dets.shape)

    es[2:] = np.array([np.diag(F/np.cbrt(dets[i])) - 1.
                       for i, F in enumerate(Fs)]).T
    es[0] = (es[3] - es[2])/2.
    es[1] = -(es[3] + es[2])/2.
    """
    plt.plot(Qs, es[0], linestyle=':', label='symmetry breaking')
    plt.plot(Qs, es[1], linestyle='--', label='non-symmetry breaking')

    labels = ['a', 'b', 'c']

    for i, e in enumerate(es[2:]):
        plt.plot(Qs, e, label=labels[i])

    """
    Fs = np.array([np.diag(e[2:] + 1.) for e in es.T])
    dets = np.linalg.det(Fs)
    plt.plot(Qs, -1. + dets)
    """

es = np.array([np.diag(expm(np.diag([-0.05 * Q + 0.011*Q*Q,
                                     0.05 * Q + 0.011*Q*Q,
                                     -0.022*Q*Q]))) - 1. for Q in Qs]).T

for i, e in enumerate(es):
    plt.plot(Qs, e, label=labels[i], linestyle=':')

plt.legend()
plt.show()


# Notes

# Cijs are calculated as V * d^2F/deidej

# There are local strain vectors associated
# with changes in volume and order parameter.
# If we constrain the order parameter to be
# volume independent, then these two strain vectors are
# orthogonal to each other.

# The change in Helmholtz free energy with strain
# is defined along these vectors by the order disorder
# model and by the ordered and disordered
# compression curves.

# There are four other independent strain vectors.
# Along these, we assume that the change in Helmholtz
# free energy is defined by the local Cij.
# (do we? is that what Carpenter does ?)

# We know:
# dF/dx (where x=[V,Q]): change in pressure
# R^{-1} = d2F / dxdx (where x=[V,Q])
# we also know de/dx


# STEP 0
# Define F(p); for the time being we are not interested
# in any excess pressures or entropies.

# STEP 1
# Define chi_ijkl as a function of p
# as well as V and T?


# F_ij = exp_M (chi_ijkl delta_kl)
# ln_M F_ij = chi_ijkl delta_kl

# We use the identity log(det(F)) = tr(log_M(F))
# In this case we want det(F) = 1 along compositional
# paths (i.e. log(det(F)) = 0).

# So we're looking for
# tr(log_M(F)) = 0
# tr(chi_ijkl delta_kl) = 0
# i.e. chi_iikk == 0
# this is just the Q-dependent part of chi

# This would give us consistent strains
# and static S_ijkls and C_ijkls


# STEP 2
# Defining the effective C_ijkls (d2F/dede).
# We have only defined properties at hydrostatic stresses.

# Can this be done? What is the mathematics given the
# susceptibility shenanigans?

# What do Carpenter's expressions actually represent?


# Maybe use the inverse matrix derivative identity?
# d Y^-1 = Y^-1 dY Y^-1
# Let Y = dchi_ijkl/dP = S_ijkl
# So Y^-1 = C_ijkl

# Hard to tell how this helps with the
# susceptibility problem.


# C_ij = C_ij0 - (C_ik d_ek/d_Qm)  d^2F/dQ_mdQ+n (C_jl de_l/dQ_n)
