import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
from slb_qtz import (
    helmholtz_free_energy_beta,
    helmholtz_free_energy_alpha,
    qtz_alpha,
    qtz_beta,
)

from slb_qtz import quartz_helmholtz_function as F_qtz


def quartz_helmholtz_function(volume, temperature, molar_amounts):
    """
    Let all three of the endmembers be alpha quartz at a standard
    (non-zero) tilt and state of disorder

    Let Q_0 be the amount of tilt
    Let Q_1 be the "misfit" given by Wells, 2002

    Furthermore, enforce that d2F/dQ_0^2 is proportional to d2F/dQ_1^2
    We can scale Q_1 freely, so that we only consider

    Starting with a standard 2-4-6 formulation:

    F = (c_00 + c_10 Q_0^2 + c_20 Q_0^4 + c_30 Q_0^6)
        + (c_01 + c_11 Q_0^2 + c_21 Q_0^4) Q_1^2
        + (c_02 + c_12 Q_0^2) Q_1^4
        + (c_03) Q_1^6

    d2F/dQ_0^2 = (2 c_10 + 12 c_20 Q_0^2 + 30 c_30 Q_0^4)
                  + (2 c_11 + 12 c_21 Q_0^2) Q_1^2
                  + (2 c_12) Q_1^4

    d2F/dQ_1^2 = 2 (c_01 + c_11 Q_0^2 + c_21 Q_0^4)
                   + 12 (c_02 + c_12 Q_0^2) Q_1^2
                   + 30 (c_03) Q_1^4

    d2F/dQ_0^2 = a * d2F/dQ_1^2

    There are therefore only five free variables
    c_00
    c_10 = a * c_01
    c_20 = a * c_11 / 6 = a^2 * c_02
    c_30 = a * c_21 / 15 = a^2 * c_12 / 15 = a^3 * c_03

    For simplicities' sake, the three endmembers are positioned at
    Q = [-1,-1], [-1,1], [1,-1] (standard state tilt and disorder).
    Therefore, we can express the Helmholtz energy as a simple sum:

    F = F_11 + F_xs

    where the excess Helmholtz energy Fxs is zero for all the endmembers:

    F_xs = (c_00 + c_10 Q_0^2 + c_20 Q_0^4 + c_30 Q_0^6)
         + (c_10 + 6 c_20 Q_0^2 + 15 c_30 Q_0^4) Q_1^2 / a
         + (c_20 + 15 c_30 Q_0^2) Q_1^4 / a^2
         + (c_30) Q_1^6 / a^3

    We consider another endmember at Q = [0,0] (no tilt, no disorder).
    The excess energy here is (F_00 - F_11)

    c_00 = (F_00 - F_11)
    0 = ((F_00 - F_11) + c_10 + c_20 + c_30)
        + (c_10 + 6 c_20 + 15 c_30) / a
        + (c_20 + 15 c_30)/ a^2
        + c_30 / a^3
    F_11 - F_00 = ((a + 1)/a) c_10 + (1 + 6/a + 1/a^2) c_20 + (1 + 15/a + 15/a^2 + 1/a^3) c_30
    c_10 = (F_11 - F_00 - (1 + 6/a + 1/a^2)c_20 - (1 + 15/a + 15/a^2 + 1/a^3)*c_30) * (a/(a+1))

    We assume that a, c_20 and c_30 are constants.

    We can also calculate the state at standard state tilt, no disorder [1,0]:
    F_10 - F_11 = ((F_00 - F_11)/(a+1) - ((5 + 1/a)c_20 + (14 + 15/a + 1/a^2)*c_30) / (a+1))

    because a, c_20 and c_30 are all constants, this implies that the entropy (or pressure)
    change of tilting is given by:

    S_10 - S_11 = ((S_00 - S_11)/(a+1)
    S(tilt, no disorder) - S(tilt, disorder) = ((S(no tilt, no disorder) - S(tilt, disorder))/(a+1)

    Now, S(tilt, no disorder) < S(tilt, disorder) < S(no tilt, no disorder), so
    the LHS and first term on the RHS should be negative. Thus a > -1.
    Also, the LHS should be more negative than the first term on the RHS, so a < 0
    """
    n_moles = sum(molar_amounts)
    molar_fractions = molar_amounts / n_moles

    F_00 = helmholtz_free_energy_beta(temperature, volume)
    F_11 = helmholtz_free_energy_alpha(temperature, volume)

    Q_0 = 2.0 * molar_fractions[1] - 1.0
    Q_1 = 2.0 * molar_fractions[2] - 1.0

    Q_0sqr = Q_0 * Q_0
    Q_1sqr = Q_1 * Q_1
    Q_1qrtc = Q_1sqr * Q_1sqr
    Q_1sxtc = Q_1qrtc * Q_1sqr

    a = -0.1
    c_20 = 1457.881403240591 / (1.0 + 6.0 / a + 1.0 / (a * a))
    c_30 = 765.7558885708155 / (1.0 + 15.0 / a + 15.0 / (a * a) + 1.0 / (a * a * a))

    c_00 = F_00 - F_11
    c_10 = (
        F_11
        - F_00
        - (1.0 + 6.0 / a + 1.0 / (a * a)) * c_20
        - (1.0 + 15.0 / a + 15.0 / (a * a) + 1.0 / (a * a * a)) * c_30
    ) * (a / (a + 1.0))

    F_xs = (
        (c_00 + c_10 * Q_0sqr + c_20 * Q_1qrtc + c_30 * Q_1sxtc)
        + (c_10 + 6.0 * c_20 * Q_0sqr + 15.0 * c_30 * Q_1qrtc) * Q_1sqr / a
        + (c_20 + 15.0 * c_30 * Q_0sqr) * np.power(Q_1sqr / a, 2.0)
        + c_30 * np.power(Q_1sqr / a, 3.0)
    )

    return n_moles * F_xs


temperatures = np.linspace(200.0, 1600.0, 9)
Qs = np.linspace(0.0, 1.0, 101)

Fs = np.empty((9, 101))
F2s = np.empty((9, 101))
F3s = np.empty((9, 101))
Fos = np.empty((9, 101))

for i, T in enumerate(temperatures):
    print(i)
    for j, Q in enumerate(Qs):
        Q_0 = Q
        Q_1 = 0.0

        Fs[i, j] = quartz_helmholtz_function(
            qtz_alpha.params["V_0"],
            T,
            np.array([-0.5 * (Q_0 + Q_1), 0.5 * (Q_0 + 1.0), 0.5 * (Q_1 + 1.0)]),
        )
        Q_0 = 0.0
        Q_1 = 1.0 - Q

        F2s[i, j] = quartz_helmholtz_function(
            qtz_alpha.params["V_0"],
            T,
            np.array([-0.5 * (Q_0 + Q_1), 0.5 * (Q_0 + 1.0), 0.5 * (Q_1 + 1.0)]),
        )
        Q_0 = Q
        Q_1 = 1.0 - Q

        F3s[i, j] = quartz_helmholtz_function(
            qtz_alpha.params["V_0"],
            T,
            np.array([-0.5 * (Q_0 + Q_1), 0.5 * (Q_0 + 1.0), 0.5 * (Q_1 + 1.0)]),
        )

        Fos[i, j] = F_qtz(
            qtz_alpha.params["V_0"],
            T,
            np.array([0.5 * (Q + 1.0), 1.0 - 0.5 * (Q + 1.0)]),
        )


def helmholtz_at_VT(volume, temperature):
    def helmholtz(Q):
        molar_amounts = np.array(
            [-0.5 * (Q[0] + Q[1]), 0.5 * (Q[0] + 1.0), 0.5 * (Q[1] + 1.0)]
        )
        F = quartz_helmholtz_function(volume, temperature, molar_amounts)
        return F

    return helmholtz


V = qtz_alpha.params["V_0"]
T = 298.15
print(minimize(helmholtz_at_VT(V, T), [1.0, 1.0e-5]))


fig = plt.figure()
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
for i, T in enumerate(temperatures):
    ax[0].plot(Qs, Fs[i])
    ax[1].plot(Qs, F2s[i])
    ax[2].plot(Qs, F3s[i])
    ax[3].plot(Qs, Fos[i])
plt.show()
exit()

Q_0s = np.linspace(0.0, 1.0, 101)
Q_1s = np.linspace(0.0, 1.0, 101)

Fs = np.empty((101, 101))

V = qtz_alpha.params["V_0"]
T = 1098.15
for i, Q_0 in enumerate(Q_0s):
    print(i)
    for j, Q_1 in enumerate(Q_1s):

        Fs[i, j] = quartz_helmholtz_function(
            qtz_alpha.params["V_0"],
            298.15,
            np.array([-0.5 * (Q_0 + Q_1), 0.5 * (Q_0 + 1.0), 0.5 * (Q_1 + 1.0)]),
        )

plt.imshow(Fs)
plt.show()
