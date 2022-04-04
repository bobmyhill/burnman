import numpy as np


def Cij0(pressure):
    C110 = 578 + 5.38 * pressure
    C330 = 776 + 4.94 * pressure
    C120 = 86 + 5.38 * pressure
    C130 = 191 + 2.72 * pressure
    C440 = 252 + 1.88 * pressure
    C660 = 323 + 3.10 * pressure

    return np.array([C110, C110, C330, C120, C130, C130, C440, C440, C660])


def Cij(pressure):
    C110, _, C330, C120, C130, _, C440, _, C660 = Cij0(pressure)

    Pcstar = 49.
    Pc = 99.7
    l1 = -8.
    l2 = 24.62
    l3 = 17.
    l4 = 20.
    l6 = 20.
    a = -0.04856
    b = 10.94

    bstar = b - 2.*((l3*l3*(C110 + C120) + 2.*l1*l1*C330 - 4.*l1*l3*C130)
                    / ((C110 + C120)*C330 - 2.*C130*C130))

    if pressure < Pcstar:
        invchi = (a*(pressure - Pc))
        Q = 0
    else:

        Q = np.sqrt(a/bstar*(Pcstar - pressure))
        invchi = 2.*a*b/bstar*(Pcstar - pressure) + a*(Pcstar - Pc)

    # chi goes to 0 a long way above and below the transition
    chi = 1./invchi

    if pressure < Pcstar:
        C11 = C110 - l2*l2*chi
        C22 = C11
        C33 = C330
        C12 = C120 + l2*l2*chi
        C13 = C130
        C23 = C130
        C44 = C440
        C55 = C440
        C66 = C660

        e1 = 0.
        e2 = 0.
        e3 = 0.

    else:
        C11 = C110 - (4.*l1*l1*Q*Q + 4.*l1*l2*Q)*chi - l2*l2*chi
        C22 = C110 - (4.*l1*l1*Q*Q - 4.*l1*l2*Q)*chi - l2*l2*chi
        C33 = C330 - 4.*l3*l3*Q*Q*chi
        C12 = C120 - (4.*l1*l1*Q*Q)*chi + l2*l2*chi
        C13 = C130 - (4.*l1*l3*Q*Q + 2.*l2*l3*Q)*chi
        C23 = C130 - (4.*l1*l3*Q*Q - 2.*l2*l3*Q)*chi
        C44 = C440 + 2*l4*Q
        C55 = C440 - 2*l4*Q
        C66 = C660 + 2*l6*Q*Q
        e1 = - l2 / (0.5*(C110 - C120))*Q/2. - l1/(0.5*(C110 - C120))*Q*Q/2.
        e2 = l2 / (0.5*(C110 - C120))*Q/2. - l1/(0.5*(C110 - C120))*Q*Q/2.
        e3 = - l3/C330*Q*Q

    G = (1./2.*a*(pressure - Pc)*Q*Q
         + 1./4.*b*Q*Q*Q*Q
         + l1*(e1 + e2)*Q*Q
         + l2*(e1 - e2)*Q
         + l3*e3*Q*Q
         + 1./4.*(C110 + C120)*np.power(e1+e2, 2.)
         + 1./4.*(C110 - C120)*np.power(e1-e2, 2.)
         + C130*(e1 + e2)*e3
         + 1./2.*C330*e3*e3)
    return {'Q': Q,
            'chi': chi,
            'e': [e1, e2, e3],
            'gibbs': G,
            'C': np.array([[C11, C12, C13, 0., 0., 0.],
                           [C12, C22, C23, 0., 0., 0.],
                           [C13, C23, C33, 0., 0., 0.],
                           [0., 0., 0., C44, 0., 0.],
                           [0., 0., 0., 0., C55, 0.],
                           [0., 0., 0., 0., 0., C66]])}
