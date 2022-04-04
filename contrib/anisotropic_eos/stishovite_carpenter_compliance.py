import matplotlib.pyplot as plt
import numpy as np
from burnman.minerals.SLB_2011 import stishovite
from scipy.optimize import minimize

Pcstar = 49.
Pc = 99.7
l1 = -8.
l2 = 24.62
l3 = 17.
l4 = 20.
l6 = 20.
a = -0.04856
b = 10.94
    

def Cij0(pressure):
    C110 = 578 + 5.38 * pressure
    C330 = 776 + 4.94 * pressure
    C120 = 86 + 5.38 * pressure
    C130 = 191 + 2.72 * pressure
    C440 = 252 + 1.88 * pressure
    C660 = 323 + 3.10 * pressure

    return np.array([C110, C110, C330, C120, C130, C130, C440, C440, C660])


def Sij0(pressure):
    cs = Cij0(pressure)
    M = np.array([[cs[0], cs[3], cs[4], 0., 0., 0.],
                  [cs[3], cs[1], cs[5], 0., 0., 0.],
                  [cs[4], cs[5], cs[2], 0., 0., 0.],
                  [0., 0., 0., cs[6], 0., 0.],
                  [0., 0., 0., 0., cs[7], 0.],
                  [0., 0., 0., 0., 0., cs[8]]])
    im = np.linalg.inv(M)
    return im
# return [im[0, 0], im[1, 1], im[2, 2],
#            im[0, 1], im[0, 2], im[1, 2],
#            im[3, 3], im[4, 4], im[5, 5]]


def fn_gibbs(args, pressure):
    Q, sigma1, sigma2, sigma3 = args
    # sigma3 = 0. - sigma1 - sigma2
    sigma = np.array([sigma1, sigma2, sigma3, 0., 0., 0.])
    # print(np.sum(sigma))
    S = Sij0(pressure)
    e = np.einsum('ij, j->i', S, sigma)
    e1, e2, e3 = e[:3]
    G_el = 1./2.*np.einsum('i, i->', e, sigma)
    G = (1./2.*a*(pressure - Pc)*Q*Q
         + 1./4.*b*Q*Q*Q*Q
         + l1*(e1 + e2)*Q*Q
         + l2*(e1 - e2)*Q
         + l3*e3*Q*Q
         + G_el)

    return G


def minG(pressure, guess=[0.2, 0., 0., 0.]):
    sol = minimize(fn_gibbs, guess, args=(pressure))
    if sol.success:
        return sol.fun, sol.x
    else:
        raise Exception('Gibbs minimizer was unsuccessful')


pressures = np.linspace(100., 0., 201)
stv = stishovite()

gibbs = []
params = []
guess = [0.2, 0., 0., 0.]
for P in pressures:
    sol = minG(P, guess)
    gibbs.append(sol[0]*1.e9*stv.params['V_0'])
    params.append(sol[1])
    guess = params[-1]

gibbs = np.array(gibbs)
params = np.array(params)

# plt.plot(pressures, gibbs)

fig = plt.figure(figsize=(12, 8))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
ax[0].plot(pressures, gibbs)
ax[1].plot(pressures, params[:, 0])
ax[2].plot(pressures, params[:, 1])
ax[2].plot(pressures, params[:, 2])
ax[2].plot(pressures, params[:, 3])


def fnG3(arg, sigma, pressure):

    Q = arg[0]

    S = Sij0(pressure)
    e = np.einsum('ij, j->i', S, sigma)
    e1, e2, e3, e4, e5, e6 = e
    deltaP = sigma[0] + sigma[1] + sigma[2]
    G_el = 1./2.*np.einsum('i, i->', e, sigma)
    G = (1./2.*a*(pressure - deltaP - Pc)*Q*Q
         + 1./4.*b*Q*Q*Q*Q
         + l1*(e1 + e2)*Q*Q
         + l2*(e1 - e2)*Q
         + l3*e3*Q*Q
         + l4*(e4*e4 - e5*e5)*Q
         + l6*e6*e6*Q*Q
         + G_el)

    return G


def minG3(pressure, sigma, Q_guess):
    sol = minimize(fnG3, [Q_guess], args=(sigma, pressure))
    if sol.success:
        return sol.fun, sol.x
    else:
        print('Bad')
        return sol.fun, sol.x


ds = 1.e-5
Snew = np.empty((len(pressures), 6, 6))
Cnew = np.empty_like(Snew)

guess = [0.2, 0., 0., 0.]
for k, P in enumerate(pressures):
    print(k)
    guess = minG(P, guess)[1]
    for i in range(6):
        for j in range(6):
            if i == j:
                Q_guess, s1, s2, s3 = guess
                s = [s1, s2, s3, 0., 0., 0.]
                G1, Q1 = minG3(P, s, Q_guess)
                s[i] -= ds
                G0, Q0 = minG3(P, s, Q_guess)
                s[i] += 2.*ds
                G2, Q2 = minG3(P, s, Q_guess)

                Snew[k][i][j] = (G2 + G0 - 2.*G1)/ds/ds
            else:
                Q_guess, s1, s2, s3 = guess
                s = [s1, s2, s3, 0., 0., 0.]
                s[i] -= ds/2.
                s[j] -= ds/2.
                G0, Q0 = minG3(P, s, Q_guess)
                s[i] += ds
                G1, Q1 = minG3(P, s, Q_guess)
                s[i] -= ds
                s[j] += ds
                G2, Q2 = minG3(P, s, Q_guess)
                s[i] += ds
                G3, Q3 = minG3(P, s, Q_guess)

                Snew[k][i][j] = ((G3 - G2) - (G1 - G0))/ds/ds
                Snew[k][j][i] = Snew[k][i][j]

    Cnew[k] = np.linalg.inv(Snew[k])


for i in range(6):
    for j in range(i, 6):
        ax[3].plot(pressures, Snew[:, i, j])

plt.show()

exit()


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
    return [Q,
            chi,
            e1,
            e2,
            e3,
            G,
            C11,
            C22,
            C33,
            C12,
            C13,
            C23,
            C44,
            C55,
            C66]


Pcstar = 49.
Pc = 99.7
l1 = -8.
l2 = 24.62
l3 = 17.
l4 = 20.
l6 = 20.
a = -0.04856
b = 10.94


def fnG(args, pressure):
    Q, e1, e2, e3 = args
    C110, _, C330, C120, C130, _, C440, _, C660 = Cij0(pressure)

    G = (1./2.*a*(pressure - Pc)*Q*Q
         + 1./4.*b*Q*Q*Q*Q
         + l1*(e1 + e2)*Q*Q
         + l2*(e1 - e2)*Q
         + l3*e3*Q*Q
         + 1./4.*(C110 + C120)*np.power(e1+e2, 2.)
         + 1./4.*(C110 - C120)*np.power(e1-e2, 2.)
         + C130*(e1 + e2)*e3
         + 1./2.*C330*e3*e3)
    return G


def minG(pressure):
    sol = minimize(fnG, [0.01, 0., 0., 0.], args=(pressure))
    if sol.success:
        return sol.fun, sol.x
    else:
        raise Exception('Gibbs minimizer was unsuccessful')


def fnG2(args, Q, pressure):
    e1, e2, e3 = args
    C110, _, C330, C120, C130, _, C440, _, C660 = Cij0(pressure)

    G = (1./2.*a*(pressure - Pc)*Q*Q
         + 1./4.*b*Q*Q*Q*Q
         + l1*(e1 + e2)*Q*Q
         + l2*(e1 - e2)*Q
         + l3*e3*Q*Q
         + 1./4.*(C110 + C120)*np.power(e1+e2, 2.)
         + 1./4.*(C110 - C120)*np.power(e1-e2, 2.)
         + C130*(e1 + e2)*e3
         + 1./2.*C330*e3*e3)
    return G


def minG2(pressure, Q, e_guess):
    sol = minimize(fnG2, e_guess, args=(Q, pressure))
    if sol.success:
        return sol.fun, sol.x
    else:
        print('Bad')
        return sol.fun, sol.x


def fnG3(arg, e, pressure):

    Q = arg[0]
    e1, e2, e3, e4, e5, e6 = e
    C110, _, C330, C120, C130, _, C440, _, C660 = Cij0(pressure)

    G = (1./2.*a*(pressure - Pc)*Q*Q
         + 1./4.*b*Q*Q*Q*Q
         + l1*(e1 + e2)*Q*Q
         + l2*(e1 - e2)*Q
         + l3*e3*Q*Q
         + l4*(e4*e4 - e5*e5)*Q
         + l6*e6*e6*Q*Q
         + 1./4.*(C110 + C120)*np.power(e1+e2, 2.)
         + 1./4.*(C110 - C120)*np.power(e1-e2, 2.)
         + C130*(e1 + e2)*e3
         + 1./2.*C330*e3*e3
         + 1./2.*C440*(e4*e4 + e5*e5)
         + 1./2.*C660*e6*e6)

    return G


def minG3(pressure, e, Q_guess):
    sol = minimize(fnG3, [Q_guess], args=(e, pressure))
    if sol.success:
        return sol.fun, sol.x
    else:
        print('Bad')
        return sol.fun, sol.x


pressures = np.linspace(0., 100., 101)
stv = stishovite()


gibbs = np.array([minG(P)[0]*1.e9*stv.params['V_0'] for P in pressures])
params = np.array([minG(P)[1] for P in pressures])

Q = params[:, 0]
e = params[:, 1:]

"""
dQ = 1.e-3
gibbs1 = np.array([minG2(pressures[i], Q[i]-dQ, e[i])[0]*1.e9*stv.params['V_0']
                   for i in range(len(pressures))])
gibbs2 = np.array([minG2(pressures[i], Q[i]+dQ, e[i])[0]*1.e9*stv.params['V_0']
                   for i in range(len(pressures))])

d2GdQ2 = (gibbs2 + gibbs1 - 2.*gibbs)/(dQ*dQ)/(1.e9*stv.params['V_0'])
"""

fig = plt.figure(figsize=(12, 8))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

ax[0].plot(pressures, gibbs)
ax[1].plot(pressures, Q)
for i in range(3):
    ax[2].plot(pressures, e[:, i])


C110, _, C330, C120, C130, _, C440, _, C660 = Cij0(pressures)
bstar = b - 2.*((l3*l3*(C110 + C120) + 2.*l1*l1*C330 - 4.*l1*l3*C130)
                / ((C110 + C120)*C330 - 2.*C130*C130))

invchi = [2.*a*b/bstar[i]*(Pcstar - P) + a*(Pcstar - Pc) if P > Pcstar
          else (a*(P - Pc)) for i, P in enumerate(pressures)]

Qs = np.array([np.sqrt(a/bstar[i]*(Pcstar - P)) if P > Pcstar
               else 0. for i, P in enumerate(pressures)])

ax[1].plot(pressures, Qs, linestyle=':')
# ax[3].plot(pressures, invchi, linestyle=':')

e1 = - l2 / (0.5*(C110 - C120))*Qs/2. - l1/(0.5*(C110 - C120))*Qs*Qs/2.
e2 = l2 / (0.5*(C110 - C120))*Qs/2. - l1/(0.5*(C110 - C120))*Qs*Qs/2.
e3 = - l3/C330*Qs*Qs

ax[2].plot(pressures, e1, linestyle=':')
ax[2].plot(pressures, e2, linestyle=':')
ax[2].plot(pressures, e3, linestyle=':')

Gs = (1./2.*a*(pressures - Pc)*Qs*Qs
      + 1./4.*b*Qs*Qs*Qs*Qs
      + l1*(e1 + e2)*Qs*Qs
      + l2*(e1 - e2)*Qs
      + l3*e3*Qs*Qs
      + 1./4.*(C110 + C120)*np.power(e1+e2, 2.)
      + 1./4.*(C110 - C120)*np.power(e1-e2, 2.)
      + C130*(e1 + e2)*e3
      + 1./2.*C330*e3*e3)*1.e9*stv.params['V_0']


de = 1.e-5
pressures = np.linspace(0., 100., 101)
C = np.empty((len(pressures), 6, 6))
for k, P in enumerate(pressures):
    print(k)
    for i in range(6):
        for j in range(6):
            if i == j:

                Q_guess, e1, e2, e3 = minG(P)[1]
                e = [e1, e2, e3, 0., 0., 0.]
                G1, Q1 = minG3(P, e, Q_guess)
                e[i] -= de
                G0, Q0 = minG3(P, e, Q_guess)
                e[i] += 2.*de
                G2, Q2 = minG3(P, e, Q_guess)

                C[k][i][j] = (G2 + G0 - 2.*G1)/de/de

            else:
                Q_guess, e1, e2, e3 = minG(P)[1]
                e = [e1, e2, e3, 0., 0., 0.]
                e[i] -= de/2.
                e[j] -= de/2.
                G0, Q0 = minG3(P, e, Q_guess)
                e[i] += de
                G1, Q1 = minG3(P, e, Q_guess)
                e[i] -= de
                e[j] += de
                G2, Q2 = minG3(P, e, Q_guess)
                e[i] += de
                G3, Q3 = minG3(P, e, Q_guess)

                C[k][i][j] = ((G3 - G2) - (G1 - G0))/de/de
                C[k][j][i] = C[k][i][j]


for i in range(6):
    for j in range(i, 6):
        ax[3].plot(pressures, C[:, i, j])


labels = ['11', '22', '33', '12', '13', '23', '44', '55', '66']

Cijs = [Cij(P) for P in pressures]
Cijs = np.array(Cijs)

Cij0s = Cij0(pressures).T

for i in range(6, len(Cijs[0])):
    ln, = plt.plot(pressures, Cijs[:, i], linestyle='--')
    ax[3].plot(pressures, Cij0s[:, i-6], linestyle=':', color=ln.get_color())

ax[0].plot(pressures, Gs, linestyle=':')

ax[0].set_ylabel('$\\Delta \\mathcal{G}$')
ax[1].set_ylabel('$Q$')
ax[2].set_ylabel('$\\varepsilon$')
ax[3].set_ylabel('$C_{ij}$')

plt.show()
