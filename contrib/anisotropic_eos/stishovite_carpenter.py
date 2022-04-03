import matplotlib.pyplot as plt
import numpy as np
from burnman.minerals.SLB_2011 import stishovite
from burnman.minerals.HP_2011_ds62 import stv as stishovite_HP
from scipy.integrate import cumulative_trapezoid


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


labels = ['11', '22', '33', '12', '13', '23', '44', '55', '66']


def Sij(cs):
    M = np.array([[cs[0], cs[3], cs[4], 0., 0., 0.],
                  [cs[3], cs[1], cs[5], 0., 0., 0.],
                  [cs[4], cs[5], cs[2], 0., 0., 0.],
                  [0., 0., 0., cs[6], 0., 0.],
                  [0., 0., 0., 0., cs[7], 0.],
                  [0., 0., 0., 0., 0., cs[8]]])
    im = np.linalg.inv(M)
    return [im[0, 0], im[1, 1], im[2, 2],
            im[0, 1], im[0, 2], im[1, 2],
            im[3, 3], im[4, 4], im[5, 5]]


pressures = np.linspace(0., 120., 1001)
Cijs = [Cij(P) for P in pressures]
Cijs = np.array(Cijs)

Sijs = [Sij(C[6:]) for C in Cijs]
Sijs = np.array(Sijs)


Cij0s = Cij0(pressures).T
Sij0s = [Sij(C) for C in Cij0s]
Sij0s = np.array(Sij0s)

fig = plt.figure(figsize=(12, 8))
ax = [fig.add_subplot(2, 3, i) for i in range(1, 7)]
ax[1].plot(pressures, Cijs[:, 0], label='Q')
ax[1].plot(pressures, Cijs[:, 1], label='$\\chi$')
ax[1].legend()

for i in range(6, len(Cijs[0])):
    ln, = ax[2].plot(pressures, Cijs[:, i])
    ax[2].plot(pressures, Cij0s[:, i-6], linestyle=':',
               color=ln.get_color())

for i in range(2, 5):
    ax[3].plot(pressures, Cijs[:, i])

beta0s = (Sij0s[:, 0] + Sij0s[:, 1] + Sij0s[:, 2]
          + 2.*(Sij0s[:, 3] + Sij0s[:, 4] + Sij0s[:, 5]))
betas = (Sijs[:, 0] + Sijs[:, 1] + Sijs[:, 2]
         + 2.*(Sijs[:, 3] + Sijs[:, 4] + Sijs[:, 5]))

KTV0s = (Cij0s[:, 0] + Cij0s[:, 1] + Cij0s[:, 2]
         + 2.*(Cij0s[:, 3] + Cij0s[:, 4] + Cij0s[:, 5]))/9.
KTVs = (Cijs[:, 6] + Cijs[:, 7] + Cijs[:, 8]
        + 2.*(Cijs[:, 9] + Cijs[:, 10] + Cijs[:, 11]))/9.

stv = stishovite()
stv_HP = stishovite_HP()

volumes, KTs = stv.evaluate(['V', 'K_T'], pressures*1.e9, pressures*0. + 300.)
volumes_HP, KTs_HP = stv_HP.evaluate(['V', 'K_T'],
                                     pressures*1.e9, pressures*0. + 300.)
f = np.log(volumes / stv.params['V_0'])


test = [837., 837., 964., 825., 263., 263., 336., 336., 463.]
Sij_test = Sij(test)
beta_test = (Sij_test[0] + Sij_test[1] + Sij_test[2]
             + 2.*(Sij_test[3] + Sij_test[4] + Sij_test[5]))


for i in range(len(Sij0s[0])):
    ln, = ax[4].plot(pressures, Sij0s[:, i]/beta0s, linestyle=':')
    ax[4].plot(pressures, Sijs[:, i]/betas,
               color=ln.get_color(), label=labels[i])
    ax[4].scatter([52.], [Sij_test[i]/beta_test], color=ln.get_color())

ax[4].plot(pressures, (Sijs[:, 0] + Sijs[:, 3])/betas,
           label='11 + 12')
ax[4].plot(pressures, (Sijs[:, 1] + Sijs[:, 3])/betas,
           label='22 + 12')

ax[4].legend(loc='lower right')
ax[4].set_ylim(-2., 2.)
ax[5].plot(pressures, (beta0s/betas - 1.), label='Reuss')
ax[5].plot(pressures, (KTVs/KTV0s - 1.), label='Voigt')
# ax[5].plot(pressures, 1./beta0s, label='Reuss (base)')
# ax[5].plot(pressures, 1./betas, label='Reuss')
# ax[5].plot(pressures, KTV0s, label='Voigt (base)')
# ax[5].plot(pressures, KTVs, label='Voigt')
# ax[5].plot(pressures, 1./beta0s, linestyle=':', label='Bare')
# ax[5].plot(pressures, KTs/1.e9, label='SLB2011')
# ax[5].plot(pressures, KTs_HP/1.e9, label='HP2011')
ax[5].set_ylabel('(K$_T$ (Landau) / K$_T$ (bare)) - 1')
ax[5].legend()

lnV = -cumulative_trapezoid(betas*1.e-9, pressures*1.e9, initial=0.)
V = np.exp(lnV) * stv_HP.params['V_0']
gibbs = cumulative_trapezoid(V, pressures*1.e9, initial=0.)

lnV = -cumulative_trapezoid(beta0s*1.e-9, pressures*1.e9, initial=0.)
V = np.exp(lnV) * stv_HP.params['V_0']
gibbs0 = cumulative_trapezoid(V, pressures*1.e9, initial=0.)

ax[0].plot(pressures, Cijs[:, 5]*1.e9*stv.params['V_0'],
           label='by Equation 1')
ax[0].plot(pressures, (gibbs - gibbs0)*1.e3, linestyle=':',
           label='by integration of $\\beta_T$, mult.d by 1000')
ax[0].legend()

ax[0].set_ylabel('$\\Delta$ Gibbs')
ax[1].set_ylabel('$Q, \\chi$')

ax[1].set_xlabel('Pressure (GPa)')
ax[3].set_xlabel('Pressure (GPa)')
ax[2].set_ylabel('Cijs')
ax[3].set_ylabel('es')

for i in range(6):
    ax[i].set_xlim(int(pressures[0]), int(pressures[-1]))
    # ax[i].set_xlim(48., 50.)

for i in range(1, 3):
    ax[i].set_ylim(0.,)

fig.set_tight_layout(True)

plt.show()
