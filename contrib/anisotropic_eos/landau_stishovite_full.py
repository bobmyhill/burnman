import matplotlib.pyplot as plt
import numpy as np
from burnman.minerals.SLB_2011 import stishovite
from scipy.optimize import minimize
import matplotlib.image as mpimg
from carpenter_functions import Cij as Cij_orig

stv = stishovite()

fig4 = mpimg.imread('figures/Carpenter_2000_Figure_4.png')
fig6 = mpimg.imread('figures/Carpenter_2000_Figure_6.png')


def Cij0(pressure):
    C110 = 578 + 5.38 * pressure
    C330 = 776 + 4.94 * pressure
    C120 = 86 + 5.38 * pressure
    C130 = 191 + 2.72 * pressure
    C440 = 252 + 1.88 * pressure
    C660 = 323 + 3.10 * pressure

    return np.array([[C110, C120, C130, 0., 0., 0.],
                     [C120, C110, C130, 0., 0., 0.],
                     [C130, C130, C330, 0., 0., 0.],
                     [0., 0., 0., C440, 0., 0.],
                     [0., 0., 0., 0., C440, 0.],
                     [0., 0., 0., 0., 0., C660]])


KT0 = 1./np.sum(np.linalg.inv(Cij0(0))[:3, :3])
KT1 = 1./np.sum(np.linalg.inv(Cij0(1))[:3, :3])


Pcstar = 49.
Pc = 99.7
l1 = -8.
l2 = 24.62
l3 = 17.
l4 = 20.
l6 = 20.
b = 10.94
a = -0.04856

print('Approx correction for K and Kprime...')
stv.params['K_0'] = KT0*1.e9
stv.params['Kprime_0'] = KT1-KT0

print('Conversion of Landau parameters to volume equivalents...')
stv.set_state(Pc * 1.e9, 300.)
V1 = stv.V
stv.set_state(Pcstar * 1.e9, 300.)
V2 = stv.V

Vc = V1
a = a*(Pc-Pcstar) / (V1 - V2)


def fn_helmholtz(args, V):
    Q, e1, e2, e3 = args
    # the epsilons are defined relative to the base state

    P_at_V = stv.method.pressure(300., V, stv.params)

    # C_base won't be fully consistent with the volume
    # until we have an isotropic baseline eos
    C_base = Cij0(P_at_V/1.e9)

    epsilon_voigt = np.array([e1, e2, e3, 0., 0., 0.])

    F_el = 1./2.*np.einsum('i, ij, j->', epsilon_voigt, C_base, epsilon_voigt)

    helmholtz = (1./2.*a*(V - Vc)*Q*Q
                 + 1./4.*b*Q*Q*Q*Q
                 + l1*(epsilon_voigt[0] + epsilon_voigt[1])*Q*Q
                 + l2*(epsilon_voigt[0] - epsilon_voigt[1])*Q
                 + l3*epsilon_voigt[2]*Q*Q
                 + F_el)*V*1.e9
    return helmholtz


def fn_helmholtz2(args, P):
    Q, e1, e2, e3, f = args
    # the epsilons are defined relative to the base state
    V = np.exp(f)*stv.params['V_0']

    # C_base won't be fully consistent with the volume
    # until we have an isotropic baseline eos
    C_base = Cij0(P/1.e9)

    epsilon_voigt = np.array([e1, e2, e3, 0., 0., 0.])

    F_el = 1./2.*np.einsum('i, ij, j->', epsilon_voigt, C_base, epsilon_voigt)

    helmholtz = (1./2.*a*(V - Vc)*Q*Q
                 + 1./4.*b*Q*Q*Q*Q
                 + l1*(epsilon_voigt[0] + epsilon_voigt[1])*Q*Q
                 + l2*(epsilon_voigt[0] - epsilon_voigt[1])*Q
                 + l3*epsilon_voigt[2]*Q*Q
                 + F_el)*V*1.e9
    return helmholtz


def fn_delta_F(args):
    """
    Ensures that the volume in the deformed state is equal
    to the volume in the undeformed state
    """
    Q, e1, e2, e3, f = args
    F1 = e1 + 1.
    F2 = e2 + 1.
    F3 = e3 + 1.
    F = np.diag([F1, F2, F3])

    return np.linalg.det(F) - 1.


def fn_delta_F2(args):
    """
    Ensures that the volume in the deformed state is equal
    to the volume in the undeformed state
    """
    Q, e1, e2, e3 = args
    F1 = e1 + 1.
    F2 = e2 + 1.
    F3 = e3 + 1.
    F = np.diag([F1, F2, F3])

    return np.linalg.det(F) - 1.


"""
def delta_P(P):
    def fn_delta_P(args):
        Q, e1, e2, e3, f = args
        V = np.exp(f)*stv.params['V_0']
        dV = 1.e-8
        sol1 = minimize(fn_helmholtz, args[:4], args=(V-dV/2.),
                        constraints=[{'type': 'eq', 'fun': fn_delta_F2}])
        sol2 = minimize(fn_helmholtz, args[:4], args=(V+dV/2.),
                        constraints=[{'type': 'eq', 'fun': fn_delta_F2}])
        P_at_V = stv.method.pressure(300., V, stv.params)

        return P_at_V + (sol2.fun - sol1.fun)/dV - P

    return fn_delta_P
"""

print('Warning: C_base will not be fully consistent with the volume '
      'until we have an isotropic baseline eos')


def fn_helmholtz3(args, epsilon_voigt, V):
    Q = args[0]

    P_at_V = stv.method.pressure(300., V, stv.params)

    # C_base won't be fully consistent with the volume
    # until we have an isotropic baseline eos
    C_base = Cij0(P_at_V/1.e9)

    F_el = 1./2.*np.einsum('i, ij, j->', epsilon_voigt, C_base, epsilon_voigt)

    helmholtz = (1./2.*a*(V - Vc)*Q*Q
                 + 1./4.*b*Q*Q*Q*Q
                 + l1*(epsilon_voigt[0] + epsilon_voigt[1])*Q*Q
                 + l2*(epsilon_voigt[0] - epsilon_voigt[1])*Q
                 + l3*epsilon_voigt[2]*Q*Q
                 + l4*(epsilon_voigt[3]*epsilon_voigt[3]
                       - epsilon_voigt[4]*epsilon_voigt[4])*Q
                 + l6*epsilon_voigt[5]*epsilon_voigt[5]*Q*Q
                 + F_el)*V*1.e9

    return helmholtz


print('Warning: Maybe not quite normalized correctly yet.')


def normalize_Ve(volume, eps):
    F = np.array([[eps[0]+1., eps[3], eps[4]],
                  [eps[3], eps[1]+1., eps[5]],
                  [eps[4], eps[5], eps[2]+1.]])

    VoverV0 = np.linalg.det(F)
    # F /= np.cbrt(VoverV0) ???
    return (volume*VoverV0,
            np.array([F[0, 0]-1., F[1, 1]-1., F[2, 2]-1.,
                      F[0, 1], F[0, 2], F[1, 2]]))


def min_helmholtz3(volume, epsilon_voigt, Q_guess):
    v, e = normalize_Ve(volume, epsilon_voigt)
    sol = minimize(fn_helmholtz3, [Q_guess], args=(e, v))
    if sol.success:
        return sol.fun, sol.x
    else:
        # print('Bad')
        return sol.fun, sol.x


def get_Cijs(volume, epsilon_voigt, Q_guess):
    de = 1.e-5
    C = np.empty((6, 6))
    for i in range(6):
        for j in range(6):
            if i == j:
                e = np.copy(epsilon_voigt)
                A1, Q1 = min_helmholtz3(volume, e, Q_guess)
                e[i] -= de
                A0, Q0 = min_helmholtz3(volume, e, Q_guess)
                e[i] += 2.*de
                A2, Q2 = min_helmholtz3(volume, e, Q_guess)

                C[i][j] = (A2 + A0 - 2.*A1)/de/de

            else:
                e = np.copy(epsilon_voigt)
                e[i] -= de/2.
                e[j] -= de/2.
                A0, Q0 = min_helmholtz3(volume, e, Q_guess)
                e[i] += de
                A1, Q1 = min_helmholtz3(volume, e, Q_guess)
                e[i] -= de
                e[j] += de
                A2, Q2 = min_helmholtz3(volume, e, Q_guess)
                e[i] += de
                A3, Q3 = min_helmholtz3(volume, e, Q_guess)

                C[i][j] = ((A3 - A2) - (A1 - A0))/de/de
                C[j][i] = C[i][j]
    return C


pressures = np.linspace(1.e5, 125.e9, 126)
new_pressures = np.empty_like(pressures)
params = np.empty((len(pressures), 4))
Cijs = np.empty((len(pressures), 6, 6))
gibbs = np.empty_like(pressures)
volumes = np.empty_like(pressures)

for i, P in enumerate(pressures):
    print(i)
    stv.set_state(P, 300.)
    # sol = minimize(fn_helmholtz2, [0.1, 0., 0., 0., 0.], args=(P),
    #               constraints=[{'type': 'eq', 'fun': fn_delta_F},
    #                            {'type': 'eq', 'fun': delta_P(P)}])

    dV = 1.e-8
    sol1 = minimize(fn_helmholtz, [0.1, 0., 0., 0.], args=(stv.V-dV/2.),
                    constraints=[{'type': 'eq', 'fun': fn_delta_F2}])
    sol2 = minimize(fn_helmholtz, [0.1, 0., 0., 0.], args=(stv.V+dV/2.),
                    constraints=[{'type': 'eq', 'fun': fn_delta_F2}])

    new_pressures[i] = P - (sol2.fun - sol1.fun)/dV
    params[i] = (sol1.x + sol2.x)/2.

    Q = params[i, 0]
    epsilon_voigt = np.zeros(6)
    epsilon_voigt[:3] = params[i, 1:]
    Cijs[i] = get_Cijs(stv.V, epsilon_voigt, Q+0.1)/stv.V
    gibbs[i] = (stv.helmholtz
                + (sol1.fun + sol2.fun)/2.
                + new_pressures[i] * stv.V)
    volumes[i] = stv.V


fig = plt.figure(figsize=(12, 8))
ax = [fig.add_subplot(2, 3, i) for i in range(1, 7)]

# ax[0].imshow(fig6, extent=[0.0, 120.0, -3600., 0.], aspect='auto')
# ax[4].imshow(fig4, extent=[0.0, 120.0, 0., 1300.], aspect='auto')

gibbs_base, volumes_base = stv.evaluate(['gibbs', 'V'],
                                        new_pressures,
                                        pressures*0. + 300.)


Carpenter_gibbs = np.array([Cij_orig(P/1.e9)['gibbs']
                            * 1.e9 * volumes_base[i]
                           for i, P in enumerate(new_pressures)])

ln, = ax[0].plot(new_pressures/1.e9, gibbs - gibbs_base)
ax[0].plot(new_pressures/1.e9, Carpenter_gibbs,
           linestyle=':', color=ln.get_color())
ax[1].plot(new_pressures/1.e9, volumes - volumes_base)

ax[2].plot(new_pressures/1.e9, params[:, 0])

ax[3].plot(new_pressures/1.e9, params[:, 1], label='$\\varepsilon_1$')
ax[3].plot(new_pressures/1.e9, params[:, 2], label='$\\varepsilon_2$')
ax[3].plot(new_pressures/1.e9, params[:, 3], label='$\\varepsilon_3$')
ax[3].legend()
Carpenter_Cijs = np.array([Cij_orig(P/1.e9)['C']
                           for P in new_pressures])

Carpenter_betas = np.array([np.sum(np.linalg.inv(Cij_orig(P/1.e9)['C'])[:3,
                                                                        :3])
                           for P in new_pressures])

for (i, j) in [(1, 1),
               (2, 2),
               (3, 3),
               (4, 4),
               (5, 5),
               (6, 6),
               (1, 2),
               (1, 3),
               (2, 3)]:
    ln, = ax[4].plot(new_pressures/1.e9, Cijs[:, i-1, j-1]/1.e9,
                     label=f'{i}{j}')
    ax[4].plot(new_pressures/1.e9, Carpenter_Cijs[:, i-1, j-1],
               linestyle=':', color=ln.get_color())

ax[4].legend()
betas = np.empty_like(new_pressures)
for i in range(len(Cijs)):
    Sijs = np.linalg.inv(Cijs[i])
    betas[i] = np.sum(Sijs[:3, :3])

ax[5].plot(new_pressures/1.e9, 1./betas/1.e9, label='from Cijs')
ax[5].plot(new_pressures/1.e9, 1./Carpenter_betas,
           linestyle=':', label='from Carpenter Cijs')
ax[5].plot(new_pressures/1.e9, -volumes*np.gradient(new_pressures,
                                                    volumes,
                                                    edge_order=2)/1.e9,
           linestyle='--', label='from volume derivative')
ax[5].legend()

ax[0].set_ylabel('$\\Delta \\mathcal{G}$')
ax[1].set_ylabel('$V$')
ax[2].set_ylabel('$Q$')
ax[3].set_ylabel('$\\varepsilon$')
ax[4].set_ylabel('$C_{ij}$')
ax[5].set_ylabel('$K_T$')
ax[3].set_xlabel('Pressure (GPa)')
ax[4].set_xlabel('Pressure (GPa)')
ax[5].set_xlabel('Pressure (GPa)')
fig.set_tight_layout(True)
fig.savefig('Carpenter_2000_comparison.pdf')
plt.show()
