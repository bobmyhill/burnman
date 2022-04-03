import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, root
import matplotlib.image as mpimg
from carpenter_functions import Cij as Cij_orig
from stishovite_isotropic import make_phase
# from burnman.tools.eos import check_anisotropic_eos_consistency
# from burnman.minerals.SLB_2011 import stishovite
# from burnman.minerals.HP_2011_ds62 import stv as stishovite_HP


stv = make_phase()


fig4 = mpimg.imread('figures/Carpenter_2000_Figure_4.png')
fig6 = mpimg.imread('figures/Carpenter_2000_Figure_6.png')

Pcstar = 49.e9
Pc = 99.7e9
l1 = -8.e9
l2 = 24.62e9
l3 = 17.e9
l4 = 20.e9
l6 = 20.e9
b = 10.94e9
a = -0.04856

print('Conversion of Landau parameters to volume equivalents...')
stv.set_state(Pc, 300.)
V1 = stv.V
stv.set_state(Pcstar, 300.)
V2 = stv.V

Vc = V1
a = a*(Pc-Pcstar) / (V1 - V2) * V1


def fn_helmholtz(args, V):
    Q, e1, e2, e3 = args
    # the epsilons are defined relative to a modified base state
    # (to ensure that the stress is relaxed)
    P_orig = stv.pressure
    T_orig = stv.temperature
    
    
    F1 = e1 + 1.
    F2 = e2 + 1.
    F3 = e3 + 1.
    F = np.diag([F1, F2, F3])
    Vr = V/np.linalg.det(F)
    P_at_Vr = stv.method.pressure(T_orig, Vr, stv.params)
    stv.set_state(P_at_Vr, T_orig)
    C_base = stv.isothermal_stiffness_tensor

    epsilon_voigt = np.array([e1, e2, e3, 0., 0., 0.])

    F_el = 1./2.*np.einsum('i, ij, j->', epsilon_voigt, C_base, epsilon_voigt)
    helmholtz = ((1./2.*a*(Vr - Vc)*Q*Q + 1./4.*b*Vr*Q*Q*Q*Q)
                 + (l1*(epsilon_voigt[0] + epsilon_voigt[1])*Q*Q
                    + l2*(epsilon_voigt[0] - epsilon_voigt[1])*Q
                    + l3*epsilon_voigt[2]*Q*Q
                    + F_el)*V)
    stv.set_state(P_orig, T_orig)
    return helmholtz


def min_helmholtz2(V, epsilon_voigt, Q_guess):
    """
    Solve for dF/dQ = 0
    Return F
    """
    P_orig = stv.pressure
    T_orig = stv.temperature

    F = np.array([[epsilon_voigt[0] + 1., epsilon_voigt[3], epsilon_voigt[4]],
                  [epsilon_voigt[3], epsilon_voigt[1]+1., epsilon_voigt[5]],
                  [epsilon_voigt[4], epsilon_voigt[5], epsilon_voigt[2]+1.]])
    Vr = V/np.linalg.det(F)

    P_at_Vr = stv.method.pressure(T_orig, Vr, stv.params)
    stv.set_state(P_at_Vr, T_orig)
    C_base = stv.isothermal_stiffness_tensor

    F_el = 1./2.*np.einsum('i, ij, j->', epsilon_voigt, C_base, epsilon_voigt)

    def dFdQ(Q):
        return ((a*(Vr - Vc)*Q + b*Vr*Q*Q*Q)/V
                + (2.*l1*(epsilon_voigt[0] + epsilon_voigt[1])*Q
                   + l2*(epsilon_voigt[0] - epsilon_voigt[1])
                   + 2.*l3*epsilon_voigt[2]*Q
                   + l4*(epsilon_voigt[3]*epsilon_voigt[3]
                         - epsilon_voigt[4]*epsilon_voigt[4])
                   + 2.*l6*epsilon_voigt[5]*epsilon_voigt[5]*Q))

    Q = root(dFdQ, [Q_guess]).x[0]


    helmholtz = ((1./2.*a*(Vr - Vc)*Q*Q + 1./4.*b*Vr*Q*Q*Q*Q)
                 + (l1*(epsilon_voigt[0] + epsilon_voigt[1])*Q*Q
                    + l2*(epsilon_voigt[0] - epsilon_voigt[1])*Q
                    + l3*epsilon_voigt[2]*Q*Q
                    + l4*(epsilon_voigt[3]*epsilon_voigt[3]
                          - epsilon_voigt[4]*epsilon_voigt[4])*Q
                    + l6*epsilon_voigt[5]*epsilon_voigt[5]*Q*Q
                    + F_el)*V)

    stv.set_state(P_orig, T_orig)
    return helmholtz, Q


print('Warning: Maybe not quite normalized correctly yet.')


def get_Cijs(volume, epsilon_voigt, Q_guess):
    """
    Super expensive way to get consistent Cijs.

    Currently each time this function is called, it does
    6*3 + 15*4 = 78 minimizations.

    There's probably some analytical derivatives we can exploit
    to speed this up.

    """
    de = 1.e-5
    C = np.empty((6, 6))
    for i in range(6):
        for j in range(i, 6):
            if i == j:
                e = np.copy(epsilon_voigt)
                A1, Q1 = min_helmholtz2(volume, e, Q_guess)
                e[i] -= de
                A0, Q0 = min_helmholtz2(volume, e, Q_guess)
                e[i] += 2.*de
                A2, Q2 = min_helmholtz2(volume, e, Q_guess)

                C[i][j] = (A2 + A0 - 2.*A1)/de/de

            else:
                e = np.copy(epsilon_voigt)
                e[i] -= de/2.
                e[j] -= de/2.
                A0, Q0 = min_helmholtz2(volume, e, Q_guess)
                e[i] += de
                A1, Q1 = min_helmholtz2(volume, e, Q_guess)
                e[i] -= de
                e[j] += de
                A2, Q2 = min_helmholtz2(volume, e, Q_guess)
                e[i] += de
                A3, Q3 = min_helmholtz2(volume, e, Q_guess)

                C[i][j] = ((A3 - A2) - (A1 - A0))/de/de
                C[j][i] = C[i][j]
                
    C /= volume
    #C += stv.isothermal_stiffness_tensor
    return C


pressures = np.linspace(45.e9, 55.e9, 21)
temperatures = 300. + pressures*0.
new_pressures = np.empty_like(pressures)
params = np.empty((len(pressures), 4))
Cijs = np.empty((len(pressures), 6, 6))
helmholtz = np.empty_like(pressures)
gibbs = np.empty_like(pressures)
volumes = np.empty_like(pressures)

guess = [0.1, 0., 0., 0.]
for i, P in enumerate(pressures):
    print(i)
    stv.set_state(P, 300.)

    dV = 1.e-9
    sol1 = minimize(fn_helmholtz, guess, args=(stv.V-dV/2.), bounds=((0,None), (-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05)))
    sol2 = minimize(fn_helmholtz, guess, args=(stv.V+dV/2.), bounds=((0,None), (-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05)))

    new_pressures[i] = P - (sol2.fun - sol1.fun)/dV
    params[i] = (sol1.x + sol2.x)/2.
    delta_helmholtz = (sol1.fun + sol2.fun)/2.
    
    guess = params[i]
    guess[0] += 0.01  # Q = 0 is always a solution

    Q = params[i, 0]
    epsilon_voigt = np.zeros(6)
    epsilon_voigt[:3] = params[i, 1:]
    Cijs[i] = get_Cijs(stv.V, epsilon_voigt, Q+0.1)

    helmholtz[i] = stv.helmholtz + delta_helmholtz
    gibbs[i] = stv.helmholtz + delta_helmholtz + new_pressures[i] * stv.V
    volumes[i] = stv.V

# Plotting
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
ax[1].plot(new_pressures/1.e9, np.gradient(gibbs - gibbs_base, new_pressures), linestyle='--')
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
KTs = np.empty_like(new_pressures)

for i in range(len(Cijs)):
    try:
        Sijs = np.linalg.inv(Cijs[i])
    except:
        Sijs = Cijs[i] * 0.
    betas[i] = np.sum(Sijs[:3, :3])
    KTs[i] = np.sum(Cijs[i, :3, :3])/9.

ax[5].plot(new_pressures/1.e9, 1./betas/1.e9, label='from Cijs (Reuss bound)')
ax[5].plot(new_pressures/1.e9, KTs/1.e9, label='from Cijs (Voigt bound)')
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
