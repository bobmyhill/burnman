from burnman.classes.anisotropicmineral import AnisotropicMineral
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.sputils import validateaxis
from stishovite_isotropic import make_phase
from scipy.linalg import expm, logm
from scipy.optimize import minimize
from carpenter_functions import Cij as Cij_orig
from scipy.interpolate import interp1d

stv = make_phase()


voigt = [[0, 0], [1, 1], [2, 2],
         [1, 2], [0, 2], [0, 1]]

if False:
    # test
    T = 300.
    P0 = stv.method.pressure(T, stv.params['V_0'], stv.params)
    stv.set_state(P0, T)

    F0 = stv.deformation_gradient_tensor

    P2 = stv.method.pressure(T, stv.params['V_0']*0.9999, stv.params)
    stv.set_state(P2, T)

    F1 = stv.deformation_gradient_tensor

    e = logm(F1) - logm(F0)
    print(e)

    P1 = stv.method.pressure(T, stv.params['V_0']*0.99995, stv.params)
    stv.set_state(P1, T)
    dsigma = -np.einsum('ijkl, kl', stv.full_isothermal_stiffness_tensor, e)
    print(dsigma)

    print(P2)
    exit()

# volume change is zero, so
# chi_11 + chi_12 + chi_13
# + chi_21 + chi_22 + chi_23
# + chi_31 + chi_32 + chi_33 = 0

# Strains in the Carpenter model are reasonably matched with:
# chi_1k (sum over k = 1,2,3) equal to -0.05 * Q + 0.011*Q*Q
# chi_2k (sum over k = 1,2,3) equal to 0.05 * Q + 0.011*Q*Q
# chi_3k (sum over k = 1,2,3) equal to -0.022*Q*Q

# In the high symmetry phase

# both phases are orthotropic, so
# chi_{4,5,6}{1,2,3} and chi_{1,2,3}{4,5,6} equal to 0

# Shear terms must have a volume dependence?
# (splitting not possible otherwise)

print('chi and dchidQ are only ever used in chi_ijkk form'
      '(i.e. only the column-sums of the first three rows are used) ')
print('Might as well put them in the diagonal elements, and make sure they sum to zero')
print('this also means that any shear terms must have a volume or temperature dependence'
      'as well as a proportion-dependence')

# Here's where the symmetry-breaking terms go
# These must be paired correctly (in this case, interchanging axes 1 and 2)
# Because the terms are always summed, there's not much point in assigning
# to anything but the diagonal components.
c110 = -0.03
c120 = 0.
c130 = 0.
c220 = 0.03
c230 = 0.
c330 = 0.
c440 = 0.
c550 = 0.
c660 = 0.

# Here's where the non-symmetry-breaking terms go
c111 = 0.003
c121 = 0.
c131 = 0.
c221 = 0.003
c231 = 0.
c331 = -0.006
c441 = 0.
c551 = 0.
c661 = 0.


chi0 = np.array([[c110, c120, c130, 0., 0., 0.],
                 [c120, c220, c230, 0., 0., 0.],
                 [c130, c230, c330, 0., 0., 0.],
                 [0., 0., 0., c440, 0., 0.],
                 [0., 0., 0., 0., c550, 0.],
                 [0., 0., 0., 0., 0., c660]])


chi1 = np.array([[c111, c121, c131, 0., 0., 0.],
                 [c121, c221, c231, 0., 0., 0.],
                 [c131, c231, c331, 0., 0., 0.],
                 [0., 0., 0., c441, 0., 0.],
                 [0., 0., 0., 0., c551, 0.],
                 [0., 0., 0., 0., 0., c661]])

# chi0 *= 0.
# chi1 *= 0.

assert np.abs(np.sum(chi0[:3, :3])) < 1.e-10
assert np.abs(np.sum(chi1[:3, :3])) < 1.e-10

Pord = 3.e9
stv.set_state(50.e9, 300.)
a = -2.*Pord*(stv.V-stv.params['V_0'])

stv.set_state(150.e9, 300.)
b = - (a + 2.*Pord*(stv.V-stv.params['V_0']))


def chi(Q):
    return chi0*Q + chi1*Q*Q


def dchidQ(Q):
    return chi0 + 2.*chi1*Q


def P(Q, V):
    Pdis = stv.method.pressure(300., V, stv.params)
    return Pdis - Pord*Q*Q


def F(Q, V):
    """
    The molar Helmholtz
    """
    return (0.5*a + Pord*(V-stv.params['V_0']))*Q*Q + 1./4.*b*np.power(Q, 4.)


def dFdQ(Q, V):
    """
    The molar Helmholtz
    """
    return (a + 2.*Pord*(V-stv.params['V_0']))*Q + b*np.power(Q, 3.)


def Q(V):
    x = -(a + 2.*Pord*(V-stv.params['V_0']))/b
    if x > 0.:
        return np.sqrt(x)
    else:
        return 0.



print('WARNING: SUSCEPTIBILITY NOT GOOD YET')

"""
def susceptibility(V):

    #d^2F/dQdQ

    #NOT GOOD YET
    #The susceptibility should be at the equilibrium Q, but *not*
    #allowing epsilon to vary.

    #In other words, this is the gradient of the change in
    # Helmholtz free energy with Q at the equilibrium Q
    # but without the elastic contribution.

    Q_eqm = Q(V)
    print(Q_eqm)
    Pdis = stv.method.pressure(300., V, stv.params)
    stv.set_state(Pdis, 300.)

    Cijkl0 = stv.full_isothermal_stiffness_tensor

    x = dchidQ(Q_eqm)
    dXdQ_full = AnisotropicMineral._voigt_notation_to_compliance_tensor(stv, x)
    depsilondQ = -np.einsum('ijkl, kl', dXdQ_full, np.eye(3))
    
    d2FdQdQ = (2.*(0.5*a + Pord*(V-stv.params['V_0'])) + 3.*b*Q_eqm*Q_eqm)
    d2FdQdQ += np.einsum('ij, ijkl, kl', depsilondQ, Cijkl0, depsilondQ)*V
    return 1./d2FdQdQ/V
"""

def deformation_gradient_tensor(Q):
    X_full = AnisotropicMineral._voigt_notation_to_compliance_tensor(stv,
                                                                     chi(Q))
    F = expm(np.einsum('ijkl, kl', X_full, np.eye(3)))
    return F


def epsilon_voigt(V):
    Q_eqm = Q(V)
    F = deformation_gradient_tensor(Q_eqm)
    return np.array([F[0, 0]-1., F[1, 1]-1., F[2, 2]-1., 0., 0., 0.])


print('Need to find analytical solution for Cijs')


def Cij_from_x(V):
    Q_eqm = Q(V)
    Pdis = stv.method.pressure(300., V, stv.params)
    stv.set_state(Pdis, 300.)

    Cijkl0 = stv.full_isothermal_stiffness_tensor

    x = dchidQ(Q_eqm)
    dXdQ_full = AnisotropicMineral._voigt_notation_to_compliance_tensor(stv, x)
    depsilondQ = -np.einsum('ijkl, kl', dXdQ_full, np.eye(3))

    d2FdQdQ = (2.*(0.5*a + Pord*(V-stv.params['V_0'])) + 3.*b*Q_eqm*Q_eqm)/V
    d2FdQdQ += np.einsum('ij, ijkl, kl', depsilondQ, Cijkl0, depsilondQ)
    susceptibility = 1./d2FdQdQ

    A = np.einsum('ijkl, kl', Cijkl0, depsilondQ)
    Cijkl = Cijkl0 - np.einsum('ij, kl', A, A)*susceptibility

    Cij = np.empty((6, 6))
    for j in range(6):
        for k in range(6):
            vj = voigt[j]
            vk = voigt[k]
            Cij[j, k] = Cijkl[vj[0], vj[1], vk[0], vk[1]]

    return Cij

"""
pressures = np.linspace(1.e5, 100.e9, 101)
volumes = stv.evaluate(['V'], pressures, pressures*0. + 300.)[0]

xs = np.empty_like(volumes)
xs2 = np.empty_like(volumes)
xs3 = np.empty_like(volumes)
xs4 = np.empty_like(volumes)
for i, V in enumerate(volumes):
    xs[i] = Cij_from_x(V)[0, 0]
    xs2[i] = Cij_from_x(V)[1, 0]
    xs3[i] = Cij_from_x(V)[1, 1]
    xs4[i] = Cij_from_x(V)[2, 2]
    
plt.plot(pressures, xs)
plt.plot(pressures, xs2)
plt.plot(pressures, xs3)
plt.plot(pressures, xs4)
plt.show()

exit()
"""

def normalize_Ve(volume, eps):
    """
    For consistent evaluation, the strains input to the Helmholtz
    function must be isochoric (have a determinant of zero).

    For non isochoric strains, the volume must be adjusted and then
    the component of strain due to the change in volume must be removed.

    If F0, F1 and F2 are the deformation gradient tensors at the
    initial volume at the reference state, final volume at the reference
    state and final volume *off* the reference state, and Fa is the
    (small) deformation gradient tensor for the deformation from F0, then the
    corrected (small) deformation gradient tensor Fb is given by the following:

    F2 = Fa.F0 = Fb.F1
    For orthotropic materials, F0 and F1 are diagonal, so we can rewrite:
    Fb = Fa.(F0/F1), where F0/F1 is also diagonal, with elements
    (F0/F1)ii = F0ii/F1ii.

    """
    F = eps + np.eye(3)
    # print(np.log(np.linalg.det(F)), np.trace(eps))
    VoverV0 = np.exp(np.trace(eps))
    volume_new = volume*VoverV0
    temperature = 300.
    pressure = stv.method.pressure(temperature, volume, stv.params)
    pressure_new = stv.method.pressure(temperature, volume_new, stv.params)

    # the following only works for orthotropic materials where the
    # deformation tensor is diagonal.
    # also seems to produce the wrong answer :(

    stv.set_state(pressure, temperature)
    F0 = np.diag(stv.deformation_gradient_tensor)
    stv.set_state(pressure_new, temperature)
    F1 = np.diag(stv.deformation_gradient_tensor)

    eps_V = np.log(F1) - np.log(F0)
    eps_new = eps - np.diag(eps_V)  # make eps 2D again

    # F_new = np.einsum('ij, jk', F, np.diag(F0/F1))
    # eps_new = F_new - np.eye(3)

    stv.set_state(pressure, temperature)
    return (volume_new, eps_new)


def dedQ(Q):
    """
    ln_M F = X_ijkl delta_kl
    de = dln_M(F) = dX delta_kl
    """
    dchidQ_full = AnisotropicMineral._voigt_notation_to_compliance_tensor(stv, dchidQ(Q))
    return np.einsum('ijkl, kl', dchidQ_full, np.eye(3))


def min_helmholtz2(volume, eps, Q_orig):
    """
    Helmholtz per unit volume
    """
    T_orig = 300.
    P_orig = stv.method.pressure(T_orig, volume, stv.params)
    stv.set_state(P_orig, T_orig)

    # F0 = stv.helmholtz
    # FV0 = stv.helmholtz/stv.V
    Cijkl0 = stv.full_isothermal_stiffness_tensor

    volum, e = normalize_Ve(volume, eps)
    Pdis = stv.method.pressure(T_orig, volum, stv.params)
    stv.set_state(Pdis, T_orig)
    F1 = stv.helmholtz
    # FV1 = stv.helmholtz/stv.V

    # F_el_iso = F1 - F0
    # FV_el_iso = FV1 - FV0

    Cijkl0 = stv.full_isothermal_stiffness_tensor
    stv.set_state(P_orig, T_orig)

    def local_helmholtz(dQ):
        """
        Helmholtz per unit volume
        Given a state defined by an initial strain away
        from a volume and Q,
        find the change to that Q that minimizes the free energy

        helmholtz0 is comprised of the
        isotropic elastic and landau contributions
        to the helmholtz energy at the present Q and V.
        eps is the modified strain
        """

        # helmholtz0 = dFdQ(Q_orig+dQ[0]/2., volume)*dQ[0]/volume
        # eps_new = eps - dedQ(Q_orig + dQ[0]/2.)*dQ[0]
        # produces no change in K_T

        helmholtz0 = dFdQ(Q_orig+dQ[0]/2., volum)*dQ[0]/volum
        eps_new = eps - dedQ(Q_orig + dQ[0]/2.)*dQ[0]

        return helmholtz0 + 1./2.*np.einsum('ij, ijkl, kl',
                                            eps_new, Cijkl0, eps_new)

    sol = minimize(local_helmholtz, [0.])

    stv.set_state(P_orig, T_orig)
    return F1 + sol.fun


def get_Cijs(volume, Q_guess):
    """
    Super expensive way to get consistent Cijs.

    Currently each time this function is called, it does
    3*3 + 18*4 = 81 minimizations.

    There's probably some analytical derivatives we can exploit
    to speed this up.

    """
    eps = np.array([[0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]])

    de = 1.e-6
    C = np.empty((6, 6))
    for i in range(6):
        for j in range(i, 6):

            deps_i = np.zeros((3, 3))
            deps_j = np.zeros((3, 3))

            i0, i1 = voigt[i]
            j0, j1 = voigt[j]
            deps_i[i0, i1] += de/2
            deps_i[i1, i0] += de/2
            deps_j[j0, j1] += de/2
            deps_j[j1, j0] += de/2

            if i == j:
                e = np.copy(eps)
                A1 = min_helmholtz2(volume, e, Q_guess)
                e -= deps_i
                A0 = min_helmholtz2(volume, e, Q_guess)
                e += 2.*deps_i
                A2 = min_helmholtz2(volume, e, Q_guess)

                C[i][j] = (A2 + A0 - 2.*A1)/de/de
            else:
                e = np.copy(eps)
                e -= deps_i/2.
                e -= deps_j/2.
                A0 = min_helmholtz2(volume, e, Q_guess)
                e += deps_i
                A1 = min_helmholtz2(volume, e, Q_guess)
                e -= deps_i
                e += deps_j
                A2 = min_helmholtz2(volume, e, Q_guess)
                e += deps_i
                A3 = min_helmholtz2(volume, e, Q_guess)

                C[i][j] = ((A3 - A2) - (A1 - A0))/de/de
                C[j][i] = C[i][j]

    return C


pressures = np.linspace(40.e9, 80.e9, 11)
volumes = np.empty_like(pressures)
new_pressures = np.empty_like(pressures)
chis = np.empty_like(pressures)
Qs = np.empty_like(pressures)
epsilons = np.empty((len(pressures), 3))
Cijs = np.empty((len(pressures), 6, 6))
Cij2s = np.empty((len(pressures), 6, 6))
helmholtz = np.empty_like(pressures)
for i, pressure in enumerate(pressures):
    if i % 1 == 0:
        print(i)
    stv.set_state(pressure, 300.)
    volumes[i] = stv.V
    Qs[i] = Q(stv.V)
    new_pressures[i] = P(Qs[i], stv.V)
    eps = epsilon_voigt(stv.V)
    epsilons[i] = eps[:3]
    helmholtz[i] = stv.helmholtz + F(Qs[i], stv.V)
    Cijs[i] = get_Cijs(stv.V, Qs[i])
    
    Cij2s[i] = Cij_from_x(stv.V)
    

# plt.plot(pressures, chis)
# plt.plot(pressures, Qs)
# plt.plot(pressures, epsilons[:, 0])
# plt.plot(pressures, epsilons[:, 1])
# plt.plot(pressures, epsilons[:, 2])


Carpenter_Cijs = np.array([Cij_orig(P/1.e9)['C']
                           for P in new_pressures])

base_Cijs = np.empty_like(Carpenter_Cijs)
for i, P in enumerate(new_pressures):
    stv.set_state(P, 300.)
    base_Cijs[i] = stv.isothermal_stiffness_tensor/1.e9


fig = plt.figure(figsize=(12, 4))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]


for (i, j) in [(1, 1),
               (2, 2),
               (3, 3),
               (4, 4),
               (5, 5),
               (6, 6),
               (1, 2),
               (1, 3),
               (2, 3)]:
    ln, = ax[2].plot(new_pressures/1.e9, Cijs[:, i-1, j-1]/1.e9,
                     label=f'{i}{j}', alpha=0.5)
    #ax[2].plot(new_pressures/1.e9, Cij2s[:, i-1, j-1]/1.e9,
    #           alpha=0.5, color=ln.get_color(), linestyle='--')
    ax[2].plot(new_pressures/1.e9, Carpenter_Cijs[:, i-1, j-1],
               linestyle=':', color=ln.get_color())

try:
    betas = np.array([np.sum(np.linalg.inv(Cij)[:3, :3]) for Cij in Cijs])
    beta2s = np.array([np.sum(np.linalg.inv(Cij)[:3, :3]) for Cij in Cij2s])
    ax[1].plot(new_pressures/1.e9, 1./betas/1.e9, label='K_T')
    #ax[1].plot(new_pressures/1.e9, 1./beta2s/1.e9, linestyle='--')
except np.linalg.LinAlgError:
    pass

interp_vol_base = interp1d(pressures, volumes, kind='cubic')
interp_vol = interp1d(new_pressures, volumes, kind='cubic')

print(pressures[-1], new_pressures[-1])
print(pressures[0], new_pressures[0])
# ax[0].plot(pressures/1.e9, volumes*1.e6)
# ax[0].plot(new_pressures/1.e9, volumes*1.e6)

interp_pressures = np.linspace(new_pressures[0] + 100000.,
                               new_pressures[-1] - 100000.,
                               len(new_pressures))

ax[0].plot(interp_pressures/1.e9,
           (interp_vol(interp_pressures)
            - interp_vol_base(interp_pressures))*1.e6)
ax[1].plot(new_pressures/1.e9,
           -volumes*np.gradient(new_pressures, volumes, edge_order=2)/1.e9,
           label='K_T from pressures')

ax[1].plot(new_pressures/1.e9,
           volumes*np.gradient(np.gradient(helmholtz,
                                           volumes, edge_order=2),
                               volumes, edge_order=2)/1.e9,
           label='K_T from helmholtz', linestyle=':')

ax[1].legend()
ax[2].legend()
plt.savefig('save.pdf')
plt.show()
