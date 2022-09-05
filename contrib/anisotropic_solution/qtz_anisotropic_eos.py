from xml.dom import minicompat
import burnman
import autograd.numpy as npa
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, root
from scipy.linalg import logm
from slb_qtz import qtz_alpha_anisotropic, qtz_beta, b_BD, c_BD, qtz_ss_anisotropic
from slb_qtz import helmholtz_free_energy_alpha, helmholtz_free_energy_beta

from burnman.tools.eos import check_anisotropic_eos_consistency


Raz_d = np.loadtxt('./data/Raz_et_al_2002_quartz_PVT.dat', unpack=True)
Raz_pressures = sorted(set(Raz_d[0]))
Raz_temperatures = sorted(set(Raz_d[1]))


sol = qtz_ss_anisotropic

def equilibrate_function(sol):
    P = sol.pressure
    T = sol.temperature
    def gibbs(Q):
        x = 0.5*(Q + 1.)
        sol.set_composition([x, 1.-x])
        sol.set_state(P, T)
        return sol.molar_gibbs
    def diff_partial_gibbs(Q):
        x = 0.5*(Q[0] + 1.)
        sol.set_composition([x, 1.-x])
        sol.set_state(P, T)
        return [sol.partial_gibbs[0] - sol.partial_gibbs[1]]
    Q = minimize_scalar(gibbs, (0.9, 1.), tol=1.e-12).x
    Q = root(diff_partial_gibbs, [Q], tol=1.e-12).x
    print(P/1.e9, T, Q)
    return Q

sol.set_composition([0.4, 0.6])
if True:
    check_anisotropic_eos_consistency(sol, 1.e5, 800., equilibration_function=equilibrate_function, verbose=True)

    print('unit cell', sol.cell_vectors)
    print('unit cell', sol.cell_parameters)
    print('volume', sol.molar_volume)
    print('K_RT', sol.isothermal_bulk_modulus_reuss)
    print('C_p', sol.molar_heat_capacity_p)
    exit()

# Goal: dsigma/deps
# At fixed state, change in stress only from change in volume (a pressure change)
# and the non-hydrostatic change in strain.
# Goal: dsigma/deps
# At fixed state, change in stress only from change in volume (a pressure change)
# and the non-hydrostatic change in strain.

if True:
    pressure0 = 1.e5
    temperature0 = 800.
    sol.set_composition([0.4, 0.6])
    sol.set_state(pressure0, temperature0)
    sol.set_relaxation(True)
    equilibrate_function(sol)
    x0 = sol.molar_fractions
    V0 = sol.molar_volume
    helm0 = sol.molar_helmholtz
    stress0 = -sol.pressure*np.eye(3)
    lnF0 = logm(sol.deformation_gradient_tensor)

    # The next value must be checked.
    # Too small and rounding errors get big.
    # Too big and errors get big
    deps = 6.e-7

    veps = np.array([deps, 0., 0., 0., 0., 0., 0.])

    eps = np.array([[veps[0], veps[5], veps[4]],
                    [veps[5], veps[1], veps[3]],
                    [veps[4], veps[3], veps[2]]])

    deltaVoverV = np.trace(eps)

    # nonhydrostatic strain
    eps2 = eps - sol.isothermal_compressibility_tensor/sol.isothermal_compressibility_reuss*deltaVoverV

    # nonhydrostatic part of the stress change
    dsigma2 = np.einsum('ijkl, kl', sol.full_isothermal_stiffness_tensor, eps2)

    # volumetric component of the stress change
    dsigma1 = deltaVoverV/sol.isothermal_compressibility_reuss*np.eye(3)

    print((dsigma1 + dsigma2)/deps)
    print(sol.isothermal_stiffness_tensor)



    unrelaxed_CT = np.zeros((3, 3, 3, 3))
    relaxed_CT = np.zeros((3, 3, 3, 3))

    for i in range(3):
        for j in range(3):
            eps = np.zeros((3, 3))
            eps[i, j] = deps

            deltaVoverV = np.trace(eps)


            def dhelmholtz(dQ):
                sol.set_composition([x0[0]-dQ, x0[1]+dQ])
                sol.set_state_with_volume(V0*(1. + deltaVoverV), temperature0)
                # eps due to changing Q and V
                F = sol.deformation_gradient_tensor
                eps_QV = logm(F) - lnF0

                # nonhydrostatic strain
                eps2 = eps - eps_QV

                dF = (sol.molar_helmholtz + 0.5*np.einsum('ij, ijkl, kl', eps2,
                                                        sol.full_isothermal_stiffness_tensor_unrelaxed,
                                                        eps2)) - helm0
                sol.set_state_with_volume(V0, temperature0)
                return dF

            def dstress(dQ):
                sol.set_composition([x0[0]-dQ, x0[1]+dQ])
                sol.set_state_with_volume(V0*(1. + deltaVoverV), temperature0)
                sol.set_composition([x0[0]-dQ, x0[1]+dQ])
                F = sol.deformation_gradient_tensor
                eps_QV = logm(F) - lnF0

                # nonhydrostatic strain
                eps2 = eps - eps_QV
                print(dQ, eps2)
                #print(sol.isothermal_stiffness_tensor)
                stress1 = -sol.pressure*np.eye(3) + np.einsum('ijkl, kl', 
                                                            sol.full_isothermal_stiffness_tensor_unrelaxed,
                                                            eps2)
                sol.set_state_with_volume(V0, temperature0)
                return stress1  - stress0

            Q_min = minimize_scalar(dhelmholtz, (-1.e-5, 1.e-5), tol=1.e-20).x
            print(i, j, Q_min, dstress(Q_min))
            relaxed_CT[:,:,i, j] = dstress(Q_min)/deps
            unrelaxed_CT[:,:,i, j] = dstress(0.)/deps


    print('RELAXED')
    relaxed_CT_Voigt = sol._contract_stiffnesses(relaxed_CT)
    relaxed_ST_Voigt = np.linalg.inv(relaxed_CT_Voigt)
    relaxed_beta_RT_Voigt = np.sum(relaxed_ST_Voigt[:3, :3])
    print(relaxed_CT_Voigt)
    print(1./relaxed_beta_RT_Voigt)
    print('UNRELAXED')
    unrelaxed_CT_Voigt = sol._contract_stiffnesses(unrelaxed_CT)
    unrelaxed_ST_Voigt = np.linalg.inv(unrelaxed_CT_Voigt)
    unrelaxed_beta_RT_Voigt = np.sum(unrelaxed_ST_Voigt[:3, :3])
    print(unrelaxed_CT_Voigt)
    print(1./unrelaxed_beta_RT_Voigt)





    sol.set_relaxation(True)
    sol.set_composition(x0)
    sol.set_state(pressure0, temperature0)

    print('OUTPUT')
    print(sol.isothermal_stiffness_tensor)
    print(sol.isothermal_bulk_modulus_reuss)


    sol.set_relaxation(False)
    print('TEST STRESS')
    sol.set_state(pressure0 + 1.e3, temperature0)
    F = sol.deformation_gradient_tensor
    eps_QV = logm(F) - lnF0
    print(np.einsum('ijkl, kl', unrelaxed_CT, eps_QV))
    print(np.einsum('ijkl, kl', sol.full_isothermal_stiffness_tensor, eps_QV))

    exit()



minT = 800.
maxT = 1000.
nT = 101
temperatures = np.linspace(minT, maxT, nT)
temperatures_2 = np.linspace(100, 1000., 101)
Qs = np.empty_like(temperatures)
Vs = np.empty_like(temperatures)
Ss = np.empty_like(temperatures)
Cps = np.empty_like(temperatures)
C_S = np.empty((nT, 6, 6))
cell_vectors = np.empty((nT, 3))


Qs_2 = np.empty_like(temperatures_2)
Vs_2 = np.empty_like(temperatures_2)
Ss_2 = np.empty_like(temperatures_2)
Cps_2 = np.empty_like(temperatures_2)

fig_T = plt.figure(figsize=(12, 8))
ax = [fig_T.add_subplot(2, 3, i) for i in range(1, 7)]


d = np.loadtxt('data/Bachheimer_Dolino_1975_quartz_Q.dat')
ax[0].scatter(d[:,0], d[:,1])
for P in [1.e5]: #, 0.1e9, 0.2e9, 0.3e9]:
    for i, T in enumerate(temperatures):
        sol.set_composition([0.9, 0.1])
        sol.set_state(P, T)
        Qs[i] = equilibrate_function(sol)
        Vs[i] = sol.V
        Ss[i] = sol.S
        Cps[i] = sol.molar_heat_capacity_p
        C_S[i] = sol.isentropic_stiffness_tensor
        cell_vectors[i] = np.diag(sol.cell_vectors)
        print(P/1.e9, T, Qs[i])
    ax[0].plot(temperatures, Qs)
    ax[1].plot(temperatures, Vs*1.e6)
    ax[2].plot(temperatures, Ss)
    ax[3].plot(temperatures, Cps)
    ax[3].plot(temperatures, temperatures*np.gradient(Ss, temperatures, edge_order=2))
    for (i, j) in [(1, 1),
                   (3, 3),
                   (4, 4),
                   (1, 2),
                   (1, 3),
                   (1, 4)]:
        ax[4].plot(temperatures, C_S[:,i-1,j-1]/1.e9, label=f'{i}{j}')

    ax[5].plot(temperatures, cell_vectors[:, 2]/cell_vectors[:, 0])
    

    qtz_HP = burnman.minerals.HP_2011_ds62.q()
    pressures_2 = P + temperatures_2*0.
    Vs_2, Ss_2, Cps_2 = qtz_HP.evaluate(['V', 'S', 'C_p'], pressures_2, temperatures_2)
    for i, T in enumerate(temperatures_2):
        qtz_HP.set_state(P, T)
        Qs_2[i] = qtz_HP.property_modifier_properties[0]['Q']

    ax[0].plot(temperatures_2, Qs_2, linestyle=':')
    ax[1].plot(temperatures_2, Vs_2*1.e6, linestyle=':')
    ax[2].plot(temperatures_2, Ss_2, linestyle=':')
    ax[3].plot(temperatures_2, Cps_2, linestyle=':')


for P in Raz_pressures:
    idx = np.where(Raz_d[0] == P)
    ax[1].scatter(Raz_d[1][idx]+273.15, Raz_d[2][idx], label=f'{P} bar')

Lak_d = np.loadtxt('data/Lakshtanov_et_al_2007_Cijs_quartz.dat', unpack=True)
# TC    rho     C11     C12     C13     C14     C33     C44     C66     K       G       Vpagg         Vsagg

ax[1].scatter(Lak_d[0]+273.15, 1000*qtz_HP.molar_mass/Lak_d[1])
ax[4].scatter(Lak_d[0]+273.15, Lak_d[2], label='C11')
ax[4].scatter(Lak_d[0]+273.15, Lak_d[3], label='C12')
ax[4].scatter(Lak_d[0]+273.15, Lak_d[4], label='C13')
ax[4].scatter(Lak_d[0]+273.15, Lak_d[5], label='C14')
ax[4].scatter(Lak_d[0]+273.15, Lak_d[6], label='C33')
ax[4].scatter(Lak_d[0]+273.15, Lak_d[7], label='C44')



ax[1].legend()
ax[4].legend()

for i in range(6):
    ax[i].set_xlabel('Temperature (K)')

ax[0].set_ylabel('Q')
ax[1].set_ylabel('Volume (cm$^3$/mol)')
ax[2].set_ylabel('Entropy (J/K/mol)')
ax[3].set_ylabel('Heat capacity (J/K/mol)')
ax[4].set_ylabel('$C_S$ (GPa)')
ax[5].set_ylabel('Unit cell ratio c/a')
fig_T.set_tight_layout(True)
fig_T.savefig('fig_T.pdf')



minP = 1.e5
maxP = 10.e9
nP = 101
pressures = np.linspace(minP, maxP, nP)
pressures_2 = np.linspace(minP, maxP, 101)
Qs = np.empty_like(pressures)
Vs = np.empty_like(pressures)
Ss = np.empty_like(pressures)
Cps = np.empty_like(pressures)
C_S = np.empty((nP, 6, 6))
cell_vectors = np.empty((nP, 3))


Qs_2 = np.empty_like(pressures_2)
Vs_2 = np.empty_like(pressures_2)
Ss_2 = np.empty_like(pressures_2)
Cps_2 = np.empty_like(pressures_2)

fig_P = plt.figure(figsize=(12, 8))
ax = [fig_P.add_subplot(2, 3, i) for i in range(1, 7)]


d = np.loadtxt('data/Jorgensen_1978_quartz_tilts_high_pressure.dat')
ax[0].scatter(d[:,0]/10., d[:,1]/((d[0,1]+d[1,1])/2.))
for T in [300.]: #, 0.1e9, 0.2e9, 0.3e9]:
    for i, P in enumerate(pressures):
        sol.set_composition([0.9, 0.1])
        sol.set_state(P, T)
        Qs[i] = equilibrate_function(sol)
        Vs[i] = sol.V
        Ss[i] = sol.S
        Cps[i] = sol.molar_heat_capacity_p
        C_S[i] = sol.isentropic_stiffness_tensor
        cell_vectors[i] = np.diag(sol.cell_vectors)
        print(P/1.e9, T, Qs[i])
    ax[0].plot(pressures/1.e9, Qs)
    ax[1].plot(pressures/1.e9, Vs*1.e6)
    ax[2].plot(pressures/1.e9, Ss)
    ax[3].plot(pressures/1.e9, Cps)
    for (i, j) in [(1, 1),
                   (3, 3),
                   (4, 4),
                   (1, 2),
                   (1, 3),
                   (1, 4)]:
        ax[4].plot(pressures/1.e9, C_S[:,i-1,j-1]/1.e9, label=f'{i}{j}')

    ax[5].plot(pressures/1.e9, cell_vectors[:, 2]/cell_vectors[:, 0])
    

    qtz_HP = burnman.minerals.HP_2011_ds62.q()
    temperatures_2 = T + pressures_2*0.
    Vs_2, Ss_2, Cps_2 = qtz_HP.evaluate(['V', 'S', 'C_p'], pressures_2, temperatures_2)
    for i, P in enumerate(pressures_2):
        qtz_HP.set_state(P, T)
        Qs_2[i] = qtz_HP.property_modifier_properties[0]['Q']

    ax[0].plot(pressures_2/1.e9, Qs_2, linestyle=':')
    ax[1].plot(pressures_2/1.e9, Vs_2*1.e6, linestyle=':')
    ax[2].plot(pressures_2/1.e9, Ss_2, linestyle=':')
    ax[3].plot(pressures_2/1.e9, Cps_2, linestyle=':')

Wang_d = np.loadtxt('data/Wang_2015_elastic_tensor_quartz_pressure.dat', unpack=True)

# P rho 0, 1
# C11 C11err 2, 3
# C33 C33err 4, 5
# C12 C12err 6, 7
# C13 C13err 8, 9
# C14 C14err 10, 11
# C44 C44err 12, 13
# K0S G RMS ...

ax[1].scatter(Wang_d[0], 1000*qtz_HP.molar_mass/Wang_d[1])
ax[4].scatter(Wang_d[0], Wang_d[2], label='C11')
ax[4].scatter(Wang_d[0], Wang_d[4], label='C33')
ax[4].scatter(Wang_d[0], Wang_d[6], label='C12')
ax[4].scatter(Wang_d[0], Wang_d[8], label='C13')
ax[4].scatter(Wang_d[0], Wang_d[10], label='C14')
ax[4].scatter(Wang_d[0], Wang_d[12], label='C44')

ax[1].legend()
ax[4].legend()

for i in range(6):
    ax[i].set_xlabel('Pressure (GPa)')

ax[0].set_ylabel('Q')
ax[1].set_ylabel('Volume (cm$^3$/mol)')
ax[2].set_ylabel('Entropy (J/K/mol)')
ax[3].set_ylabel('Heat capacity (J/K/mol)')
ax[4].set_ylabel('$C_S$ (GPa)')
ax[5].set_ylabel('Unit cell ratio c/a')
fig_P.set_tight_layout(True)
fig_P.savefig('fig_P.pdf')
