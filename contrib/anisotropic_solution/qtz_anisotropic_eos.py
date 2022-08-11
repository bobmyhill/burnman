import burnman
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from slb_qtz import qtz_alpha, qtz_beta, b_BD, c_BD, qtz_ss_anisotropic
from slb_qtz import helmholtz_free_energy_alpha, helmholtz_free_energy_beta

from burnman.tools.eos import check_anisotropic_eos_consistency


Raz_d = np.loadtxt('./data/Raz_et_al_2002_quartz_PVT.dat', unpack=True)
Raz_pressures = sorted(set(Raz_d[0]))
Raz_temperatures = sorted(set(Raz_d[1]))


sol = qtz_ss_anisotropic

qtz_ss_anisotropic.set_composition([0.3, 0.7])
check_anisotropic_eos_consistency(sol, 1.e9, 500., verbose=True)
exit()

qtz_HP = burnman.minerals.HP_2011_ds62.q()

def eqm_order(P, T):
    def gibbs(Q):
        x = 0.5*(Q + 1.)
        sol.set_composition([x, 1.-x])
        sol.set_state(P, T)
        return sol.molar_gibbs
    return minimize_scalar(gibbs, (0.9, 1.))

nT = 101
temperatures = np.linspace(10, 1000., nT)
temperatures_2 = np.linspace(100, 1000., 101)
Qs = np.empty_like(temperatures)
Vs = np.empty_like(temperatures)
Ss = np.empty_like(temperatures)
Cps = np.empty_like(temperatures)
C_S = np.empty((nT, 6, 6))


Qs_2 = np.empty_like(temperatures_2)
Vs_2 = np.empty_like(temperatures_2)
Ss_2 = np.empty_like(temperatures_2)
Cps_2 = np.empty_like(temperatures_2)

fig = plt.figure(figsize=(12, 8))
ax = [fig.add_subplot(2, 3, i) for i in range(1, 6)]


d = np.loadtxt('data/Bachheimer_Dolino_1975_quartz_Q.dat')
ax[0].scatter(d[:,0], d[:,1])
for P in [1.e5]: #, 0.1e9, 0.2e9, 0.3e9]:
    for i, T in enumerate(temperatures):
        Qs[i] = eqm_order(P, T).x
        Vs[i] = sol.V
        Ss[i] = sol.S
        C_S[i] = sol.isentropic_stiffness_tensor
        print(P/1.e9, T, Qs[i])
    ax[0].plot(temperatures, Qs)
    ax[1].plot(temperatures, Vs)
    ax[2].plot(temperatures, Ss)
    ax[3].plot(temperatures, temperatures*np.gradient(Ss, temperatures, edge_order=2))
    for (i, j) in [(1, 1),
                   (3, 3),
                   (4, 4),
                   (1, 2),
                   (1, 3),
                   (1, 4)]:
        ax[4].plot(temperatures, C_S[:,i-1,j-1], label=f'{i}{j}')

    pressures_2 = P + temperatures_2*0.
    Vs_2, Ss_2, Cps_2 = qtz_HP.evaluate(['V', 'S', 'C_p'], pressures_2, temperatures_2)
    for i, T in enumerate(temperatures_2):
        qtz_HP.set_state(P, T)
        Qs_2[i] = qtz_HP.property_modifier_properties[0]['Q']

    ax[0].plot(temperatures_2, Qs_2, linestyle=':')
    ax[1].plot(temperatures_2, Vs_2, linestyle=':')
    ax[2].plot(temperatures_2, Ss_2, linestyle=':')
    ax[3].plot(temperatures_2, Cps_2, linestyle=':')


for P in Raz_pressures:
    idx = np.where(Raz_d[0] == P)
    ax[1].scatter(Raz_d[1][idx]+273.15, Raz_d[2][idx]/1.e6, label=f'{P} bar')

ax[4].legend()

plt.show()
exit()

