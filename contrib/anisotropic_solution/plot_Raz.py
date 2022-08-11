import burnman
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from slb_qtz import qtz_alpha, qtz_beta, b_BD, c_BD, qtz_ss_scalar
from slb_qtz import helmholtz_free_energy_alpha, helmholtz_free_energy_beta




Raz_d = np.loadtxt('./data/Raz_et_al_2002_quartz_PVT.dat', unpack=True)
Raz_pressures = sorted(set(Raz_d[0]))
Raz_temperatures = sorted(set(Raz_d[1]))


for P in Raz_pressures:
    idx = np.where(Raz_d[0] == P)
    plt.scatter(Raz_d[1][idx], Raz_d[2][idx], label=f'{P} bar')
plt.legend()
plt.xlabel('Temperature (C)')
plt.ylabel('Volume (cm$^3$/mol)')
plt.show()


fig = plt.figure(figsize=(12, 4))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

for T in Raz_temperatures[:2]:
    idx = np.where(Raz_d[1] == T)
    ax[0].scatter(Raz_d[0][idx], Raz_d[2][idx], label=f'{T} C')
    ax[1].scatter(Raz_d[0][idx], Raz_d[3][idx], label=f'{T} C')
    ax[2].scatter(Raz_d[0][idx], Raz_d[4][idx], label=f'{T} C')
    ax[3].scatter(Raz_d[0][idx], Raz_d[5][idx], label=f'{T} C')
ax[0].legend()
ax[0].set_xlabel('Pressure (bar)')
ax[0].set_ylabel('Volume (cm$^3$/mol)')
plt.show()
