import numpy as np
import matplotlib.pyplot as plt
import burnman_path
import burnman
from simple_fper_solution import ferropericlase

assert burnman_path  # silence pyflakes warning


fper = ferropericlase()

pressures = np.linspace(10.e9, 150.e9, 101)
volumes = np.empty_like(pressures)
volumes_HS = np.empty_like(pressures)
volumes_LS = np.empty_like(pressures)
delta_MgO_volumes = np.empty_like(pressures)
p_LS = np.empty_like(pressures)

fig = plt.figure(figsize=(15, 5))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]


dat = np.loadtxt('Komabayashi_2010.dat')

per = burnman.minerals.SLB_2011.periclase()
# per = burnman.minerals.HP_2011_ds62.per()
expt_MgO_volumes = np.empty_like(dat[:, 0])
for i, d in enumerate(dat):
    per.set_state(d[2]*1.e9, d[0])
    expt_MgO_volumes[i] = per.V

ax[0].scatter(dat[:, 2], dat[:, 6]*6.022/4./10.)
ax[2].scatter(dat[:, 2], dat[:, 6] - expt_MgO_volumes/6.022*4.*10.e6)

X_Fe = 0.19  # Komabayashi et al. (2010) composition
for T in [300., 1800., 3300.]:
    for i, P in enumerate(pressures):

        fper.set_state(P, T)
        fper.set_equilibrium_composition(X_Fe)

        volumes[i] = fper.V
        p_LS[i] = (fper.molar_fractions[2]
                   / (fper.molar_fractions[1]+fper.molar_fractions[2]))

        fper.set_composition([1., 0., 0.])
        delta_MgO_volumes[i] = volumes[i] - fper.V

        fper.set_composition([1.-X_Fe, X_Fe, 0.])
        volumes_HS[i] = fper.V
        fper.set_composition([1.-X_Fe, 0., X_Fe])
        volumes_LS[i] = fper.V

    ax[0].plot(pressures/1.e9, volumes*1.e6, label=f'{T} K')
    ax[0].plot(pressures/1.e9, volumes_HS*1.e6, linestyle=':')
    ax[0].fill_between(pressures/1.e9, volumes_HS, volumes_LS, alpha=0.2)
    ax[1].plot(pressures/1.e9, 1.-p_LS, label=f'{T} K')
    ax[2].plot(pressures/1.e9, delta_MgO_volumes/6.022*4.*10.e6,
               label=f'{T} K')


ax[0].set_xlabel('Pressure (GPa)')
ax[1].set_xlabel('Pressure (GPa)')
ax[0].set_ylabel('Volume (cm$^3$/mol)')
ax[1].set_ylabel('high spin fraction')

ax[0].set_ylim(7, 13)
ax[0].legend()
ax[1].legend()
plt.show()
