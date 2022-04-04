import numpy as np
import matplotlib.pyplot as plt

from burnman.minerals.SLB_2011 import stishovite, coesite

# The following is for a tricritical Landau transition.

Edis = 1000.
Sdis = 10.

stv = stishovite()
coe = coesite()


def G_ord(P, T):
    # return 0.
    stv.set_state(P, T)
    return stv.gibbs


def G_dis(P, T):
    # return Edis - T*Sdis
    coe.set_state(P, T)
    return coe.gibbs


def delta_G(P, T):
    Gord = G_ord(P, T)
    Gdis = G_dis(P, T) - (G_ord(P, 0) - G_dis(P, 0)) / 3.
    Gord0 = G_ord(P, 0)
    Gdis0 = G_dis(P, 0) - (G_ord(P, 0) - G_dis(P, 0)) / 3.
    if Gdis > Gord:
        Q2 = np.sqrt((Gord - Gdis)/(Gord0 - Gdis0))
    else:
        Q2 = 0.

    Q6 = Q2 * Q2 * Q2

    delta_G = (Gord - Gdis) * (Q2 - 1.) - (Gord0 - Gdis0) * (Q6 - 1.) / 3.
    return delta_G


def Q(P, T):
    Gord = G_ord(P, T)
    Gdis = G_dis(P, T) - (G_ord(P, 0) - G_dis(P, 0)) / 3.
    Gord0 = G_ord(P, 0)
    Gdis0 = G_dis(P, 0) - (G_ord(P, 0) - G_dis(P, 0)) / 3.
    if Gdis > Gord:
        Q2 = np.sqrt((Gord - Gdis)/(Gord0 - Gdis0))
    else:
        Q2 = 0.
    return np.sqrt(Q2)


temperatures = np.linspace(0., 3500., 101)

fig = plt.figure(figsize=(8, 4))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]


for i, P in enumerate([6.e9, 8.e9, 10.e9, 12.e9]):
    gibbs = np.array([delta_G(P, T) + G_ord(P, T) for T in temperatures])
    Qs = np.array([Q(P, T) for T in temperatures])
    ln, = ax[0].plot(temperatures, gibbs, label=f'{P/1.e9} GPa')
    Gdis = [G_dis(P, T) for T in temperatures]
    Gord = [G_ord(P, T) for T in temperatures]
    ax[0].fill_between(temperatures, Gdis, Gord, color=ln.get_color(),
                       alpha=0.2)
    ax[0].plot(temperatures, Gdis, color=ln.get_color(), linestyle=':',
               alpha=0.5)
    ax[0].plot(temperatures, Gord, color=ln.get_color(), linestyle=':',
               alpha=0.5)

    ax[1].plot(temperatures, np.power(Qs, 2.), label=f'{P/1.e9} GPa')
    # plt.plot(temperatures, -np.gradient(gibbs, temperatures, edge_order=2))

ax[0].set_ylabel('Gibbs (J/mol)')
ax[1].set_ylabel('Proportion of ordered phase (Q$^2$)')

for i in range(2):
    ax[i].legend()
    ax[i].set_xlabel('Temperature (K)')

fig.set_tight_layout(True)



pressures = np.linspace(4.e9, 16.e9, 101)

fig = plt.figure(figsize=(8, 4))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]


for i, T in enumerate([0., 500., 1000.]):
    gibbs = np.array([delta_G(P, T) + G_ord(P, T) for P in pressures])
    Qs = np.array([Q(P, T) for T in temperatures])
    ln, = ax[0].plot(temperatures, gibbs, label=f'{P/1.e9} GPa')
    Gdis = [G_dis(P, T) for T in temperatures]
    Gord = [G_ord(P, T) for T in temperatures]
    ax[0].fill_between(pressures/1.e9, Gdis, Gord, color=ln.get_color(),
                       alpha=0.2)
    ax[0].plot(pressures/1.e9, Gdis, color=ln.get_color(), linestyle=':',
               alpha=0.5)
    ax[0].plot(pressures/1.e9, Gord, color=ln.get_color(), linestyle=':',
               alpha=0.5)

    ax[1].plot(pressures/1.e9, np.power(Qs, 2.), label=f'{T} K')
    # plt.plot(temperatures, -np.gradient(gibbs, temperatures, edge_order=2))

ax[0].set_ylabel('Gibbs (J/mol)')
ax[1].set_ylabel('Proportion of ordered phase (Q$^2$)')

for i in range(2):
    ax[i].legend()
    ax[i].set_xlabel('Temperature (K)')

fig.set_tight_layout(True)
plt.show()
