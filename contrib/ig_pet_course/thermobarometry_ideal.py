import numpy as np
import matplotlib.pyplot as plt


def P(T, a, b):
    return a*T + b


E = 2000.
V = 1.e-6
S = 10.
R = 8.31446

fig = plt.figure(figsize=(8, 3))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

Ts = np.linspace(200., 1000., 101)
for lnK in [0.2, 0.4, 0.6, 0.8]:
    a = (S - R*lnK) / V
    b = -E/V

    ax[0].plot(Ts, P(Ts, a, b)/1.e9,
               label=f'$\\ln K = {lnK}$')


E = 000.
V = 1.e-5
S = 10.
R = 8.31446

Ts = np.linspace(200., 1000., 101)
for lnK in [0.2, 0.4, 0.6, 0.8]:
    a = (S - R*lnK) / V
    b = -E/V

    ax[1].plot(Ts, P(Ts, a, b)/1.e9,
               label=f'$\\ln K = {lnK}$')

for i in range(2):
    ax[i].set_xlim(200., 1000.)
    ax[i].set_ylim(0., 1.)
    ax[i].set_xlabel('T (K)')
    ax[i].set_ylabel('P (GPa)')
    ax[i].legend()

fig.set_tight_layout(True)
fig.savefig('figures/thermometer_barometer.pdf')
plt.show()
