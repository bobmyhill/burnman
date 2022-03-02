import matplotlib.pyplot as plt
import numpy as np


def quad(x, x_c, y_c, scale):
    return y_c + np.power((x - x_c)/scale, 2.)


fig = plt.figure(figsize=(6, 3))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

xs = np.linspace(0.3, 0.35, 101)
ax[0].scatter([1./3.], [-1000.])
ax[1].plot(xs, quad(xs, 1./3., -1000., 0.001))


for i in range(2):
    ax[i].set_xlabel('O/(O + Si) (molar)')
    ax[i].set_xlim(0., 1.)
    ax[i].set_ylabel('$\\mathcal{G}$ (kJ/mol)')
    ax[i].set_ylim(-1020., -900.)

fig.set_tight_layout(True)
fig.savefig('pure_phase_vs_solution_SiO2.pdf')
plt.show()
