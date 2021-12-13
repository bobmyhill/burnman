import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(4, 3))
ax = [fig.add_subplot(1, 1, 1)]


def S(x):
    return 8.31446*(x*np.log(x) + (1. - x)*np.log(1. - x))


def quad(x, x_c, y_c, scale):
    return y_c + np.power((x - x_c)/scale, 2.)


xs = np.linspace(0., 0.02, 101)
ax[0].plot(xs, quad(xs, 0.005, -25., 0.002),
           label='"pure" phase A')

xs = np.linspace(0.98, 1, 101)
ax[0].plot(xs, quad(xs, 0.995, 5., 0.002),
           label='"pure" phase B')

xs = np.linspace(0.000001, 0.999999, 101)

ax[0].plot(xs, 8.31446*1000*S(xs)/1000. - 10. + 30.*xs,
           label="solution C")


ax[0].set_xlim(0., 1.)
ax[0].set_ylim(-50., 30.)
ax[0].set_xlabel('Composition')
ax[0].set_ylabel('Gibbs energy (kJ/mol)')
ax[0].legend()
fig.set_tight_layout(True)
fig.savefig('pure_and_solution_phases.pdf')
plt.show()
