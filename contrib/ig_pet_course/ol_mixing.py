import numpy as np
import matplotlib.pyplot as plt
import burnman 


ol = burnman.minerals.SLB_2011.mg_fe_olivine()

ol.set_state(1.e5, 1000.)

xs = np.linspace(0., 1., 101)
Exs = np.empty_like(xs)
gammas = np.empty_like(xs)


for i, x in enumerate(xs):
    ol.set_composition([x, 1.-x])
    Exs[i] = ol.excess_enthalpy
    gammas[i] = ol.activity_coefficients[0]

fig = plt.figure(figsize=(4, 3))
ax = [fig.add_subplot(1, 1, 1)]
ax[0].plot(xs, Exs/1000.)

fig2 = plt.figure(figsize=(4, 3))
ax2 = [fig2.add_subplot(1, 1, 1)]
ax2[0].plot(xs, gammas, label='fo')
ax2[0].plot(1.-xs, gammas, label='fa')

ax[0].set_xlabel('x(fo)')
ax[0].set_ylabel('Excess energy (kJ/mol)')

ax2[0].set_xlabel('x(fo)')
ax2[0].set_ylabel('$\\gamma$')
ax2[0].legend()

fig.set_tight_layout(True)
fig2.set_tight_layout(True)

fig.savefig('ol_excess_energy.pdf')
fig2.savefig('ol_gammas.pdf')

plt.show()

