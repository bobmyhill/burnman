import numpy as np
import matplotlib.pyplot as plt
import burnman 


ol = burnman.minerals.SLB_2011.mg_fe_olivine()

T = 300.
ol.set_state(1.e5, T)

xs = np.linspace(0., 1., 101)
Exs = np.empty_like(xs)
Sxs = np.empty_like(xs)
Gmechxs = np.empty_like(xs)

gammas = np.empty_like(xs)

fo_gibbs = ol.endmembers[0][0].gibbs
fa_gibbs = ol.endmembers[1][0].gibbs


for i, x in enumerate(xs):
    ol.set_composition([x, 1.-x])
    Exs[i] = ol.excess_enthalpy
    Sxs[i] = ol.excess_entropy
    Gmechxs[i] = fo_gibbs*x + fa_gibbs*(1. - x)

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


figs = [plt.figure(figsize=(4, 3)) for i in range(4)]
ax3 = [figs[i-1].add_subplot(1, 1, 1) for i in range(1, 5)]
for T in [150, 300, 450, 600]:
    ol.set_state(1.e5, T)

    xs = np.linspace(0., 1., 101)
    Exs = np.empty_like(xs)
    Sxs = np.empty_like(xs)
    Gmechxs = np.empty_like(xs)

    gammas = np.empty_like(xs)

    fo_gibbs = ol.endmembers[0][0].gibbs
    fa_gibbs = ol.endmembers[1][0].gibbs

    for i, x in enumerate(xs):
        ol.set_composition([x, 1.-x])
        Exs[i] = ol.excess_enthalpy
        Sxs[i] = ol.excess_entropy
        Gmechxs[i] = fo_gibbs*x + (fa_gibbs)*(1. - x)

        gammas[i] = ol.activity_coefficients[0]


    ax3[0].plot(xs, Gmechxs/1000.)
    ax3[1].plot(xs, -T*Sxs/1000.)
    ax3[2].plot(xs, Exs/1000.)
    ax3[3].plot(xs, (-T*Sxs + Exs)/1000., label=f'{T} K')


for i in range(4):
    ax3[i].set_xlabel('p(fo)')
ax3[0].set_ylabel('$\mathcal{{G}}_{{mech}}$ (kJ/mol)')
ax3[1].set_ylabel('$\mathcal{{G}}_{{conf}}$ (kJ/mol)')
ax3[2].set_ylabel('$\mathcal{{G}}_{{xs}}$ (kJ/mol)')
ax3[3].set_ylabel('$\mathcal{{G}}_{{conf}}+\mathcal{{G}}_{{xs}}$ (kJ/mol)')

ax3[3].legend()

for i in range(4):
    figs[i].set_tight_layout(True)
    figs[i].savefig(f'nonideal_components_{i}.pdf')

plt.show()

