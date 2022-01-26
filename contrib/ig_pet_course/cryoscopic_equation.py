from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt

import burnman
from burnman import equilibrate
from burnman.minerals import HGP_2018_ds633
from collections import Counter


diL = HGP_2018_ds633.diL()
anL = HGP_2018_ds633.anL()
di = HGP_2018_ds633.di()
an = HGP_2018_ds633.an()

# A super simple ideal model
liq = burnman.SolidSolution(name='ideal anL-diL',
                            solution_type='symmetric',
                            endmembers=[[diL, '[Mg]1'],
                                        [anL, '[Ca]1']],
                            energy_interaction=[[-15.e3]])

total = sum(di.formula.values())
fdi = Counter({k: v / total for k, v in di.formula.items()})
total = sum(an.formula.values())
fan = Counter({k: -v / total for k, v in an.formula.items()})

d = fan.copy()
d.update(fdi)
free_compositional_vectors = [d]

assemblage = burnman.Composite([di, diL])
assemblage.set_state(1.e5, 1473.15)
equality_constraints = [('phase_fraction', (di, np.array([0.0]))),
                        ('P', 1.e5)]
sol, prm = equilibrate(di.formula, assemblage,
                       equality_constraints)

T_di_fus = sol.assemblage.temperature
S_di_fus = diL.S - di.S
V_di_fus = diL.V - di.V

liq.set_composition([0.1, 0.9])
composition = liq.formula
assemblage = burnman.Composite([liq, di])

temperatures = np.linspace(T_di_fus - 100., T_di_fus, 101)

equality_constraints = [('phase_fraction', (di, np.array([0.0]))),
                        ('T', temperatures),
                        ('P', 1.e5)]

sols, prm = equilibrate(composition, assemblage,
                        equality_constraints,
                        free_compositional_vectors)

T = [sol.assemblage.temperature for sol in sols]
x = np.array([sol.assemblage.phases[0].molar_fractions[1]
              for sol in sols])


def cryoscopic(xs, Ps, S_fus, V_fus, T_fus, P_fus):
    return ((S_fus*T_fus + V_fus*(Ps - P_fus))
            / (S_fus - burnman.constants.gas_constant*np.log(xs)))

def cryoscopic_2(Ps, Ts):
    G_sol = di.evaluate(['gibbs'], Ps, Ts)[0]
    G_liq = diL.evaluate(['gibbs'], Ps, Ts)[0]
    return np.exp((G_sol - G_liq)
                  /(burnman.constants.gas_constant*Ts))


fig = plt.figure(figsize=(4, 3))
ax = [fig.add_subplot(1, 1, 1)]

ax[0].plot(x, T,
           label='model curve')

ax[0].plot(x, cryoscopic(1.-x, 1.e5, S_di_fus,
                         V_di_fus, T_di_fus, 1.e5),
           label='cryoscopic approximation')


ax[0].set_xlabel('$1 - p_{{di}}$ ($=p_{{an}}$)')
ax[0].set_ylabel('$T$ (K)')
ax[0].legend()
fig.set_tight_layout(True)
fig.savefig('figures/cryoscopic_di_an_melting.pdf')


ax[0].plot(1.-cryoscopic_2(1.e5+0.*temperatures, temperatures),
           temperatures,
           label='cryoscopic approximation (2)')
ax[0].legend()
fig.savefig('figures/cryoscopic_di_an_melting_2.pdf')

plt.show()
