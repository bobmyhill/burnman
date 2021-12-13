import numpy as np
import matplotlib.pyplot as plt
from burnman.minerals import HGP_2018_ds633, HP_2011_fluids
from burnman import Composite
from burnman.constants import gas_constant

fa = HGP_2018_ds633.fa()
mt = HGP_2018_ds633.mt()
q = HGP_2018_ds633.q()
O2 = HP_2011_fluids.O2()


FMQ = Composite([fa, mt, q], [0.2, 0.3, 0.5])

P = 1.e5
Pstd = 1.e5

temperatures = np.linspace(700., 1500., 101)
log10fO2 = np.empty_like(temperatures)

for i, T in enumerate(temperatures):

    FMQ.set_state(P, T)
    O2.set_state(Pstd, T)
    muO2_FMQ = FMQ.chemical_potential([{'O': 2.}])

    log10fO2[i] = ((muO2_FMQ - O2.gibbs)
                   / (gas_constant*T*np.log(10)))

fig, ax1 = plt.subplots(figsize=(4, 3))

ax1.set_xlabel('Temperature (K)')
ax1.set_ylabel('$\\log_{{10}}f^*_{{O_2}}$')
ax1.tick_params(axis='y')

# instantiate a second axes that shares the same x-axis
ax2 = ax1.twiny()
ln, = ax2.plot(temperatures-273.15, log10fO2, label='FMQ')

ax2.set_xlabel('Temperature ($^{{\\circ}}$C)')
ax2.tick_params(axis='y')

ax2.text(450., -10, 'more oxidised than FMQ', color=ln.get_color())
ax2.text(720., -22, 'less oxidised than FMQ', color=ln.get_color())

ax2.legend(loc='lower right')
fig.tight_layout()  # otherwise the right y-label is slightly clipped


ax1.set_xlim(temperatures[0], temperatures[-1])
ax2.set_xlim(temperatures[0]-273.15, temperatures[-1]-273.15)
fig.savefig('figures/fO2-FMQ.pdf')
plt.show()
