import numpy as np
import matplotlib.pyplot as plt

# in celsius / GPa
max_pts = [['cold seal', 800, 0.7],
           ['IHPV', 1300, 1.3],
           ['piston cylinder', 2050, 5],
           ['multi-anvil (WC)', 2900., 25],
           ['multi-anvil (sintered diamond)', 3100., 65.],
           ['diamond anvil cell', 7000., 770.]]

fig = plt.figure(figsize=(6, 4))
ax = [fig.add_subplot(1, 1, 1)]

ax[0].set_yscale('log')

for name, T, P in max_pts[::-1]:
    ax[0].fill_between([0., T],
                       [1.e8, 1.e8],
                       [P*1.e9, P*1.e9],
                       color='white')
    ax[0].fill_between([0., T],
                       [1.e8, 1.e8],
                       [P*1.e9, P*1.e9],
                       alpha=0.4)
    ax[0].text(T, P*1.e9, name,
               horizontalalignment='right',
               verticalalignment='top',
               rotation=90)

ax[0].set_xlabel('Temperature ($^{\circ}$C)')
ax[0].set_ylabel('Pressure (Pa)')
ax[0].set_xlim(0.,)
ax[0].set_ylim(1.e8,)
fig.savefig('figures/experimental_PT_range.pdf')
plt.show()