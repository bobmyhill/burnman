import numpy as np
import matplotlib.pyplot as plt
from burnman.minerals.HP_2011_ds62 import q
from scipy.integrate import cumtrapz

qtz = q()
qtz.set_state(1.e5, 300.)
print(qtz.gibbs)

T = np.array([1.e-10, 10., 15, 20., 25, 30., 40, 50,
              60., 70., 80., 90., 100.,
              120., 140., 160., 180., 200.,
              220., 240., 260., 280., 300.])

Cp = np.array([0.,
               0.0007,
               0.0040,
               0.0113,
               0.0221,
               0.0353,
               0.0653,
               0.0969,
               0.129,
               0.162,
               0.195,
               0.228,
               0.261,
               0.325,
               0.385,
               0.441,
               0.494,
               0.543,
               0.588,
               0.631,
               0.671,
               0.709,
               0.745])




fig = plt.figure(figsize=(6, 3))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

ax[0].plot(T, Cp*1000.*qtz.molar_mass)
ax[1].plot(T, cumtrapz(Cp*1000.*qtz.molar_mass/T, T, initial=0.))


ax[0].set_xlabel('Temperature (K)')
ax[0].set_ylabel('Heat capacity (J/K/mol)')
ax[1].set_xlabel('Temperature (K)')
ax[1].set_ylabel('Entropy  (J/K/mol)')

#T = np.linspace(300., 1000.)
#ax[1].plot(T, qtz.evaluate(['S'], T*0. + 1.e5, T)[0])
fig.set_tight_layout(True)

fig.savefig('LT_qtz_Cp.pdf')
plt.show()