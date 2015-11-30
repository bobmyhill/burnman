import burnman
import os, sys, numpy as np, matplotlib.pyplot as plt
tro = burnman.minerals.HP_2011_ds62.tro()
tro2 = burnman.minerals.HP_2011_ds62.tro2()

P = 1.e9
temperatures = np.linspace(298.15, 1000., 101)
Ss = np.empty_like(temperatures)
Cps = np.empty_like(temperatures)
volumes = np.empty_like(temperatures)
Ss2 = np.empty_like(temperatures)
Cps2 = np.empty_like(temperatures)
volumes2 = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    tro.set_state(P, T)
    Ss[i] = tro.S
    Cps[i] = tro.C_p
    volumes[i] = tro.V
    tro2.set_state(P, T)
    Ss2[i] = tro2.S
    Cps2[i] = tro2.C_p
    volumes2[i] = tro2.V
plt.plot(temperatures, Ss, label='SLB')
plt.plot(temperatures, Ss2, label='HP')
plt.title('Entropies')
plt.legend(loc='lower right')
plt.show()

plt.plot(temperatures, Cps, label='SLB')
plt.plot(temperatures, Cps2, label='HP')
plt.title('Heat capacities')
plt.legend(loc='lower right')
plt.show()

plt.plot(temperatures, volumes, label='SLB')
plt.plot(temperatures, volumes2, label='HP')
plt.title('Volumes')
plt.legend(loc='lower right')
plt.show()
