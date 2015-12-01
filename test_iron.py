import burnman
import os, sys, numpy as np, matplotlib.pyplot as plt
iron = burnman.minerals.HP_2011_ds62.iron()

P = 1.e9
temperatures = np.linspace(300., 1400., 101)
Ss = np.empty_like(temperatures)
Cps = np.empty_like(temperatures)
K_Ts = np.empty_like(temperatures)
volumes = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    iron.set_state(P, T)
    Ss[i] = iron.S
    Cps[i] = iron.C_p
    volumes[i] = iron.V
    K_Ts[i] = iron.K_T


    
    
plt.plot(temperatures, Ss, label='HP')
plt.title('Enironpies')
plt.legend(loc='lower right')
plt.show()

plt.plot(temperatures, Cps, label='HP')
plt.title('Heat capacities')
plt.legend(loc='lower right')
plt.ylim(0., 100.)
plt.show()

plt.plot(temperatures, volumes, label='HP')
plt.title('Volumes')
plt.legend(loc='lower right')
plt.show()

plt.plot(temperatures, K_Ts/1.e9, label='HP')
plt.title('K_T')
plt.legend(loc='lower right')
plt.show()
