import burnman
import numpy as np
import matplotlib.pyplot as plt

per = burnman.minerals.SLB_2011.periclase()

temperatures = np.linspace(1., 1000., 11)
Ps = []
alphas = []
for i, T in enumerate(temperatures):
    per.set_state_with_volume(per.params['V_0'], T)
    Ps.append(per.pressure)
    alphas.append(per.alpha)


plt.plot(Ps, alphas)
plt.show()