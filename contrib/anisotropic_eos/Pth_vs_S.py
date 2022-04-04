import burnman
import matplotlib.pyplot as plt
import numpy as np

per = burnman.minerals.SLB_2011.forsterite()

temperatures = np.linspace(10., 2000., 501)
pressures = 1.e9 + 0.*temperatures 

Ss, Vs, Cps, Gs = per.evaluate(['S', 'V', 'C_p', 'K_T'], pressures, temperatures)

static_pressures = np.copy(pressures)
for i, V in enumerate(Vs):
    static_pressures[i] = per.method.pressure(0., V, per.params)    

Pth = pressures - static_pressures
plt.plot(temperatures, (Gs[0]-Gs)/(Gs[0]-Gs[-1]))
plt.plot(temperatures, (Pth[0]-Pth)/(Pth[0]-Pth[-1]))
plt.show()