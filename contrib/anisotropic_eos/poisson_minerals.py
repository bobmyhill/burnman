from burnman.minerals import SLB_2011
import matplotlib.pyplot as plt
import numpy as np

fo = SLB_2011.forsterite()
per = SLB_2011.periclase()
py = SLB_2011.pyrope()
gr = SLB_2011.grossular()
alm = SLB_2011.almandine()
mpv = SLB_2011.mg_perovskite()

cs = ['red', 'blue', 'green', 'orange', 'purple']
lst = ['-', '--', ':']
for i, T in enumerate([300., 1000., 2000.]):
    for j, m in enumerate([fo, per]):
        pressures = np.linspace(1.e5, 150.e9, 101)
        temperatures = T + pressures*0.
        V, KT, G = m.evaluate(['V', 'K_T', 'G'], pressures, temperatures)

        poisson = (3.*KT - 2.*G)/(2.*(3.*KT + G))
        #plt.plot(np.log(V/V[0]), poisson, label=m.name)
        plt.plot(V/V[0], poisson, label=f'{m.name} {T} K', linestyle=lst[i], c=cs[j])

plt.legend()
plt.show()
