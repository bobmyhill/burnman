import numpy as np
from burnman.tools.unitcell import molar_volume_from_unit_cell_volume
import matplotlib.pyplot as plt

from burnman.minerals.HP_2011_ds62 import q
from burnman.minerals.SLB_2011 import quartz

"""
qtz = q()
#qtz.property_modifiers[0][1]['V_D'] = 1.25e-06
#qtz = quartz()

temperatures = np.linspace(10., 1300., 101)
pressures = temperatures * 0. + 1.e5
Qs = temperatures * 0. + 1.e5
V = qtz.evaluate(['V'], pressures, temperatures)[0]

for i, T in enumerate(temperatures):
    qtz.set_state(1.e5, T)
    Qs[i] = qtz.property_modifier_properties[0]['Q']


d = np.loadtxt('data/Carpenter_1998_quartz_unit_cell.dat', unpack=True)

plt.errorbar(d[0], molar_volume_from_unit_cell_volume(d[-2], 3.),
             xerr=d[1], yerr=molar_volume_from_unit_cell_volume(d[-1], 3.),
             fmt=' ', color='b')
plt.plot(temperatures, V)
plt.show()
"""

d = np.loadtxt('data/Lakshtanov_et_al_2007_Cijs_quartz.dat', unpack=True)

for i in range(7):
    plt.scatter(d[0], d[i+2])
plt.show()


TC = d[0]
TK = TC + 273.15
C11, C12, C13, C14, C33, C44 = d[2:8]
nul = C11*0.
C = np.array([[C11, C12, C13, C14, nul, nul],
              [C12, C11, C13, -C14, nul, nul],
              [C13, C13, C33, nul, nul, nul],
              [C14, -C14, nul, C44, nul, nul],
              [nul, nul, nul, nul, C44, C14],
              [nul, nul, nul, nul, C14, 0.5*(C11 - C12)]]).T

#for i in range(6):
#    for j in range(6):
#        plt.scatter(TC, C[:, i, j])

l = []
m = []
for c in C:
    eigenValues, eigenVectors = np.linalg.eig(c)
    eigenValues
    #l.append(eigenValues[np.lexsort(eigenVectors)])
    l.append(eigenValues[np.argsort(eigenValues)])
    m.append(eigenVectors)

l = np.array(l)
m = np.array(m)

for i in range(6):
    plt.scatter(TK, l[:, i])
plt.xlim(0., )
plt.ylim(0., )
#plt.plot(V, Qs*Qs)
plt.show()