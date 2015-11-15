import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

print 'See http://www.sciencedirect.com/science/article/pii/S0364591614001023'

import burnman
from burnman import minerals
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()

fper = burnman.minerals.SLB_2011.wuestite()

temperatures = np.linspace(100., 300., 101)
Cps = np.empty_like(temperatures)
Ss = np.empty_like(temperatures)

P = 1.e5
for i, T in enumerate(temperatures):
    fper.set_state(P, T)
    Cps[i] = fper.C_p
    Ss[i] = fper.S

plt.plot(temperatures, Cps)
plt.plot(temperatures, Ss, 'r--')
plt.show()
