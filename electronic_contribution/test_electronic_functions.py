import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

from burnman.eos import electronic, einstein
from burnman import debye
import numpy as np
import os, sys, numpy as np, matplotlib.pyplot as plt

Cvel_0 = 2.7
Tel_0 = 6500.

'''
# First check einstein
einstein_T = 500.
n = 3.
T = 1000.
dT = 1.
E0 = einstein.entropy(T-0.5*dT, einstein_T, n)
E1 = einstein.entropy(T+0.5*dT, einstein_T, n)
Cv = einstein.heat_capacity_v(T, einstein_T, n)

print Cv, (E1-E0)/dT

debye_T = 500.
n = 3.
T = 1000.
dT = 1.
S0 = debye.entropy(T-0.5*dT, debye_T, n)
S1 = debye.entropy(T+0.5*dT, debye_T, n)
Cv = debye.heat_capacity_v(T, debye_T, n)

print Cv, T*(S1-S0)/dT
'''


temperatures=  np.linspace(0., 6000., 101)
C_v = np.empty_like(temperatures)
S = np.empty_like(temperatures)
E = np.empty_like(temperatures)

C_v_check = np.empty_like(temperatures)
C_v_check_2 = np.empty_like(temperatures)

V0overVs = np.linspace(1.0, 2.0, 6)
for x in V0overVs:
    for i, T in enumerate(temperatures):
        C_v[i] = electronic.heat_capacity_v(T, x, Tel_0, Cvel_0)

        dT = 1.
        S0 = electronic.entropy(T-0.5*dT, x, Tel_0, Cvel_0)
        S1 = electronic.entropy(T+0.5*dT, x, Tel_0, Cvel_0)
        
        # S = \int Cv/T dT
        C_v_check[i] = (T*(S1-S0)/dT)

        dT = 1.
        E0 = electronic.thermal_energy(T-0.5*dT, x, Tel_0, Cvel_0)
        E1 = electronic.thermal_energy(T+0.5*dT, x, Tel_0, Cvel_0)
        
        C_v_check_2[i] = (E1-E0)/dT
        
    plt.plot(temperatures, C_v)
    plt.plot(temperatures, C_v_check, 'r--')
    plt.plot(temperatures, C_v_check_2, 'b.')
plt.show()

for x in V0overVs:
    for i, T in enumerate(temperatures):
        S[i] = electronic.entropy(T, x, Tel_0, Cvel_0)
    plt.plot(temperatures, S)
plt.show()

for x in V0overVs:
    for i, T in enumerate(temperatures):
        E[i] = electronic.thermal_energy(T, x, Tel_0, Cvel_0)
    plt.plot(temperatures, E)
plt.show()
