import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

from burnman.constants import gas_constant, Avogadro
from burnman.eos import electronic, einstein
from burnman import debye
import numpy as np
import os, sys, numpy as np, matplotlib.pyplot as plt
import matplotlib.image as mpimg

Cvel_0 = 2.7
Tel_0 = 7000.

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


fig1 = mpimg.imread('data/Fe_fcc_Cvel_6000_2.5_Wassermann_et_al_1996.png')  # Uncomment these two lines if you want to overlay the plot on a screengrab from SLB2011
plt.imshow(fig1, extent=[0., 6000.0, 0., 2.5*gas_constant], aspect='auto')
    
temperatures=  np.linspace(0., 6000., 101)
C_v = np.empty_like(temperatures)
S = np.empty_like(temperatures)
E = np.empty_like(temperatures)

C_v_check = np.empty_like(temperatures)
C_v_check_2 = np.empty_like(temperatures)


from scipy.constants import physical_constants
r_B = physical_constants['Bohr radius'][0]
V_B = np.power(r_B, 3.)

V0 = 6.97e-6/Avogadro/V_B # in au (bohr^3/atom)
Vs = np.array([48., 55., 60., 65., 70.])
V0overVs = V0/Vs
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
