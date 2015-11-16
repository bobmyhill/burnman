import os, sys
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

import burnman

'''
Tel = 5700.
n = 2.6
VoverV0s = [0.923, 0.821, 0.726, 0.639, 0.559, 0.487, 0.420]
'''

V0 = 79.73
Tel0 = 6500.
n = 2.7
VoverV0s = [70./V0, 65./V0, 60./V0, 55./V0, 48./V0]


plt.xlim(0., 6000.)
plt.ylim(0., 2.5)
for VoverV0 in VoverV0s:

    Tel = Tel0/np.power(VoverV0, 3./2.)
    temperatures = np.linspace(0., 12000, 101)
    Cv = np.empty_like(temperatures)
    
    for i, T in enumerate(temperatures):
        if T < 0.388247*Tel:
            Cv[i] = (38./3.*n/burnman.constants.gas_constant)*T/Tel
        else:
            Cv[i] = n*(Tel/T)*(Tel/T)*np.exp(Tel/T)/((np.exp(Tel/T) - 1.)*(np.exp(Tel/T) - 1.))



    plt.plot(temperatures, Cv)
plt.show()
