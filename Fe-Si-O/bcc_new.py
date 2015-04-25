# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman import tools
from burnman.mineral import Mineral
from burnman.chemicalpotentials import *
from burnman import constants

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from fitting_functions import *
from scipy import optimize

R=constants.gas_constant


'''
# Order-disorder model for FeSi

Notes:
The bcc phase is pretty complicated, with structural changes:
A2->B2->D03

The D03 structure appears around 25 at% Si (as [Fe]0.5[Si]0.25[Si]0.25) at low P and T, probably becoming unstable at about 60 GPa. It has a fairly limited stability range above about 20 GPa, especially at temperatures > 1500 C. At ambient pressure, above the Curie temperature the B2-D03 transition is second order, but below this temperature (~700 C at 10 at.% Si) B2 and D03 structures can coexist, due to interactions between magnetic and chemical ordering.

To describe the D03 structure as well as B2 requires three sites, four endmembers and two order parameters:
Fe    (A2)  [Fe]0.5[Fe]0.25[Fe]0.25 
FeSi  (B2)  [Fe]0.5[Fe]0.25[Si]0.25 
Fe3Si (D03) [Fe]0.5[Si]0.25[Si]0.25
Si    (A2)  [Si]0.5[Si]0.25[Si]0.25

With only B2 to worry about, there are two sites and three endmembers:
Fe    (A2)  [Fe]0.5[Fe]0.5
FeSi  (B2)  [Fe]0.5[Si]0.5
Si    (A2)  [Si]0.5[Si]0.5


Another complication is the magnetism in the phase. Lacaze and Sundman (1991) proposed a quadratic compositional dependence of the Curie temperature and linear dependence on the magnetic moment (i.e. no effect of ordering).

'''

'''

A2-B2 only...

'''

def eqm_order(Q, X, T, m, n, DeltaH, W): # Wab, Wao, Wbo

    A = DeltaH + (m+n)*W[1] - n*W[0]
    B = 2./(m+n)*(-m*m*W[1] + m*n*(W[0] - W[1] - W[2]) - n*n*W[2])
    C = m*(W[2] - W[1] - W[0]) + n*(W[2] - W[1] + W[0])

    pa = 1. - X - m/(m+n)*Q
    pb = X - n/(m+n)*Q

    Kd=(1.-pa)*(1.-pb)/(pa*pb)
    return A + B*Q + C*X + m*n*R*T*np.log(Kd)

# Test diopside jadeite
X=0.5
m=1.
n=1.
DeltaH=-6000.
W=[26000., 16000., 16000.]

temperatures=np.linspace(373.15, 1373.15, 101)
order=np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    order[i]=optimize.fsolve(eqm_order, 0.999*X*2, args=(X, T, m, n, DeltaH, W))


plt.plot( temperatures, order, linewidth=1, label='order')
plt.title('FeSi ordering')
plt.xlabel("Temperature")
plt.ylabel("Order")
plt.legend(loc='upper right')
plt.show()
