
Pcstar = 49.
Pc = 99.7
l1 = -8.
l2 = 24.62
l3 = 17.
l4 = 20.
l6 = 20.
b = 10.94
a = -0.04856

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, root
import matplotlib.image as mpimg
from carpenter_functions import Cij as Cij_orig
from stishovite_isotropic import make_phase
# from burnman.tools.eos import check_anisotropic_eos_consistency
# from burnman.minerals.SLB_2011 import stishovite
# from burnman.minerals.HP_2011_ds62 import stv as stishovite_HP




def Cij0(pressure):
    C110 = 578 + 5.38 * pressure
    C330 = 776 + 4.94 * pressure
    C120 = 86 + 5.38 * pressure
    C130 = 191 + 2.72 * pressure
    C440 = 252 + 1.88 * pressure
    C660 = 323 + 3.10 * pressure

    return np.array([C110, C110, C330, C120, C130, C130, C440, C440, C660])

def Cij_matrix(pressure):
    C110, _, C330, C120, C130, _, C440, _, C660 = Cij0(pressure)
    return np.array([[C110, C120, C130, 0., 0., 0.],
                    [C120, C110, C130, 0., 0., 0.],
                    [C130, C130, C330, 0., 0., 0.],
                    [0., 0., 0., C440, 0., 0.],
                    [0., 0., 0., 0., C440, 0.],
                    [0., 0., 0., 0., 0., C660]])

def strains(Q, pressure):
    C110, _, C330, C120, C130, _, C440, _, C660 = Cij0(pressure)
    
    ea = l2 / (0.5*(C110 - C120))*Q/2.
    eb = l1/(0.5*(C110 - C120))*Q*Q/2.
    e1 = - ea - eb
    e2 = ea - eb
    e3 = - l3/C330*Q*Q
    
    return np.array([e1, e2, e3, 0., 0., 0.])

Q = 0.
P = 0.

C = Cij_matrix(P)
e0 = strains(Q - 1.e-5, P)
e1 = strains(Q + 1.e-5, P)
print(C)
print(-C.dot((e1 - e0))/2.e-5)

print(2.*l1*Q + l2)