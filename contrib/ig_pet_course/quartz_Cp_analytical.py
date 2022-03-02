import matplotlib.pyplot as plt
import numpy as np
from burnman.minerals.HP_2011_ds62 import q 
from scipy.integrate import quad


qtz = q()

qtz.set_state(1.e5, 800)
V0 = qtz.V
print(qtz.V)

qtz.set_state(1.e5 + 10000., 800)

print((qtz.V - V0)/10000.)



qtz.property_modifiers = []
a, b, c, d = qtz.params['Cp']
S_0 = qtz.params['S_0']
print(S_0)
def Cp(T):
    return a + b*T + c/T/T + d/np.sqrt(T)

def CpoverT(T):
    return Cp(T)/T

T0 = 298.15
T1 = 800.

intCp = quad(Cp, T0, T1)
intCpoverT = quad(CpoverT, T0, T1)

print(intCp[0] + T0*S_0 - T1*(S_0 + intCpoverT[0]))

G, S, H = qtz.evaluate(['gibbs', 'S', 'H'],
                 [1.e5, 1.e5],
                 [T0, T1])

print(G[1] - G[0])