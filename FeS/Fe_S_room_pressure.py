import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

g0 = [-104888.10, 0.338] # 0
gx0 = [[0., 0.],
       [35043.32, -9.880], # 10
       [23972.27, 0.], # 20
       [30436.82, 0.], # 30
       [0., 0.]]
g0x = [[0., 0.],
       [8626.26, 0.], # 01
       [72954.29, -26.178], # 02
       [0., 0.],
       [25106., 0.]] # 04

def Gex(XFeS, XS, T):
    XSS = XS - XFeS[0]/2.
    XFeFe = 1. - XS - XFeS[0]/2.
     
    deltaGFeS = g0[0] + T*g0[1]

    for i, prm in enumerate(gx0):
        deltaGFeS = deltaGFeS + np.power(XFeFe, i)*(prm[0] + T*prm[1])
    for i, prm in enumerate(g0x):
        deltaGFeS = deltaGFeS + np.power(XSS, i)*(prm[0] + T*prm[1])
    Sconf = -8.3145*(XSS*np.log(XSS) 
                     + XFeFe*np.log(XFeFe)
                     + XFeS[0]*np.log(XFeS[0]))
 
    #print XSS, XFeFe, XFeS[0]
    #print XFeS[0]
    return XFeS[0]/2.*deltaGFeS - T*Sconf


print minimize(Gex, [0.7], args=(0.4, 1600.), bounds=[(0.,0.79999),], method='TNC').x[0]

xFeSs = np.linspace(0.0001, 0.9999, 101)
Gs = np.empty_like(xFeSs)
for i, XFeS in enumerate(xFeSs):
    Gs[i] = Gex([XFeS], 0.5, 1600.)

plt.plot(xFeSs, Gs)
plt.show()


def Gex_nonconf_simple(XFeS, T):
    XFeFe = 1. - XFeS
     
    deltaGFeS = 0.

    for i, prm in enumerate(gx0):
        deltaGFeS = deltaGFeS + np.power(XFeFe, i)*(prm[0] + T*prm[1])

    return XFeS/2.*deltaGFeS


xFeSs = np.linspace(0.0001, 0.9999, 101)
Gs = np.empty_like(xFeSs)
for i, XFeS in enumerate(xFeSs):
    Gs[i] = Gex_nonconf_simple(XFeS, 1600.)
plt.plot(xFeSs, Gs)
plt.show()
