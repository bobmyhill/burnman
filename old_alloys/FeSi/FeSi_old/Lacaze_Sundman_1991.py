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

def std_Fe(T):
    if T < 1811.:
        HSERFe= 1224.83 + 124.134*T - 23.5143*T*np.log(T) - 0.00439752*T*T - 5.89269e-8*T*T*T + 77358.5/T
    else:
        HSERFe= -25384.451 + 299.31255*T - 46*T*np.log(T) + 2.2960305e31*np.power(T,-9)
    return HSERFe


def std_Si(T):
    if T < 1687.:
        HSERSi = -8162.61 + 137.227*T - 22.8318*T*np.log(T) - 0.00191129*T*T - 3.55178e-9*T*T*T + 176667./T 
    else:
        HSERSi = -9457.64 + 167.272*T - 27.196*T*np.log(T) - 4.20369e30*np.power(T,-9)
    return HSERSi



def G_bcc(site_fractions, T):
    FeA, FeB, SiA, SiB = site_fractions
    a1=0.5
    a2=0.5
    L0=-27809+11.62*T
    L1=-11544.
    L2=3890.
    xFe=a1*FeA + a2*FeB
    xSi=a1*SiA + a2*SiB
    GFeFe=std_Fe(T)
    GFeSi=-1260.*R + L0
    GSiSi=std_Si(T)

    Gmech0=FeA*FeB*GFeFe + FeA*SiB*GFeSi + SiA*FeB*GFeSi + SiA*SiB*GSiSi

    Gmech=FeB*GFeFe + SiA*GSiSi + (FeA-FeB)*GFeSi

    print Gmech0, Gmech, site_fractions, GFeFe, GSiSi, GFeSi

    Gmix=a1*R*T*(FeA*np.log(FeA) + SiA*np.log(SiA)) + a2*R*T*(FeB*np.log(FeB) + SiB*np.log(SiB))
    Gex=xFe*xSi*(4.*L0 + 8*(xFe-xSi)*L1 + 16*(xFe-xSi)*(xFe-xSi)*L2)
    Gmag=0. # not implemented yet...

    return Gmech + Gmix + Gex + Gmag 

print G_bcc([1., 1.e-12, 1.e-12, 1.], 400.)



def gibbs_at_order(order_parameter, xSi, T):
    # Find value of order_parameter FeA-FeB which minimises G
    # FeA+FeB = (1-xSi)*2
    # Therefore 
    FeA=1. - xSi + 0.5*order_parameter[0]
    FeB=1. - xSi - 0.5*order_parameter[0]
    SiA=1. - FeA
    SiB=1. - FeB
    site_fractions=[FeA, FeB, SiA, SiB]
    #print site_fractions
    return G_bcc(site_fractions, T)

temperatures=np.linspace(300., 1600., 1000)
order=np.empty_like(temperatures)
for i,T in enumerate(temperatures):
    order[i]=optimize.minimize(gibbs_at_order, 0.999999, method='SLSQP', bounds=((0., 1.0),), args=(0.5, T)).x[0]


plt.plot( temperatures, order, linewidth=1, label='order')
plt.title('FeSi ordering')
plt.xlabel("Temperature")
plt.ylabel("Order")
plt.legend(loc='upper right')
plt.show()
