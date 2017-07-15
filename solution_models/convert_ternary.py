import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(fname='data/Fe_FeSi_FeO_ternary_Iyengar_Philbrook_1973.dat', unpack=True)

boundaries = np.array([[0., 0.223, 0.166667, 0.],
                       [0., 0., 0.3333333, 0.]])
    

def convert(data):
    x, y = data
    
    xSi = y
    xO2 = (x - y/2.)
    xFe = 1. - xSi - xO2
    
    # Convert to mol % Fe, FeSi0.5 and FeO
    mFe = 55.845
    mO = 15.9994
    mSi = 28.000
    
    molFe=xFe/mFe
    molO = xO2/mO
    molSi = xSi/mSi

    molFeO = molO
    molFeSi = 2.*molSi
    molFe = molFe - molFeO - 0.5*molFeSi

    moltotal = (molFeO + molFe + molFeSi) / 100.
    molFeO = molFeO / moltotal
    molFe = molFe / moltotal
    molFeSi = molFeSi / moltotal
    
    return molFe, molFeSi, molFeO

molFe, molFeSi, molFeO = convert(data)
plt.plot(molFe, molFeSi, marker='o')

molFe, molFeSi, molFeO = convert(boundaries)
plt.plot(molFe, molFeSi)

plt.xlim(0., 100.)
plt.ylim(0., 100.)
plt.xlabel('mol % Fe')
plt.ylabel('mol % Fe0.5Si0.5')
plt.title('Fe - Fe0.5Si0.5 - FeO from *sketch* ternary of Iyengar and Philbrook (1973)')
plt.show()
