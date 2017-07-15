#!/usr/bin/python

from numpy import exp, linspace, random, array, sqrt
from scipy.optimize import curve_fit

T0=298.15 # K

def calcV(P, T, mbrS, mbratoms, mbrk, mbra, mbrV):

    theta = (10636.0) / ((mbrS*1000.0 / (mbratoms)) + 6.44) # checked, p.346
    a = (1.0 + mbrk[1])/(1.0 + mbrk[1] + mbrk[0]*mbrk[2]) # checked
    b = (mbrk[1]/mbrk[0]) - (mbrk[2]/(1.0 + mbrk[1])) # checked
    c = (1.0 + mbrk[1] + mbrk[0]*mbrk[2])/(mbrk[1]*mbrk[1] + mbrk[1] - mbrk[0]*mbrk[2]) # checked
    
    u = theta / T # checked
    uzero = theta/T0 # checked
    ksi0 = uzero**2*exp(uzero)/((exp(uzero)-1.0)**2) # checked (eq. 10-11)
    d = mbra*mbrk[0]/ksi0 # checked (eq. 11)
    Ptherm = d*(theta/(exp(u)-1.0) - theta/(exp(uzero)-1.0)) # checked (eq. 11)
    
    
    mbrvolume = mbrV*(1.0-a*(1.0-(1.0+b*(P-Ptherm))**(-1.0*c))) # checked (eq. 12)

    return mbrvolume

mbrS=0.0900
mbratoms=7.
mbrk=[1781.0, 4.35, -0.0024]
mbra=0.0000251
mbrV=3.9490
Z=8.
nA=6.02214e23
voltoa=1.e24*10.
vinoue=536./voltoa*nA/8
P=0.0001

# inputs: x[i,j], a[0], a[1], ... a[k]
def fita(T, mbra, mbrV):

    theta = (10636.0) / ((mbrS*1000.0 / (mbratoms)) + 6.44) # checked, p.346
    a = (1.0 + mbrk[1])/(1.0 + mbrk[1] + mbrk[0]*mbrk[2]) # checked
    b = (mbrk[1]/mbrk[0]) - (mbrk[2]/(1.0 + mbrk[1])) # checked
    c = (1.0 + mbrk[1] + mbrk[0]*mbrk[2])/(mbrk[1]*mbrk[1] + mbrk[1] - mbrk[0]*mbrk[2]) # checked
    
    u = theta / T # checked
    uzero = theta/T0 # checked
    ksi0 = uzero**2*exp(uzero)/((exp(uzero)-1.0)**2) # checked (eq. 10-11)
    d = mbra*mbrk[0]/ksi0 # checked (eq. 11)
    Ptherm = d*(theta/(exp(u)-1.0) - theta/(exp(uzero)-1.0)) # checked (eq. 11)
    
    
    mbrvolume = mbrV*(1.0-a*(1.0-(1.0+b*(P-Ptherm))**(-1.0*c))) # checked (eq. 12)

    return mbrvolume

with open('Ye_mrw3_expansion_TVsig.dat') as file:
    arr = [[float(digit) for digit in line.split()] for line in file]
    data = array(arr)

# Data
T=data.T[0]
volumes=data.T[1]*(nA/Z/voltoa)

# Uncertainties on volume data
sigma=data.T[2]*(nA/Z/voltoa)
TV=array([T,volumes]) # P in GPa

print TV.T
# Initial guess.
guesses=array([0.0000300, 3.9])




popt, pcov = curve_fit(fita, T, volumes, guesses, sigma)

print "a0: ", popt[0], "+/-", sqrt(pcov[0][0]), "GPa"
print "V0: ", popt[1], "+/-", sqrt(pcov[1][1]), "kJ/kbar"

mbra=popt[0]
mbrV=popt[1]
for i in range(101):
    T=1.+float(i)*20.
    print T, calcV(P, T, mbrS, mbratoms, mbrk, mbra, mbrV)
