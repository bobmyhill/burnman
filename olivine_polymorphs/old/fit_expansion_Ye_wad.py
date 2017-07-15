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

mbrS=0.0939
mbratoms=7.
mbrk=[1660.0, 4.2, -0.00242]
mbra=0.0000223 
ds62mbra=0.0000237
mbrV=4.055
Z=8.
nA=6.02214e23
voltoa=1.e24*10.
mbrkfrost=[1462.544, 4.21]
vfrost=4.2206
P=0.0001

# inputs: x[i,j], a[0], a[1], ... a[k]
def fita(T, mbra):

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

with open('Ye_wad_expansion_TVsig.dat') as file:
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
guesses=array([0.0000300])




popt, pcov = curve_fit(fita, T, volumes, guesses, sigma)

print "a0: ", popt[0], "+/-", sqrt(pcov[0][0]), "GPa"

mbra=0.0000216
for i in range(101):
    T=1.+float(i)*20.
    print T, calcV(P, T, mbrS, mbratoms, mbrk, mbra, mbrV)
