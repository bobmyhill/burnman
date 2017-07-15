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


mbrS=0.0939 # to iterate manually
mbratoms=7.
Z=8.
nA=6.02214e23
voltoa=1.e24
mbrV=4.053*(nA/Z/voltoa)

def fitV(data, k0):
    P=data[0]
    T=data[1]

    k2=-1.*k1/k0

    theta = (10636.0) / ((mbrS*1000.0 / (mbratoms)) + 6.44) # checked, p.346
    a = (1.0 + k1)/(1.0 + k1 + k0*k2) # checked
    b = (k1/k0) - (k2/(1.0 + k1)) # checked
    c = (1.0 + k1 + k0*k2)/(k1*k1 + k1 - k0*k2) # checked
    
    u = theta / T # checked
    uzero = theta/T0 # checked
    ksi0 = uzero**2*exp(uzero)/((exp(uzero)-1.0)**2) # checked (eq. 10-11)
    d = mbra*k0/ksi0 # checked (eq. 11)
    Ptherm = d*(theta/(exp(u)-1.0) - theta/(exp(uzero)-1.0)) # checked (eq. 11)
    
    
    mbrvolume = mbrV*(1.0-a*(1.0-(1.0+b*(P-Ptherm))**(-1.0*c))) # checked (eq. 12)

    return mbrvolume

with open('Katsura_et_al_wad_volumes.dat') as file:
    arr = [[float(digit) for digit in line.split()] for line in file]
    data = array(arr)

# Data
P=data.T[1]
T=data.T[0]
volumes=data.T[3]*538.49

# Uncertainties on volume data
sigma=data.T[4]
pt=array([P,T]) # P in GPa


# Initial guess.
k1=4.2
mbrV=538.28 # Trots et al.
mbra=2.34895e-05
guesses=array([110.])




popt, pcov = curve_fit(fitV, pt, volumes, guesses, sigma)

print "k0: ", popt[0], "+/-", sqrt(pcov[0][0]), "GPa"

print

print "Covariance matrix:"
print pcov

mbrk=array([popt[0], k1, -1.*k1/popt[0]])
print mbrk
sumsq=0
sumsig=0
sumdev=0
for i in range(len(P)):
    Pi=P[i]
    Ti=T[i]
    Vcalc=calcV(Pi, Ti, mbrS, mbratoms, mbrk, mbra, mbrV)
    sumsq=sumsq+(Vcalc-volumes[i])**2
    sumsig=sumsig+sigma[i]
    sumdev=sumdev+sqrt((Vcalc-volumes[i])**2)
    #print Pi, Ti, volumes[i], sigma[i], Vcalc, sqrt((Vcalc-volumes[i])**2)

RMSD=sqrt(sumsq/len(P))
RMSU=sqrt(sumsig/len(P))
print
print "Root mean square deviation:", RMSD, "cm^3/mol"
print "Root mean square uncertainty:", RMSU, "cm^3/mol"
print
print "Average uncertainty:", sumsig/len(P), "cm^3/mol"
print "Average error:      ", sumdev/len(P), "cm^3/mol"
