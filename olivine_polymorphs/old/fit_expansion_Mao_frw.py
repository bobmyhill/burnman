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

mbrS=0.141
mbratoms=7.
mbrk=[1977.0, 4.92, -0.0025]
mbra=0.0000222
mbrV=4.198
Z=8.
nA=6.02214e23
voltoa=1.e24*10.
P=0.0001

theta = (10636.0) / ((mbrS*1000.0 / (mbratoms)) + 6.44)
print 'theta:', theta

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

# inputs: x[i,j], a[0], a[1], ... a[k]
def fitlina(T, y, x):

    mbrvolume=x*(T-8.) + y

    return mbrvolume

with open('Mao_et_al_alpha_Fe2SiO4.dat') as file:
    arr = [[float(digit) for digit in line.split()] for line in file]
    data = array(arr)

# Data
T=data.T[0]-273.15
volumes=data.T[1]

# Uncertainties on volume data
sigma=data.T[2]
TV=array([T,volumes])

# Initial guess.
guesses=array([0.001, 23])

popt, pcov = curve_fit(fitlina, T, volumes, guesses, sigma)

print "V8: ", popt[0], "+/-", sqrt(pcov[0][0]), ""
print "a0: ", popt[1], "+/-", sqrt(pcov[1][1]), "GPa"

print 'That V8 is not zero shows that Mao et al fixed their curve to pass through a volume of 4.203 cm^3/mol at 25 degrees C.'
print 'This gives us a point at which to convert the data in Figure 1 back into volumes.'

V25=4.203
V8=V25/(23.e-6*(25-8)+1)
print 'V8:', V8

for i in range(len(T)):
    volumes[i]=V8 + volumes[i]*V8
    sigma[i]=V8*sigma[i]

T=data.T[0]
TV=array([T,volumes])
print TV.T
# Initial guess.
guesses=array([0.000023, 4.202])
popt, pcov = curve_fit(fita, T, volumes, guesses, sigma)

print "a0:", popt[0], "+/-", sqrt(pcov[0][0]), ""
print "V0:", popt[1], "+/-", sqrt(pcov[1][1]), ""
mbra=popt[0]
mbrV=popt[1]
for i in range(101):
    T=1.+float(i)*20.
    print T, calcV(P, T, mbrS, mbratoms, mbrk, mbra, mbrV)

