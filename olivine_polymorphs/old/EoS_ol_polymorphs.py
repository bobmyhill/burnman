#!/usr/bin/python
from sympy import Symbol
from sympy.solvers import nsolve
from numpy import exp 
T0=298.15 # K

print 'Olivine polymorph models'
print 'Equations of state from ds62 and Frost (2003)'
print ''

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


def calcVbm(P, mbrk, mbrV):
    v = Symbol('v')
    return nsolve(P-(1.5*mbrk[0]*((mbrV/v)**(7./3.)-(mbrV/v)**(5./3.))*(1+0.75*(mbrk[1]-4.)*((mbrV/v)**(2./3.)-1.))), v, (mbrV,))

def calck(P, T, mbrS, mbratoms, mbrk, mbra, mbrV):

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

    return mbrk[0]*(1+b*(P-Ptherm))*(a+(1-a)*(1+b*(P-Ptherm))**(1.0*c))

P=0.0001 # 1 bar = 0.1 GPa / 1000
T=1673.15 
Ptriple=12.87 # GPa
print 'FORSTERITE (fo):'
mbrS=0.0951
mbratoms=7.
mbrk=[1285.0, 3.84, -0.003]
mbra=0.0000280
ds62mbra=0.0000285
mbrV=4.3660
Z=4.
nA=6.02214e23
voltoa=1.e24*10.
mbrkfrost=[957.0, 4.6]
vfrost=4.6053

print 'Volume (Frost):', vfrost
print 'Volume (ds62):', calcV(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "kJ/kbar @", T, "K and", P, "GPa"
print ''
print 'Volume (TP,Frost):', calcVbm(Ptriple, mbrkfrost, vfrost)
print 'Volume (TP,ds62):', calcV(Ptriple, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "kJ/kbar @", T, "K and", Ptriple, "GPa"
print ''
print 'k(1673,ds62):', calck(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "GPa @", T, "K and", P, "GPa"
print 'k(1673,Frost):', mbrkfrost[0], 'diff =', mbrkfrost[0] - calck(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV)
print ''
print ''

print 'FAYALITE (fa):'
mbrS=0.151
mbratoms=7.
mbrk=[1256.0, 4.68, -0.00370]
mbra=0.0000270 
ds62mbra=0.0000282
mbrV=4.6310
Z=4.
nA=6.02214e23
voltoa=1.e24*10.
mbrkfrost=[998.484, 4.]
vfrost=4.8494

print 'Volume (Frost):', vfrost
print 'Volume (ds62):', calcV(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "kJ/kbar @", T, "K and", P, "GPa"
print ''
print 'Volume (TP,Frost):', calcVbm(Ptriple, mbrkfrost, vfrost)
print 'Volume (TP,ds62):', calcV(Ptriple, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "kJ/kbar @", T, "K and", Ptriple, "GPa"
print ''
print 'k(1673,ds62):', calck(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "GPa @", T, "K and", P, "GPa"
print 'k(1673,Frost):', mbrkfrost[0], 'diff =', mbrkfrost[0] - calck(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV)
print ''
print ''

print 'MAGNESIOWADSLEYITE (mwd):'
mbrS=0.0951
mbratoms=7.
mbrk=[1726.0, 3.84, -0.00220]
mbra=0.0000223 
ds62mbra=0.0000237
mbrV=4.0510
Z=4.
nA=6.02214e23
voltoa=1.e24*10.
mbrkfrost=[1462.544, 4.21]
vfrost=4.2206

print 'Volume (Frost):', vfrost
print 'Volume (ds62):', calcV(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "kJ/kbar @", T, "K and", P, "GPa"
print ''
print 'Volume (TP,Frost):', calcVbm(Ptriple, mbrkfrost, vfrost)
print 'Volume (TP,ds62):', calcV(Ptriple, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "kJ/kbar @", T, "K and", Ptriple, "GPa"
print ''
print 'k(1673,ds62):', calck(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "GPa @", T, "K and", P, "GPa"
print 'k(1673,Frost):', mbrkfrost[0], 'diff =', mbrkfrost[0] - calck(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV)
print ''
print ''


print 'FERROWADSLEYITE (fwd):'
mbrS=0.1460
mbratoms=7.
mbrk=[1690.0, 4.35, -0.00260]
mbra=0.0000214
ds62mbra=0.0000273
mbrV=4.3210
Z=4.
nA=6.02214e23
voltoa=1.e24*10.
mbrkfrost=[3999.958, 4.]
vfrost=4.4779

print 'Volume (Frost):', vfrost
print 'Volume (ds62):', calcV(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "kJ/kbar @", T, "K and", P, "GPa"
print ''
print 'Volume (TP,Frost):', calcVbm(Ptriple, mbrkfrost, vfrost)
print 'Volume (TP,ds62):', calcV(Ptriple, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "kJ/kbar @", T, "K and", Ptriple, "GPa"
print ''
print 'k(1673,ds62):', calck(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "GPa @", T, "K and", P, "GPa"
print 'k(1673,Frost):', mbrkfrost[0], 'diff =', mbrkfrost[0] - calck(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV)
print ''
print ''



print 'MAGNESIORINGWOODITE (mrw):'
mbrS=0.0900
mbratoms=7.
mbrk=[1781.0, 4.35, -0.0024]
mbra=0.0000256
ds62mbra=0.0000201
mbrV=3.9490
Z=4.
nA=6.02214e23
voltoa=1.e24*10.
mbrkfrost=[1453.028, 4.4]
vfrost=4.1484

print 'Volume (Frost):', vfrost
print 'Volume (ds62):', calcV(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "kJ/kbar @", T, "K and", P, "GPa"
print ''
print 'Volume (TP,Frost):', calcVbm(Ptriple, mbrkfrost, vfrost)
print 'Volume (TP,ds62):', calcV(Ptriple, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "kJ/kbar @", T, "K and", Ptriple, "GPa"
print ''
print 'k(1673,ds62):', calck(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "GPa @", T, "K and", P, "GPa"
print 'k(1673,Frost):', mbrkfrost[0], 'diff =', mbrkfrost[0] - calck(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV)
print ''
print ''


print 'FERRORINGWOODITE (frw):'
mbrS=0.140
mbratoms=7.
mbrk=[1977.0, 4.92, -0.00250]
mbra=0.0000242 
ds62mbra=0.0000222
mbrV=4.2030
Z=4.
nA=6.02214e23
voltoa=1.e24*10.
mbrkfrost=[1607.810, 5.]
vfrost=4.3813

print 'Volume (Frost):', vfrost
print 'Volume (ds62):', calcV(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "kJ/kbar @", T, "K and", P, "GPa"
print ''
print 'Volume (TP,Frost):', calcVbm(Ptriple, mbrkfrost, vfrost)
print 'Volume (TP,ds62):', calcV(Ptriple, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "kJ/kbar @", T, "K and", Ptriple, "GPa"
print ''
print 'k(1673,ds62):', calck(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "GPa @", T, "K and", P, "GPa"
print 'k(1673,Frost):', mbrkfrost[0], 'diff =', mbrkfrost[0] - calck(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV)
print ''
print ''


print 'MAGNESIORINGWOODITE(2) (mrw):'
P=0.0001
T=700+273.15
mbrS=0.0900
mbratoms=7.
mbrk=[1781.0, 4.35, -0.0024]
mbra=0.0000251
ds62mbra=0.0000201
mbrV=3.9490
Z=8.
nA=6.02214e23
voltoa=1.e24*10.
vinoue=536./voltoa*nA/8


print 'Volume (ds62,700):', calcV(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "kJ/kbar @", T, "K and", P, "GPa"
print 'Volume (new,700):', calcV(P, T, mbrS, mbratoms, mbrk, mbra, mbrV), "kJ/kbar @", T, "K and", P, "GPa"
print 'Volume (Inoue,700):', vinoue 
print ''
T=1673.15
print 'Volume (ds62,1400):', calcV(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "kJ/kbar @", T, "K and", P, "GPa"
print 'Volume (new,1400):', calcV(P, T, mbrS, mbratoms, mbrk, mbra, mbrV), "kJ/kbar @", T, "K and", P, "GPa"
print ''
print 'Volume (ds62,700):', calcV(Ptriple, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "kJ/kbar @", T, "K and", Ptriple, "GPa"
print 'Volume (new,700):', calcV(Ptriple, T, mbrS, mbratoms, mbrk, mbra, mbrV), "kJ/kbar @", T, "K and", Ptriple, "GPa"
print ''

print 'MAGNESIOWADSLEYITE(2) (mwd):'
mbrS=0.0951
mbratoms=7.
mbrk=[1726.0, 3.84, -0.00220]
mbra=0.0000265
ds62mbra=0.0000237
mbrV=4.0510
Z=8.
nA=6.02214e23
voltoa=1.e24*10.
vinoue=550.3/voltoa*nA/8

print 'Volume (ds62,700):', calcV(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "kJ/kbar @", T, "K and", P, "GPa"
print 'Volume (new,700):', calcV(P, T, mbrS, mbratoms, mbrk, mbra, mbrV), "kJ/kbar @", T, "K and", P, "GPa"
print 'Volume (Inoue,700):', vinoue 
print ''
T=1673.15
print 'Volume (ds62,1400):', calcV(P, T, mbrS, mbratoms, mbrk, ds62mbra, mbrV), "kJ/kbar @", T, "K and", P, "GPa"
print 'Volume (new,1400):', calcV(P, T, mbrS, mbratoms, mbrk, mbra, mbrV), "kJ/kbar @", T, "K and", P, "GPa"
print ''
