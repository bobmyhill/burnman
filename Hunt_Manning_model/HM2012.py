#!/usr/python
import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    DKS_2013_liquids_tweaked, \
    DKS_2013_liquids, \
    DKS_2013_solids, \
    SLB_2011, \
    HP_2011_ds62
from burnman import constants
import numpy as np
from scipy.optimize import fsolve, curve_fit
from scipy.interpolate import UnivariateSpline, interp1d, splrep, splev
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


R=8.31446 # from wiki

# P in bar, T in K
def Gl(P, T, params): # [J], P [bar], T [K]
    return params[0] + params[1]*T + params[2]*P + params[3]*P*P + params[4]*T*T + params[5]*P*T

def Kl(P,T):
    return np.exp(-Gl(P,T,Gl_params)/(R*T))

def Ws(P, T, params): # [J], P [bar], T [K]
    return params[0] + params[1]*T + params[2]*P + params[3]*P*P

def Wh(P, T, params): # [J], P [bar], T [K]
    return params[0] + params[1]*T +params[2]*P

def Gmix(Xs, P, T):
    return R*T*(Xs*np.log(Xs) + (1.-Xs)*np.log(1.-Xs)) + Gex(Xs, P, T)

def XOH(Xs, P, T):
    nO2=(Xs+1.)/2.
    factor=(1./4.) - (1./(Kl(P,T)*Kl(P,T)))
    return (nO2 - np.sqrt(nO2*nO2 - 4*factor*(2.*Xs - 2.*Xs*Xs)))/(2.*factor)

def Gex(Xs, P, T):
    return Xs*(1.-Xs)*(Ws(P,T,Ws_params)*(1.-Xs) + Wh(P,T,Wh_params)*Xs) - XOH(Xs, P, T)*Gl(P,T,Gl_params)

def eqm_stv_L3(T, P, Xs, solid, liquid):
    
    K=Kl(P,T)
    factor2=np.sqrt(((K*K*(1.-3.*Xs)*(1.-3.*Xs) - 32.*(Xs-1.)*Xs))/(K*K))
    dXOHdXs=(K*K*(factor2 - 9.*Xs + 3.) + 32.*Xs - 16.)/((K*K - 4.)*factor2)

    pressure = P*1.e5
    solid.set_state(pressure, T)
    liquid.set_state(pressure, T)
    LHS=solid.gibbs - liquid.gibbs
    RHS=R*T*np.log(Xs) + (1.-Xs)*(1.-Xs)*(Ws(P,T,Ws_params) + 2.*Xs*(Wh(P, T,Wh_params) - Ws(P,T,Ws_params))) - Gl(P,T,Gl_params)*(XOH(Xs, P, T) + (1.-Xs)*dXOHdXs)
    return LHS - RHS

def eqm_stv_L2(Xs, P, T, solid, liquid):
    
    K=Kl(P,T)
    factor2=np.sqrt(((K*K*(1.-3.*Xs)*(1.-3.*Xs) - 32.*(Xs-1.)*Xs))/(K*K))
    dXOHdXs=(K*K*(factor2 - 9.*Xs + 3.) + 32.*Xs - 16.)/((K*K - 4.)*factor2)

    solid.set_state(P*1.e5, T)
    liquid.set_state(P*1.e5, T)
    LHS=solid.gibbs - liquid.gibbs
    RHS=R*T*np.log(Xs) + (1.-Xs)*(1.-Xs)*(Ws(P,T,Ws_params) + 2.*Xs*(Wh(P, T,Wh_params) - Ws(P,T,Ws_params))) - Gl(P,T,Gl_params)*(XOH(Xs, P, T) + (1.-Xs)*dXOHdXs)
    return LHS - RHS

def eqm_stv_L(Xs, P, T):
    K=Kl(P,T)
    factor2=np.sqrt(((K*K*(1.-3.*Xs)*(1.-3.*Xs) - 32.*(Xs-1.)*Xs))/(K*K))
    dXOHdXs=(K*K*(factor2 - 9.*Xs + 3.) + 32.*Xs - 16.)/((K*K - 4.)*factor2)

    LHS=(T - Tmelt)*deltaSQL
    RHS=R*T*np.log(Xs) + (1.-Xs)*(1.-Xs)*(Ws(P,T,Ws_params) + 2.*Xs*(Wh(P, T,Wh_params) - Ws(P,T,Ws_params))) - Gl(P,T,Gl_params)*(XOH(Xs, P, T) + (1.-Xs)*dXOHdXs)
    return LHS - RHS


def eqm_ice_L(Xs, P, T):
    K=Kl(P,T)
    factor2=np.sqrt(((K*K*(1.-3.*Xs)*(1.-3.*Xs) - 32.*(Xs-1.)*Xs))/(K*K))
    dXOHdXs=(K*K*(factor2 - 9.*Xs + 3.) + 32.*Xs - 16.)/((K*K - 4.)*factor2)

    LHS=(T - Ticemelt)*deltaSIL

    RHS=R*T*np.log(1.-Xs) + Xs*Xs*(Wh(P,T,Wh_params) + 2.*(1.-Xs)*(Ws(P, T,Ws_params) - Wh(P,T,Wh_params))) - Gl(P,T,Gl_params)*(XOH(Xs, P, T) - Xs*dXOHdXs)
    return LHS - RHS



'''
def eqm_stv_L_2(Xs, P, T):

    Tmelt=(1695.6+273.15) + 35.46*(P/1000. - 6.) - 0.3766*(P/1000. - 6.)*(P/1000. - 6.) # K, Jackson (1976)
    deltaSQL=5.53 # J/K

    delta=0.0001
    Xs1=Xs-delta
    Xs2=Xs+delta
    Xs3=1.

    G1=Gmix(Xs1, P, T)
    G2=Gmix(Xs2, P, T)
    G3=(T - Tmelt)*deltaSQL

    dGdX1=(G3-G1)/(Xs3-Xs1)
    dGdX2=(G3-G2)/(Xs3-Xs2)
    return dGdX1-dGdX2

def dGdXs(Xs, P, T):
    K=Kl(P,T)
    factor2=np.sqrt(((K*K*(1.-3.*Xs)*(1.-3.*Xs) - 32.*(Xs-1.)*Xs))/(K*K))
    dXOHdXs=(K*K*(factor2 - 9.*Xs + 3.) + 32.*Xs - 16.)/((K*K - 4.)*factor2)
    return R*T*(np.log(Xs) - np.log(1.-Xs)) + Ws(P, T)*(3.*Xs-1.)*(Xs-1.) + Wh(P, T)*Xs*(2.-3*Xs) - Gl(P,T)*dXOHdXs

def eqm_stv_L_3(Xs, P, T):

    Tmelt=(1695.6+273.15) + 35.46*(P/1000. - 6.) - 0.3766*(P/1000. - 6.)*(P/1000. - 6.) # K, Jackson (1976)
    deltaSQL=5.53 # J/K

    G=Gmix(Xs, P, T)
    GQL=(T - Tmelt)*deltaSQL

    dGdX1=dGdXs(Xs, P, T)
    dGdX2=(GQL-G)/(1.-Xs)
    return dGdX1-dGdX2
'''


# Gl = a+ bT + cP + dPP + eTT + fPT
Gl_params=[22070., - 19.08, - 1.3168, 2.2987e-5, 5.4464e-3, 18.990e-5]

# Ws = a+ bT + cP + dPP 
Ws_params=[92631., -24.585, -4.9086, 9.1719e-5]

# Wh = a+ bT + cP
Wh_params=[110740.,-65.569,-1.1141]


pressures=[10000., 20000.]
for P in pressures:
    Tmelt=(1695.6+273.15) + 35.46*(P/1000. - 6.) - 0.3766*(P/1000. - 6.)*(P/1000. - 6.) # K, Jackson (1976)
    deltaSQL=5.53 # J/K
    temperatures=np.linspace(873.15, Tmelt, 101)
    XSiO2=np.empty_like(temperatures)


    for i, T in enumerate(temperatures):
        XSiO2[i]=fsolve(eqm_stv_L, 0.01, args=(P, T))[0]

    plt.plot( XSiO2, temperatures-273.15, '-', linewidth=2., label=str(P/10000.)+' GPa')

def print_params(pressure, temperature):
    print Gl_params[0] + Gl_params[1]*temperature + Gl_params[2]*pressure + Gl_params[3]*pressure*pressure + Gl_params[4]*temperature*temperature + Gl_params[5]*pressure*temperature, \
        Ws_params[0] + Ws_params[1]*temperature + Ws_params[2]*pressure + Ws_params[3]*pressure*pressure,\
        Wh_params[0] + Wh_params[1]*temperature + Wh_params[2]*pressure 

temperatures = np.linspace(1000., 2000., 3)
for temperature in temperatures: # p is in bar, T is in K
    for pressure in pressures:
        print pressure, temperature,
        print_params(pressure, temperature)

# We need Tmelt and Smelt of stishovite at 13 GPa
# Tmelt can be obtained from Zhang et al., 1993
# Smelt can be obtained from the melting curve of stv, extrapolated to 13 GPa. The expression of Millot et al., 2015 closely matches the fit of Zhang et al.. Both rely heavily on the data by Lyzenga, 1983.

P=130000. # bar

# Millot et al., 2015
# T = 1968.5 + 307.8*P^0.485
Tmelt=1968.5 + 307.8*np.power(P/1.e4,0.485)
print Tmelt

#Thus, at 13 GPa
dTdP = 307.8*0.485*np.power(P/1.e4,0.485-1.)/1.e9
print 'dtdP', dTdP, 'K/Pa'

# dT/dP = DV/DS
# Zhang et al. show that the melting curve of coesite is essentially flat at 13 GPa, indicating that the volume change of melting is zero. The equation of state of coesite in Holland and Powell (2011) gives a volume of 1.91043016645e-05 at 13 GPa, 2973 K. Stishovite gives a volume of 1.43616420241e-05.

#Thus
Vliq=1.91043016645e-05 # (volume of coesite, m^3/mol)
Vstv=1.43616420241e-05
deltaSQL= -(Vstv - Vliq)/dTdP # J/K
Tmelt = 2700. # ##############################
deltaSQL = 10. # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxx
print deltaSQL

print 'test', (Vstv - Vliq)*0.0028*1.e9


# Now let's do the same for water and ice VII
# The paper we will use is Frank et al. (2004)
Ticemelt = 355. * np.power((((P/1.e4-2.17)/0.764) + 1), 1./4.32)
print Ticemelt

#Thus, at 13 GPa
dTdP = 355.*np.power((((P/1.e4-2.17)/0.764) + 1), 1./4.32 - 1.)/(0.764*4.32) / 1.e9
print dTdP

# dT/dP = DS/DV
# From Figure 6, we have
Vliq=10.21e-6
Vice=9.38e-6

#Thus
deltaSIL= -(Vice - Vliq)/dTdP # J/K
print deltaSIL


# Now let's reset the melt model parameters
# Gl = a+ bT + cP + dPP + eTT + fPT
# Ws = a+ bT + cP + dPP 
# Wh = a+ bT + cP

'''
Gl_params=[-6000., 4., 0., 0., 0, 0.]
Ws_params=[0., 0., 0., 0.]
Wh_params=[0.,0., 0.]
'''

'''
Gl_params=[-20000., 0., 0., 0., 0, 0.]
Ws_params=[0., 0., 0., 0.]
Wh_params=[0.,0., 0.]
'''
'''
Gl_params=[12000., 0., 0., 0., 0, 0.]
Ws_params=[10000., 0., 0., 0.]
Wh_params=[10000.,0., 0.]
'''
'''
Gl_params=[14000., 0., 0., 0., 0, 0.]
Ws_params=[12000., 0., 0., 0.]
Wh_params=[12000.,0., 0.]
'''
'''
Gl_params=[12000., 0., 0., 0., 0, 0.]
Ws_params=[50000., -20., 0., 0.]
Wh_params=[50000.,-20., 0.]
'''

'''
Gl_params=[-3000., 0., 0., 0., 0, 0.]
Ws_params=[-10000., 0., 0., 0.]
Wh_params=[-10000.,0., 0.]
'''

P=130000. # bar
Gl_params=[000., 0., 0., 0., 0, 0.]
Ws_params=[0000., 0., 0., 0.]
Wh_params=[0000.,0., 0.]

stv=SLB_2011.stishovite()
SiO2_liq = DKS_2013_liquids_tweaked.SiO2_liquid()
SiO2_liq_alt = DKS_2013_liquids_tweaked.SiO2_liquid_alt()
'''
# And solve for the equilibrium temperature between ice, stv and melt
def eqm_ice_stv(T, P):
    return fsolve(eqm_ice_L, 0.99, args=(P, T))[0] - fsolve(eqm_stv_L2, 0.01, args=(P, T, stv, SiO2_liq))[0]

T_stv_ice=fsolve(eqm_ice_stv, 660., args=(P))[0]
print T_stv_ice

plt.plot( np.array([0., 1.]), np.array([T_stv_ice, T_stv_ice])-273.15, 'r-', linewidth=2.)
'''          
T_stv_ice=1000.
temperatures=np.linspace(T_stv_ice, Tmelt, 101)
XSiO2=np.empty_like(temperatures)

#XSiO2=np.linspace(0.1, 0.90, 41)
#temperatures=np.empty_like(XSiO2)
#for i, X in enumerate(XSiO2):
for i, T in enumerate(temperatures):
    XSiO2[i]=fsolve(eqm_stv_L2, 0.01, args=(P, T, stv, SiO2_liq_alt))[0]
    #temperatures[i]=fsolve(eqm_stv_L3, 1000., args=(P, X, stv, SiO2_liq))[0]
    #print X, temperatures[i]
    
plt.plot( XSiO2, temperatures-273.15, 'r-', linewidth=2., label=str(P/10000.)+' GPa')


temperatures=np.linspace(T_stv_ice, Ticemelt, 10)
XSiO2=np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    XSiO2[i]=fsolve(eqm_ice_L, 0.99, args=(P, T))[0]

plt.plot( XSiO2, temperatures-273.15, 'r-', linewidth=2.)




fig1 = mpimg.imread('2GPa_Hunt_Manning.png')
#fig1 = mpimg.imread('1GPa_Hunt_Manning.png')
plt.imshow(fig1, extent=[0,1,600,1800], aspect='auto')


stishovite=[]
liquid=[]
for line in open('data/13GPa_SiO2-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[2] == 's':
            stishovite.append([float(content[0]), float(content[1])])
        if content[2] == 'l':
            liquid.append([float(content[0]), float(content[1])])

stishovite=zip(*stishovite)
liquid=zip(*liquid)
plt.plot( stishovite[1], stishovite[0], marker='.', linestyle='none', label='stv+liquid')
plt.plot( liquid[1], liquid[0], marker='.', linestyle='none', label='superliquidus')

plt.ylim(200., 3000.)
plt.xlim(0., 1.)

plt.ylabel("Temperature (C)")
plt.xlabel("XSiO2")
plt.legend(loc='upper left')
plt.show()
