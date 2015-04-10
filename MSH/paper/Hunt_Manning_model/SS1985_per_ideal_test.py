#!/usr/python
import os, sys
import numpy as np

from scipy.optimize import fsolve, minimize, fmin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if not os.path.exists('burnman'):
    sys.path.insert(1,os.path.abspath('/home/rm438/projects/burnman/'))


# Benchmarks for the solid solution class
import burnman
from burnman import minerals
from burnman import tools
from burnman.mineral import Mineral
from burnman.minerals import Myhill_calibration_iron
from burnman.processchemistry import *
from burnman.chemicalpotentials import *
from burnman import constants
atomic_masses=read_masses()

R=8.31446 # from wiki

class dummy (Mineral): 
    def __init__(self):
        formula='Fe'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'For excess properties only!!',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 0. ,
            'S_0': 0. ,
            'V_0': 1e-6 ,
            'Cp': [0., 0., 0., 0.] ,
            'a_0': 3.71e-05 ,
            'K_0': 1.857e+11 ,
            'Kprime_0': 4.05 ,
            'Kdprime_0': -2.2e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class dummy_br (Mineral): 
    def __init__(self):
        formula='Fe'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'For excess properties only!!',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -32000.-10*298.15 ,
            'S_0': 10. ,
            'V_0': 1.e-6 ,
            'Cp': [0., 0., 0., 0.] ,
            'a_0': 3.71e-05 ,
            'K_0': 1.857e+11 ,
            'Kprime_0': 4.05 ,
            'Kdprime_0': -2.2e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

# Configurational entropy
class hydrous_melt(burnman.SolidSolution):
    def __init__(self):
        # Name
        self.name='Hydrous melt'
        self.type='subregular'
        self.endmembers = [[dummy(), '[O]'],[dummy_br(), '[Oh]'],[dummy(), '[Ho]']]

        # Interaction parameters
        self.enthalpy_interaction=[[[0., 0.],[0.e3,0e3]],[[0.e3, 0.]]]
        self.volume_interaction=[[[0., 0.],[0.,0.]],[[0., 0.]]]
        self.entropy_interaction=[[[0., 0.],[0.,0.]],[[0., 0.]]]

        burnman.SolidSolution.__init__(self)


liq=hydrous_melt()

liq.set_composition([0.1,0.5,0.4])
liq.set_state(13.e9, 1000.)

print 'hi', liq.excess_gibbs


def eqm_order(Xbr, X, r, P, T):
    XH2O=X*(1. + Xbr) - Xbr
    if XH2O[0]<0.:
        XH2O[0]=0.

    XMgO=1. - XH2O - Xbr
    '''
    if XMgO[0]<0.:
        XMgO[0]=0.
        XH2O[0]=1. - XMgO - Xbr
    '''
    liq.set_composition([XMgO[0], Xbr[0], XH2O[0]])
    liq.set_state(1.e5, T)

    return liq.gibbs


X=0.67
r=1.
P=13.e9
T=1300.
for X in np.linspace(0.001, 0.999, 11):
    print minimize(eqm_order, 0.01, method='TNC', bounds=((0.0, 1.-np.abs(1.-2.*X)),), args=(X, r, P, T), options={'disp': False}).x[0]
    print liq.gibbs, liq.molar_fractions


def eqm_with_L(X, Tmelt, Smelt, r, K): # X is mole fraction H2O
    Xb=X/(X + r*(1.-X)) # eq. 5.3b
    XO=1.-Xb-(0.5 - np.sqrt(0.25 - (K-4.)/K*(Xb-Xb*Xb)))/((K-4)/K) # eq. 5.3

    factor=1.-(r*R*np.log(XO))/Smelt
    T=Tmelt/factor # eq. 13
    return T


def excesses_nonideal(X, T, r, K, Wsh, Whs): # X is mole fraction H2O
    Xb=X/(X + r*(1.-X)) # eq. 5.3b
    XO=1.-Xb-(0.5 - np.sqrt(0.25 - (K-4.)/K*(Xb-Xb*Xb)))/((K-4)/K) # eq. 5.3

    activity_anhydrous_phase=np.power(XO,r)
    activity_H2O=XO + 2*Xb - 1.0

    partial_excess_anhydrous_phase=R*T*np.log(activity_anhydrous_phase)
    partial_excess_H2O=R*T*np.log(activity_H2O)

    #partial_excess_anhydrous_phase+=Xb*Xb*W
    #partial_excess_H2O+=(1.-Xb*Xb)*W

    Xs=1.-Xb
    partial_excess_anhydrous_phase+= 2.*Xb*Xb*(1.-Xb)*Whs - Xb*Xb*(1.-2.*Xb)*Wsh 
    partial_excess_H2O+= 2.*Xs*Xs*(1.-Xs)*Wsh - Xs*Xs*(1.-2.*Xs)*Whs 
    return partial_excess_anhydrous_phase, partial_excess_H2O

def excesses_nonideal_2(X, T, K): # X is mole fraction H2O
    Xb=X/(X + r*(1.-X)) # eq. 5.3b

    XO=1.-Xb-(0.5 - np.sqrt(0.25 - (K-4.)/K*(Xb-Xb*Xb)))/((K-4)/K) # eq. 5.3
    XH2O=XO + 2*Xb - 1.0
    XOH=1.- XO - XH2O

    liq.set_composition([XO, XOH, XH2O])
    liq.set_state(13.e9, T)

    return liq.excess_partial_gibbs

def activities(X, r, K): # X is mole fraction H2O
    Xb=X/(X + r*(1.-X)) # eq. 5.3b
    XO=1.-Xb-(0.5 - np.sqrt(0.25 - (K-4.)/K*(Xb-Xb*Xb)))/((K-4)/K) # eq. 5.3
    XH2O=XO + 2*Xb - 1.0
    XOH=2.*(Xb-XH2O)
    return np.power(XO,r), XH2O, XOH

def solveT(T, Xs, Tmelt, Smelt, r, K):
    return T-eqm_with_L(Xs, Tmelt, Smelt, r, K(T))

def solve_composition(Xs, T, Tmelt, Smelt, r, K, Wsh, Whs):
    return Smelt*(T-Tmelt)-excesses_nonideal(Xs, T, r, K(T), Wsh(T), Whs(T))[0]

'''
    P=13.e9
    minimize(eqm_order, 0.001, method='TNC', bounds=((0.0, 1.-np.abs(1.-2.*X)),), args=(X, r, P, T), options={'disp': False}).x[0] 
    print liq.partial_gibbs
    print liq.partial_gibbs[0]
    return Smelt*(T-Tmelt)-liq.partial_gibbs[0]
'''

'''
# 6 GPa, en
r=3.0 # Oxygens available for bonding
K = lambda T: np.exp(-6000./T + 4.5) # equilibrium constant
Smelt=40. # J/K/mol
Tmelt=2010. # C
f=0.5 # ternary is 0.5*MgO + 0.5*SiO2 = 0.5*MgSiO3
Msil=100.3887
MH2O=18.01528
'''

# 13 GPa, per
r=1. # Oxygens available for bonding
Kinf = lambda T: 100000000000.
K0 = lambda T: 0.00000000001
K = lambda T: np.exp(-(-64000.-20*T)/(R*T))
Wsh = lambda T: 0000.
Whs = lambda T: 00000.
Smelt=22. # Cohen and Gong
Tmelt=5373. # K, Cohen and Gong
f=1.0
Msil=40.3044
MH2O=18.01528

Xbr=0.67 # composition of fluid in eqm with Xbr
#Xbr=0.76 
Tbr=1210.+273.15 # K


XH2O = np.exp(excesses_nonideal(Xbr, Tbr, r, K(Tbr), Wsh(Tbr), Whs(Tbr))[1]/(R*Tbr))
dGH2O= R*Tbr*np.log(XH2O)


dGH2O_HP=1000.*((-829.82 - -547.59)- -274.10)
XH2O_HP=np.exp(dGH2O_HP/(R*Tbr))
print XH2O_HP, XH2O
print dGH2O_HP, dGH2O


compositions=np.linspace(0.0001, 0.99, 101)
Gex=np.empty_like(compositions)
Gex_2=np.empty_like(compositions)
for i, X in enumerate(compositions):
    print i
    Gex[i]=(1-X)*excesses_nonideal(X, Tbr, r, K(Tbr), Wsh(Tbr), Whs(Tbr))[0] + X*excesses_nonideal(X, Tbr, r, K(Tbr), Wsh(Tbr), Whs(Tbr))[1]
    res=minimize(eqm_order, 0.001, method='TNC', bounds=((0.0, 1.-np.abs(1.-2.*X)),), args=(X, r, P, T), options={'disp': False})
    Gex_2[i]=liq.gibbs

#(1-X)*excesses_nonideal_2(X, Tbr, K(Tbr))[0] + X*excesses_nonideal_2(X, Tbr, K(Tbr))[2]



plt.plot( compositions, Gex, '-', linewidth=2., label='SS1985 model')
plt.plot( compositions, Gex_2, '-', linewidth=2., label='Subregular model')
#plt.plot ( 0.0, Smelt*(Tbr-Tmelt), marker='o', label='model per')
plt.plot ( [1.0], [dGH2O_HP], marker='o', label='HP H2O activity')
plt.plot ( [0.0, 1.0], [excesses_nonideal(Xbr, Tbr, r, K(Tbr), Wsh(Tbr), Whs(Tbr))[0], excesses_nonideal(Xbr, Tbr, r, K(Tbr), Wsh(Tbr), Whs(Tbr))[1]], marker='o', label='model H2O activity')
plt.ylabel("Excess Gibbs (J/mol)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()



H2Omolfraction=np.linspace(0.0001, 0.8, 1001)
temperatures=np.empty_like(H2Omolfraction)
temperaturesinf=np.empty_like(H2Omolfraction)
temperatures0=np.empty_like(H2Omolfraction)
H2Owtfraction=np.empty_like(H2Omolfraction)
ternaryH2Omolfraction=np.empty_like(H2Omolfraction)
for i, Xs in enumerate(H2Omolfraction):
    #temperatures[i]=eqm_with_L(Xs, Tmelt, Smelt, r, K)
    temperatures[i]=fsolve(solveT, 1400., args=(Xs, Tmelt, Smelt, r, K))
    temperaturesinf[i]=fsolve(solveT, 1400., args=(Xs, Tmelt, Smelt, r, Kinf))
    temperatures0[i]=fsolve(solveT, 1400., args=(Xs, Tmelt, Smelt, r, K0))
    H2Owtfraction[i]=(H2Omolfraction[i]*MH2O)/(H2Omolfraction[i]*MH2O + (1.-H2Omolfraction[i])*Msil)
    ternaryH2Omolfraction[i]=H2Omolfraction[i]/(H2Omolfraction[i] + (1.-H2Omolfraction[i])/f)

temperatures_2=np.linspace(600., 5000., 101)
compositions_2=np.empty_like(temperatures_2)
for i, T in enumerate(temperatures_2):
    compositions_2[i]=fsolve(solve_composition, 0.001, args=(T, Tmelt, Smelt, r, K, Wsh, Whs))

plt.plot( compositions_2, temperatures_2, linewidth=1, label='test')

periclase=[]
brucite=[]
liquid=[]
for line in open('../figures/13GPa_per-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[2] == 'p' or content[2] == 'sp':
            periclase.append([float(content[0])+273.15, float(content[1])/100.])
        if content[2] == 'l':
            liquid.append([float(content[0])+273.15, float(content[1])/100.])
        if content[2] == 'b':
            brucite.append([float(content[0])+273.15, float(content[1])/100.])

periclase=zip(*periclase)
brucite=zip(*brucite)
liquid=zip(*liquid)
plt.plot( periclase[1], periclase[0], marker='.', linestyle='none', label='per+liquid')
plt.plot( brucite[1], brucite[0], marker='.', linestyle='none', label='br+liquid')
plt.plot( liquid[1], liquid[0], marker='.', linestyle='none', label='superliquidus')

plt.plot( H2Omolfraction, temperatures, '-', linewidth=2., label='r='+str(r)+'; K=K(T)')
plt.plot( H2Omolfraction, temperaturesinf, '-', linewidth=2., label='r='+str(r)+'; K=inf')
plt.plot( H2Omolfraction, temperatures0, '-', linewidth=2., label='r='+str(r)+'; K=0')

plt.ylim(1000.,5500.)
plt.xlim(0.,1.)
plt.ylabel("Temperature (K)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()

