# This file calculates the 1 bar adjustments to the free energies of 
# DKS2013 MgO and SiO2 liquids, making the assumption that the melting curves are
# as found in the literature and the equations of state (the volumetric part) is correct.

# These adjustments are then compared to the low temperature models of Wu et al. (1993)



import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import HP_2011_ds62
from burnman.minerals import SLB_2011
from burnman.minerals import DKS_2008_fo
from burnman.minerals import DKS_2013_solids
from burnman.minerals import DKS_2013_liquids
from burnman import constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit, fsolve
from scipy.interpolate import interp1d

class params():
    def __init__(self):
        self.A = 0.
        self.B = 0.
        self.a = [0., 0., 0., 0., 0.]
        
def enthalpy(T, prm):
    return prm.A + ( prm.a[0]*( T - 298.15 ) +
                     0.5  * prm.a[1]*( T*T - 298.15*298.15 ) +
                     -1.0 * prm.a[2]*( 1./T - 1./298.15 ) +
                     2.0  * prm.a[3]*(np.sqrt(T) - np.sqrt(298.15) ) +
                     -0.5 * prm.a[4]*(1./(T*T) - 1./(298.15*298.15) ) )

def entropy(T, prm):
    return prm.B + ( prm.a[0]*(np.log(T/298.15)) +
                     prm.a[1]*(T - 298.15) +
                     -0.5 * prm.a[2]*(1./(T*T) - 1./(298.15*298.15)) +
                     -2.0 * prm.a[3]*(1./np.sqrt(T) - 1./np.sqrt(298.15)) +
                     -1./3. * prm.a[4]*(1./(T*T*T) - 1./(298.15*298.15*298.15) ) )

def heat_capacity(T, prm):
    return ( prm.a[0] +
             prm.a[1]*T +
             prm.a[2]/(T*T) +
             prm.a[3]/np.sqrt(T) +
             prm.a[4]/(T*T*T) )

def G_MgO_liquid(T):
    prm = params()
    prm.A = -130340.58
    prm.B = 6.4541207
    prm.a = [17.398557, -0.751e-3, 1.2494063e5, -70.793260, 0.013968958e8]

    return 4.184*(enthalpy(T, prm) - T*entropy(T, prm))


def G_SiO2_liquid(T):
    prm = params()
    if T < 1996.: # note, agreement with melting curve much better if this LT law also applies at HT
        prm.A = -214339.36
        prm.B = 12.148448
        prm.a = [19.960229, 0.e-3, -5.8684512e5, -89.553776, 0.66938861e8]
    else:
        prm.A = -221471.21
        prm.B = 2.3702523
        prm.a = [20.50000, 0., 0., 0., 0.]

    return 4.184*(enthalpy(T, prm) - T*entropy(T, prm))


# Quick check SiO2
temperatures = np.linspace(500., 2500., 101)
Cps = np.empty_like(temperatures)
Cps2 = np.empty_like(temperatures)
Cps3 = np.empty_like(temperatures)
Cps4 = np.empty_like(temperatures)
prm = params()


crst = HP_2011_ds62.crst()
qtz = HP_2011_ds62.q()
coe = HP_2011_ds62.coe()
stv = HP_2011_ds62.stv()
SiO2_liq = DKS_2013_liquids.SiO2_liquid()

def S_fit(T, B, a, b, c, d, e):
    prm = params()
    prm.A = 0.
    prm.B = B
    prm.a = [a, b, c, d, e]
    
    return entropy(T, prm)

for i, T in enumerate(temperatures):
    if T < 1996.: # note, agreement with melting curve much better if this LT law also applies at HT
        prm.A = -214339.36
        prm.B = 12.148448
        prm.a = [19.960229, 0.e-3, -5.8684512e5, -89.553776, 0.66938861e8]
    else:
        prm.A = -221471.21
        prm.B = 2.3702523
        prm.a = [20.50000, 0., 0., 0., 0.]
    Cps[i] = 4.184*heat_capacity(T, prm)
    
    SiO2_liq.set_state(1.e5, T)
    Cps2[i] = SiO2_liq.heat_capacity_p

    prm.A = -2.20401092e+05
    prm.B = 4.78243403e+00
    prm.a = [1.82964211e+01,   6.76081909e-04, 0., 0., 0.]
    Cps3[i] = 4.184*heat_capacity(T, prm)

    
    crst.set_state(1.e5, T)
    Cps4[i] = crst.heat_capacity_p

plt.plot(temperatures, Cps, label='Wu et al., 1993')
plt.plot(temperatures, Cps2, label='DKS liquid')
plt.plot(temperatures, Cps3, label='To fit melting curve with dV=3.5e-7')
plt.plot(temperatures, Cps4, label='crst')

plt.plot(temperatures, 4.184*heat_capacity(temperatures, prm), label='HT_fit to melting curve')


S_fo = np.loadtxt('data/entropies_fo_Richet_1993.dat', unpack=True)
popt, pcov = curve_fit(S_fit, S_fo[0], S_fo[1])

prm_foL = params()
prm_foL.A = 0.
prm_foL.B = popt[0]
prm_foL.a = [popt[1], popt[2], popt[3], popt[4], popt[5]]

Sc_fo = np.loadtxt('data/conf_entropies_fo_Richet_1993.dat', unpack=True)
popt, pcov = curve_fit(S_fit, Sc_fo[0], Sc_fo[1])

prm_foL.A = 0.
prm_foL.B = prm_foL.B - popt[0]
prm_foL.a = np.array(prm_foL.a) - np.array([popt[1], popt[2], popt[3], popt[4], popt[5]])

prm_perL = params()
prm_perL.A = -130340.58
prm_perL.B = 6.4541207
prm_perL.a = [17.398557, -0.751e-3, 1.2494063e5, -70.793260, 0.013968958e8]


temperatures = np.linspace(1000., 2500., 101)
plt.plot(temperatures, heat_capacity(temperatures, prm_foL) - 2.*4.184*heat_capacity(temperatures, prm_perL), label='Richet fo, per')
plt.plot(temperatures, heat_capacity(temperatures, prm_foL)/3., label='Richet fo/3')
plt.plot(temperatures, 4.184*heat_capacity(temperatures, prm_perL), label='per')


'''
temperatures = np.linspace(1100., 1900., 9)
amorphous_SiO2 = np.loadtxt('data/Richet_SiO2_Cp.dat', unpack=True)
Cp = interp1d(amorphous_SiO2[0], amorphous_SiO2[1], kind='cubic')
#plt.plot(temperatures, Cp(temperatures))

Hdiff = Cp(temperatures)*(temperatures - 273.)

plt.plot(temperatures, np.gradient(Hdiff)/100., label='')
'''

plt.plot([1850.], [81.373], marker='o', label='Richet (obs)')


temperatures = np.linspace(500., 3500., 101)
def Cp_amorphous_SiO2(T): # Richet et al., 1982
    Cps = []
    for t in T:
        if t<1607:
            Cps.append(127.2 - 10.777e-3*t + 4.3127e5/t/t - 1463.8/np.sqrt(t))
        else:
            Cps.append(81.373)
    return np.array(Cps)

plt.plot(temperatures, Cp_amorphous_SiO2(temperatures), label='Richet et al., 1982')

plt.legend(loc='upper left')
plt.show()

fig1 = mpimg.imread('figures/entropies_fo_0_700J_Richet_1993.png')
plt.imshow(fig1, extent=[0., 2500., 0., 700.], aspect='auto')

plt.plot(temperatures, entropy(temperatures, prm_foL), label='Richet foL (S)')
plt.plot(temperatures, heat_capacity(temperatures, prm_foL), label='Richet foL (Cp)')
plt.show()

triple_points = [(crst, 0.e9, 1996.),
                 (crst, 0.6e9, 2000.),
                 (qtz, 4.4e9, 2660.),
                 (coe, 13.62e9, 3080.)]

Tm = 1999.
crst.set_state(1.e5, Tm)
SiO2_liq.set_state(1.e5, Tm)
Hm = 8920.
Sm = Hm/Tm
delta_G = crst.gibbs - SiO2_liq.gibbs
delta_S = crst.S - SiO2_liq.S + Sm
delta_E = delta_G + Tm*delta_S
SiO2_liq.property_modifiers = [['linear', {'delta_E':  delta_E, 'delta_S': delta_S, 'delta_V': 0.}]]
burnman.Mineral.__init__(SiO2_liq)

for (m, P, T) in triple_points:
    m.set_state(P, T)
    SiO2_liq.set_state(P, T)

    print m.params['name'], SiO2_liq.V - m.V, SiO2_liq.S - m.S, (SiO2_liq.V - m.V)/(SiO2_liq.S - m.S)*1.e9





# MgO

MgO_liq = DKS_2013_liquids.MgO_liquid()
per = DKS_2013_solids.periclase()



Tm = 3098.
per.set_state(1.e5, Tm)
per.property_modifiers = [['linear', {'delta_E': G_MgO_liquid(Tm) - per.gibbs, 'delta_S':0., 'delta_V': 0.} ]]

Tm = 3098.
MgO_liq.set_state(1.e5, Tm)
per.set_state(1.e5, Tm)

MgO_liq.property_modifiers = [['linear', {'delta_E': per.gibbs - MgO_liq.gibbs, 'delta_S': 0., 'delta_V': 0.} ]]
burnman.Mineral.__init__(MgO_liq)


temperatures = np.linspace(1000., 5000., 101)
states = zip(*[temperatures*0. + 1.e11, temperatures])
states1 = zip(*[temperatures*0. + 1.e11, temperatures+1.])

Ss_per0 = burnman.tools.property_evaluation(per, states, ['entropy'])
Ss_per1 = burnman.tools.property_evaluation(per, states1, ['entropy'])
Cps_liq = burnman.tools.property_evaluation(MgO_liq, states, ['heat_capacity_p'])

Cps_per = temperatures*(Ss_per1 - Ss_per0)

plt.plot(temperatures, Cps_per, label='per')
plt.plot(temperatures, Cps_liq, label='liq')
plt.legend(loc='lower right')
plt.show()

TG_diff = [] 
TGGdiffs = []

for f in ['data/Alfe_MgO_melting.dat', 'data/Zhang_Fei_MgO_melting.dat',
          'data/Zhang_Fei_MgO_melting_lo.dat', 'data/Zhang_Fei_MgO_melting_hi.dat']:
    MgO_melting_curve = np.loadtxt(f, unpack=True)
    pressures = MgO_melting_curve[0]*1.e9
    MgO_melting_curve = interp1d(MgO_melting_curve[0]*1.e9, MgO_melting_curve[1], kind='cubic')
    
    temperatures= np.empty_like(pressures)
    G_diff = np.empty_like(pressures)
    
    for i, P in enumerate(pressures):
        T = MgO_melting_curve(P)
        MgO_liq.set_state(P, T)
        per.set_state(P, T)
        G_diff[i] = per.gibbs - MgO_liq.gibbs
        temperatures[i] = T
        

    TG_diff.append([temperatures, G_diff])
    
    
    # Now show the gibbs energies at 1 bar

    Ts = []
    Gs = []
    Gdiffs = []
    
    for i, T in enumerate(temperatures):
        try:
            MgO_liq.set_state(1.e5, T)
            Gs.append(MgO_liq.gibbs + G_diff[i])
            Gdiffs.append(MgO_liq.gibbs + G_diff[i] - G_MgO_liquid(T))
            Ts.append(T)
        except:
            print 'T outside EoS range'

    TGGdiffs.append([Ts, Gs, Gdiffs])

Tguess = Tm
pressures = np.linspace(1.e9, 100.e9, 101)
temperatures = np.empty_like(pressures)
G_diff = np.empty_like(pressures)
Ts = []
Gs = []
Gdiffs = []
for i, P in enumerate(pressures):
    temperatures[i] = burnman.tools.equilibrium_temperature([per, MgO_liq],
                                                            [1., -1.],
                                                            P, Tguess)
    Tguess = temperatures[i]
    G_diff[i] = 0.
    MgO_liq.set_state(1.e5, temperatures[i])

for i, T in enumerate(temperatures):
    try:
        MgO_liq.set_state(1.e5, T)
        Gs.append(MgO_liq.gibbs + G_diff[i])
        Gdiffs.append(MgO_liq.gibbs + G_diff[i] - G_MgO_liquid(T))
        Ts.append(T)
    except:
        print 'T outside EoS range'
    
TG_diff.append([temperatures, G_diff])
TGGdiffs.append([Ts, Gs, Gdiffs])

for TG in TG_diff:
    plt.plot(TG[0], TG[1], marker='o', linestyle='None')
plt.show()
    
            
temperatures = np.linspace(1000., 8000., 101)
gibbs = np.array([G_MgO_liquid(T) for T in temperatures])
plt.plot(temperatures, gibbs)

for TGG in TGGdiffs:
    plt.plot(TGG[0], TGG[1], marker='o', linestyle='None')
plt.show()

for TGG in TGGdiffs:
    plt.plot(TGG[0], TGG[2], marker='o', linestyle='None')
plt.show()


# SiO2

crst = HP_2011_ds62.crst()
qtz = HP_2011_ds62.q()
coe = HP_2011_ds62.coe()
stv = HP_2011_ds62.stv()
stv.property_modifiers = [['linear', {'delta_E': -4.e3, 'delta_S': 0., 'delta_V': 0.}]]


SiO2_liq = DKS_2013_liquids.SiO2_liquid()

Tm = 1999.
crst.set_state(1.e5, Tm)
SiO2_liq.set_state(1.e5, Tm)
Hm = 8920.
Sm = Hm/Tm
delta_G = crst.gibbs - SiO2_liq.gibbs
delta_S = crst.S - SiO2_liq.S + Sm
delta_E = delta_G + Tm*delta_S
SiO2_liq.property_modifiers = [['linear', {'delta_E':  delta_E, 'delta_S': delta_S, 'delta_V': 3.5e-7}]]
burnman.Mineral.__init__(SiO2_liq)


crst.set_state(1.e5, Tm)
SiO2_liq.set_state(1.e5, Tm)
print crst.gibbs, SiO2_liq.gibbs


SiO2_melting_curve = np.loadtxt('data/SiO2_melting.dat', unpack=True)
pressures = SiO2_melting_curve[0]*1.e9
SiO2_melting_curve = interp1d(SiO2_melting_curve[0]*1.e9, SiO2_melting_curve[1], kind='linear')

temperatures= np.empty_like(pressures)
G_diff = np.empty_like(pressures)
for i, P in enumerate(pressures):
    T = SiO2_melting_curve(P)
    SiO2_liq.set_state(P, T)

    if P < 16.e9:
        crst.set_state(P, T)
        qtz.set_state(P, T)
        coe.set_state(P, T)
        stv.set_state(P, T)
        
        solid_gibbs = np.min([crst.gibbs, qtz.gibbs, coe.gibbs, stv.gibbs])
    else:
        stv.set_state(P, T)
        solid_gibbs = stv.gibbs

    G_diff[i] = solid_gibbs - SiO2_liq.gibbs
    temperatures[i] = T

plt.plot(temperatures, G_diff, marker='o')
plt.show()


# Now show the gibbs energies at 1 bar

Ts = []
Gs = []
Gdiffs = []

for i, T in enumerate(temperatures):
    try:
        SiO2_liq.set_state(1.e5, T)
        Gs.append(SiO2_liq.gibbs + G_diff[i])
        Ts.append(T)
        Gdiffs.append(SiO2_liq.gibbs + G_diff[i] - G_SiO2_liquid(T))
    except:
        print 'T outside EoS range'
        
temperatures = np.linspace(1000., 3200., 101)
gibbs = np.array([G_SiO2_liquid(T) for T in temperatures])

Ts = np.array(Ts)
Gs = np.array(Gs)
plt.plot(temperatures, gibbs)
plt.plot(Ts, Gs, marker='o', linestyle='None')

def G_fit(T, A, B, a, b):
    prm = params()
    prm.A = A
    prm.B = B
    prm.a = [a, b, 0., 0., 0.]
    
    return 4.184*(enthalpy(T, prm) - T*entropy(T, prm))




popt, pcov = curve_fit(G_fit, Ts, Gs)
print popt, pcov

plt.plot(temperatures, G_fit(temperatures, *popt))
plt.show()

prm = params()
prm.A = popt[0]
prm.B = popt[1]
prm.a = [popt[2], popt[3], 0., 0., 0.]


temperatures = np.linspace(500., 2000., 101)
Cps = np.empty_like(temperatures)

for i, T in enumerate(temperatures):
    crst.set_state(1.e5, T)
    Cps[i] = crst.heat_capacity_p

plt.plot(temperatures, Cps, label='crst')
plt.legend(loc='lower right')
plt.show()


plt.plot(Ts, Gdiffs, marker='o', linestyle='None')
plt.show()



def SiO2_liquidus_temperature(T, P):
    G_liq = G_fit(temperatures, *popt)
