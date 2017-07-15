import numpy as np
from scipy.optimize import fsolve, brentq
import matplotlib.pyplot as plt


import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    DKS_2013_liquids, \
    DKS_2013_solids, \
    SLB_2011, \
    HP_2011_ds62
from burnman import constants
import numpy as np
from scipy.optimize import fsolve, curve_fit
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

R = 8.31446

SiO2_liq = DKS_2013_liquids.SiO2_liquid()
stv = DKS_2013_solids.stishovite()
stv2 = SLB_2011.stishovite()

#SiO2_liq = DKS_2013_liquids.MgO_liquid()
#stv = DKS_2013_solids.periclase()
#stv2 = SLB_2011.periclase()
for P in [1.e5, 20.e9, 40.e9, 80.e9, 120.e9]:
    #for P in [1.e9]:
    temperatures = np.linspace(100., 3500. + P/1.e9*10., 101)
    S_liq = np.empty_like(temperatures)
    S_sol = np.empty_like(temperatures)
    S_sol2 = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        SiO2_liq.set_state(P, T)
        stv.set_state(P, T)
        stv2.set_state(P, T)
        S_liq[i] = SiO2_liq.S
        S_sol[i] = stv.S
        S_sol2[i] = stv2.S
        #S_liq[i] = SiO2_liq.heat_capacity_p
        #S_sol[i] = stv.heat_capacity_p
        #S_sol2[i] = stv2.heat_capacity_p
    #plt.plot(temperatures, S_liq-S_sol)
    plt.plot(temperatures, S_liq, linestyle='--')
    #plt.plot(temperatures, S_sol)
    plt.plot(temperatures, S_sol2, linestyle=':')

SiO2_data = np.loadtxt(fname='data/JANAF_SiO2.dat', unpack=True)
plt.plot(SiO2_data[0], SiO2_data[2], label='JANAF')

MgO_data = np.loadtxt(fname='data/JANAF_MgO.dat', unpack=True)
plt.plot(MgO_data[0], MgO_data[2], label='JANAF')

plt.ylim(0., 250.)
plt.show()
exit()
def heat_capacity(T, prm):
    return ( prm['a'][0] +
             prm['a'][1]*T +
             prm['a'][2]/(T*T) +
             prm['a'][3]/np.sqrt(T) +
             prm['a'][4]/(T*T*T) )

def enthalpy(T, prm):
    return prm['A'] + ( prm['a'][0]*( T - 298.15 ) +
                     0.5  * prm['a'][1]*( T*T - 298.15*298.15 ) +
                     -1.0 * prm['a'][2]*( 1./T - 1./298.15 ) +
                     2.0  * prm['a'][3]*(np.sqrt(T) - np.sqrt(298.15) ) +
                     -0.5 * prm['a'][4]*(1./(T*T) - 1./(298.15*298.15) ) )

def entropy(T, prm):
    return prm['B'] + ( prm['a'][0]*(np.log(T/298.15)) +
                     prm['a'][1]*(T - 298.15) +
                     -0.5 * prm['a'][2]*(1./(T*T) - 1./(298.15*298.15)) +
                     -2.0 * prm['a'][3]*(1./np.sqrt(T) - 1./np.sqrt(298.15)) +
                     -1./3. * prm['a'][4]*(1./(T*T*T) - 1./(298.15*298.15*298.15) ) )

def linear(x, a, b):
    return a + b*x

def func_Cp(T, a, b, c, d):
    return ( a + b*T + c/(T*T) + d/np.sqrt(T))

def func_Cp_full(T, a, b, c, d, e):
    return ( a + b*T + c/(T*T) + d/np.sqrt(T) + e/(T*T*T))

class SiO2_liquid(): # modified to fit melting curve...
    def __init__(self):
        self.prm0 = {'A': -890687.,
                     'B': 65.,
                     'a': np.array([7.177548e+01,   1.55168e-02,   2.489344e+05,
                                    -7.9021e+02, 0.])
                    }
        self.prm1 = self.prm0
        
    def set_state(self, P, T):
        if T < 1996.:
            self.prm = self.prm0
        else:
            self.prm = self.prm1
        self.gibbs = (enthalpy(T, self.prm) - T*entropy(T, self.prm))
        self.H = (enthalpy(T, self.prm))
        self.S = (entropy(T, self.prm))
        self.heat_capacity_p = (heat_capacity(T, self.prm))

class per():
    def __init__(self):
        self.prm = {'A': -143761.95,
                    'B': 52.053/4.184,
                    'a': np.array([5.12718918e+01,   3.06644690e-03,  -1.06741414e+06, -6.58899260e+01, 0.]) / 4.184
                }
    def set_state(self, P, T):
        self.gibbs = 4.184*(enthalpy(T, self.prm) - T*entropy(T, self.prm))
        self.H = 4.184*(enthalpy(T, self.prm))
        self.S = 4.184*(entropy(T, self.prm))
        self.heat_capacity_p = 4.184*(heat_capacity(T, self.prm))

class MgO_liquid():
    def __init__(self):
        self.prm = {'A': -129346.,
                    'B': 9.04,
                    'a': [1.29253257e+01,   9.44364230e-04,  -3.72469663e+05, 7.91765434e+00, 0.]
                    }
    def set_state(self, P, T):
        self.gibbs = 4.184*(enthalpy(T, self.prm) - T*entropy(T, self.prm))
        self.H = 4.184*(enthalpy(T, self.prm))
        self.S = 4.184*(entropy(T, self.prm))
        self.heat_capacity_p = 4.184*(heat_capacity(T, self.prm))

class MgO_liquid_tweak(): 
    def __init__(self):
        self.prm0 = {'A': 54178.83752719708, 'a': [2.839602347217558, 0.00341484389174643, -1502552.5191544946, -922.37318553906005, 0.0], 'B': 66.993850170202407}
        self.prm1 = {'A': 58562.842558220582, 'a': [5.7165061030308122, -1.0221081120981808e-05, -6269356.7755687768, -722.99967198621596, 0.0], 'B': 79.347606239344714}
        
    def set_state(self, P, T):
        if T < 1750.:
            self.prm = self.prm0
        else:
            self.prm = self.prm1
        self.gibbs = (enthalpy(T, self.prm) - T*entropy(T, self.prm))
        self.H = (enthalpy(T, self.prm))
        self.S = (entropy(T, self.prm))
        self.heat_capacity_p = (heat_capacity(T, self.prm))


class SiO2_liquid_tweak(): 
    def __init__(self):
        self.prm0 = {'A': 29317.598670186722, 'a': [3.14355762533431, 0.0010122323779954451, -17272.057503375127, -666.69586046751851, 0.0], 'B': 39.790628572419884}
        self.prm1 = {'A': -75643.03446637874, 'a': [138.45503432848142, -0.018434113923524766, 64483238.833861053, -5613.4139101984401, 0.0], 'B': -187.76074293402641}
        
    def set_state(self, P, T):
        if T < 2250.:
            self.prm = self.prm0
        else:
            self.prm = self.prm1
        self.gibbs = (enthalpy(T, self.prm) - T*entropy(T, self.prm))
        self.H = (enthalpy(T, self.prm))
        self.S = (entropy(T, self.prm))
        self.heat_capacity_p = (heat_capacity(T, self.prm))


MgO_tweak = MgO_liquid_tweak()
SiO2_tweak = SiO2_liquid_tweak()

en_liq = DKS_2013_liquids.MgSiO3_liquid()
MgO_liq = DKS_2013_liquids.MgO_liquid()
SiO2_liq = DKS_2013_liquids.SiO2_liquid()


per = SLB_2011.periclase()
per.params['F_0'] += -40036.738
per.params['q_0'] = 0.15

Tm = 3098.
per.set_state(1.e5, Tm)
MgO_liq.set_state(1.e5, Tm)
dS = 169.595 - 144.532 # JANAF
MgO_liq.property_modifiers = [['linear', {'delta_E': (per.gibbs - MgO_liq.gibbs) + Tm*(per.S - MgO_liq.S + dS), 'delta_S': (per.S - MgO_liq.S + dS), 'delta_V': 0.}]]


Hm = 8900.
Tm = 1999.
Sm = Hm/Tm
crst = HP_2011_ds62.crst()
crst.set_state(1.e5, Tm)
SiO2_liq.set_state(1.e5, Tm)
SiO2_liq.property_modifiers = [['linear', {'delta_E': (crst.gibbs - SiO2_liq.gibbs) + Tm*(crst.S - SiO2_liq.S + Sm), 'delta_S': (crst.S - SiO2_liq.S + Sm), 'delta_V': 0.}]]




SiO2_liq2 = SiO2_liquid()
MgO_liq2 = MgO_liquid()

qtz = SLB_2011.quartz()


temperatures = np.linspace(500., 3500., 101)
Cp_SiO2 = np.empty_like(temperatures)
Cp_SiO22 = np.empty_like(temperatures)
Cp_SiO23 = np.empty_like(temperatures)
Cp_qtz = np.empty_like(temperatures)
Cp_MgO = np.empty_like(temperatures)
Cp_MgO2 = np.empty_like(temperatures)
Cp_per = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    MgO_liq.set_state(1.e5, T)
    MgO_tweak.set_state(1.e5, T)
    MgO_liq2.set_state(1.e5, T)
    per.set_state(1.e5, T)

    
    en_liq.set_state(1.e5, T)
    
    SiO2_liq.set_state(1.e5, T)
    SiO2_tweak.set_state(1.e5, T)
    SiO2_liq2.set_state(1.e5, T)
    qtz.set_state(1.e5, T)
    
    Cp_SiO2[i] = en_liq.S - MgO_liq.S
    Cp_SiO22[i] = SiO2_liq.S #+ SiO2_tweak.heat_capacity_p
    Cp_SiO23[i] = SiO2_liq2.S
    Cp_qtz[i] = qtz.S
    
    Cp_MgO[i] = MgO_liq.heat_capacity_p + MgO_tweak.heat_capacity_p
    Cp_MgO[i] = MgO_liq.S #+ MgO_tweak.S
    Cp_MgO2[i] = MgO_liq2.S
    Cp_per[i] = per.S

# Get heat capacity estimate
mask = [i for (i, T) in enumerate(temperatures) if T < 1750.]
print curve_fit(func_Cp, temperatures[mask], (Cp_SiO2)[mask])

# Get entropy and enthalpy
Hm = 8900.
Tm = 1999.
Sm = Hm/Tm
crst = HP_2011_ds62.crst()
crst.set_state(1.e5, Tm)
SiO2_liq2.set_state(1.e5, Tm)
print crst.S - SiO2_liq2.S + Sm
print crst.gibbs - SiO2_liq2.gibbs


plt.plot(temperatures, Cp_SiO2, label='en - per')
plt.plot(temperatures, Cp_SiO22, label='DKS')
plt.plot(temperatures, Cp_SiO23, label='model liquid')
plt.plot(temperatures, Cp_qtz, label='quartz')
plt.legend(loc='lower right')
plt.show()


MgO_data = np.loadtxt(fname='data/JANAF_MgO.dat', unpack=True)

mask = [i for (i, T) in enumerate(temperatures) if T > 300. and T < 2500.]
print curve_fit(func_Cp, temperatures[mask], 1.126*Cp_per[mask]/4.184)

MgO_liq2.set_state(1.e5, 3100.)
print (169.488 - MgO_liq2.S)/4.184

plt.plot(temperatures, Cp_per, label='periclase')
plt.plot(temperatures, Cp_MgO, label='DKS')
plt.plot(temperatures, Cp_MgO2, label='parameterised liquid model')
plt.plot(MgO_data[0], MgO_data[2], marker='o', linestyle='None', label='JANAF')
plt.legend(loc='lower right')
plt.show()






MgO_liq = DKS_2013_liquids.MgO_liquid()
SiO2_liq = DKS_2013_liquids.SiO2_liquid()


MgO_liq2 = MgO_liquid()
SiO2_liq2 = SiO2_liquid()



mm = [[MgO_liq, MgO_liq2],
      [SiO2_liq, SiO2_liq2]]

'''
for (m0, m1) in mm:
    for P in [1.e5, 10.e9, 100.e9]:
        temperatures = np.linspace(2000., 3500. + P/1.e9*30., 101)
        Cp0 = np.empty_like(temperatures)
        Cp1 = np.empty_like(temperatures)
        for i, T in enumerate(temperatures):
            m0.set_state(P, T)
            m1.set_state(P, T)
            
            Cp0[i] = m0.heat_capacity_p
        plt.plot(temperatures, Cp0, label=str(P/1.e9)+' GPa')
        
    plt.legend(loc='lower right')
    plt.show()
'''


P = 1.e5
Cps = []

temperatures = np.linspace(500., 3500., 101)
Cp0 = np.empty_like(temperatures)
Cp1 = np.empty_like(temperatures)

for (m0, m1) in mm:
    for i, T in enumerate(temperatures):
        m0.set_state(P, T)
        m1.set_state(P, T)

        Cp0[i] = m0.heat_capacity_p
        Cp1[i] = m1.heat_capacity_p
    Cps.append([list(Cp0), list(Cp1)]) # for a deep copy
    plt.plot(temperatures, Cp0, label='DKS')
    plt.plot(temperatures, Cp1, label='Wu')
    plt.legend(loc='lower right')
    plt.show()
    

# MgO 
print 'MgO'
Cp0, Cp1 = Cps[0]
Cp0 = np.array(Cp0)
Cp1 = np.array(Cp1)
mask = [i for (i, Ti) in enumerate(temperatures) if (Ti < 1750.)] 
popt_MgO_0, pcov_MgO_0 = curve_fit(func_Cp, temperatures[mask], (Cp1-Cp0)[mask])


plt.plot(temperatures, func_Cp(temperatures, *popt_MgO_0))

mask = [i for (i, Ti) in enumerate(temperatures) if (Ti > 1750.) and (Ti < 3000.)]
popt_MgO_1, pcov_MgO_1 = curve_fit(func_Cp, temperatures[mask], (Cp1-Cp0)[mask])


plt.plot(temperatures, func_Cp(temperatures, *popt_MgO_1))

plt.plot(temperatures, (Cp1-Cp0))
plt.show()

# SiO2
print 'SiO2'
Cp0, Cp1 = Cps[1]
Cp0 = np.array(Cp0)
Cp1 = np.array(Cp1)
mask = [i for (i, Ti) in enumerate(temperatures) if (Ti < 2250.)] 
popt_SiO2_0, pcov_SiO2_0 = curve_fit(func_Cp, temperatures[mask], (Cp1-Cp0)[mask])


plt.plot(temperatures, func_Cp(temperatures, *popt_SiO2_0))

mask = [i for (i, Ti) in enumerate(temperatures) if (Ti > 2250.) and (Ti < 2750.)]
popt_SiO2_1, pcov_SiO2_1 = curve_fit(func_Cp, temperatures[mask], (Cp1-Cp0)[mask])


plt.plot(temperatures, func_Cp(temperatures, *popt_SiO2_1))

plt.plot(temperatures, (Cp1-Cp0))
plt.show()


# Calculate G0 and S0 if the gibbs free energy and entropies are zero at the melting point
Tm_MgO = 3098.
Tc_MgO = 1750.

prm_MgO_0 = {'A': 0.,
        'B': 0.,
        'a': [popt_MgO_0[0], popt_MgO_0[1], popt_MgO_0[2], popt_MgO_0[3], 0.]}
prm_MgO_1 = {'A': 0.,
        'B': 0.,
        'a': [popt_MgO_1[0], popt_MgO_1[1], popt_MgO_1[2], popt_MgO_1[3], 0.]}



prm_MgO_1['A'] = -enthalpy(Tm_MgO, prm_MgO_1)
prm_MgO_1['B'] = -entropy(Tm_MgO, prm_MgO_1)

prm_MgO_0['A'] = enthalpy(Tc_MgO, prm_MgO_1) - enthalpy(Tc_MgO, prm_MgO_0)
prm_MgO_0['B'] = entropy(Tc_MgO, prm_MgO_1) - entropy(Tc_MgO, prm_MgO_0)

print Tc_MgO
print prm_MgO_0
print prm_MgO_1

Tm_SiO2 = 1999.
Tc_SiO2 = 2250.

prm_SiO2_0 = {'A': 0.,
        'B': 0.,
        'a': [popt_SiO2_0[0], popt_SiO2_0[1], popt_SiO2_0[2], popt_SiO2_0[3], 0.]}
prm_SiO2_1 = {'A': 0.,
        'B': 0.,
        'a': [popt_SiO2_1[0], popt_SiO2_1[1], popt_SiO2_1[2], popt_SiO2_1[3], 0.]}

prm_SiO2_0['A'] = -enthalpy(Tm_SiO2, prm_SiO2_0)
prm_SiO2_0['B'] = -entropy(Tm_SiO2, prm_SiO2_0)

prm_SiO2_1['A'] = enthalpy(Tc_SiO2, prm_SiO2_0) - enthalpy(Tc_SiO2, prm_SiO2_1)
prm_SiO2_1['B'] = entropy(Tc_SiO2, prm_SiO2_0) - entropy(Tc_SiO2, prm_SiO2_1)

print Tc_SiO2
print prm_SiO2_0
print prm_SiO2_1
        
temperatures = np.linspace(500., 7000., 101)
mask = [i for (i, Ti) in enumerate(temperatures) if (Ti < 1750.)]
plt.subplot(131)
plt.title('Gibbs')
plt.plot(temperatures[mask], (enthalpy(temperatures, prm_MgO_0) - temperatures*entropy(temperatures, prm_MgO_0))[mask])
plt.subplot(132)
plt.title('Entropy')
plt.plot(temperatures[mask], (entropy(temperatures, prm_MgO_0))[mask])
plt.subplot(133)
plt.title('Heat capacity')
plt.plot(temperatures[mask], (heat_capacity(temperatures, prm_MgO_0))[mask])

mask = [i for (i, Ti) in enumerate(temperatures) if (Ti > 1750.)]
plt.subplot(131)
plt.plot(temperatures[mask], (enthalpy(temperatures, prm_MgO_1) - temperatures*entropy(temperatures, prm_MgO_1))[mask])
plt.subplot(132)
plt.plot(temperatures[mask], (entropy(temperatures, prm_MgO_1))[mask])
plt.subplot(133)
plt.plot(temperatures[mask], (heat_capacity(temperatures, prm_MgO_1))[mask])

plt.show()

mask = [i for (i, Ti) in enumerate(temperatures) if (Ti < 2250.)]
plt.subplot(131)
plt.title('Gibbs')
plt.plot(temperatures[mask], (enthalpy(temperatures, prm_SiO2_0) - temperatures*entropy(temperatures, prm_SiO2_0))[mask])
plt.subplot(132)
plt.title('Entropy')
plt.plot(temperatures[mask], (entropy(temperatures, prm_SiO2_0))[mask])
plt.subplot(133)
plt.title('Heat capacity')
plt.plot(temperatures[mask], (heat_capacity(temperatures, prm_SiO2_0))[mask])

mask = [i for (i, Ti) in enumerate(temperatures) if (Ti > 2250.)]
plt.subplot(131)
plt.plot(temperatures[mask], (enthalpy(temperatures, prm_SiO2_1) - temperatures*entropy(temperatures, prm_SiO2_1))[mask])
plt.subplot(132)
plt.plot(temperatures[mask], (entropy(temperatures, prm_SiO2_1))[mask])
plt.subplot(133)
plt.plot(temperatures[mask], (heat_capacity(temperatures, prm_SiO2_1))[mask])

plt.show()

# Let's take the high temperature tracks (as they result in generally lower entropies, 
mm = [[MgO_liq, MgO_liq2, prm_MgO_1],
      [SiO2_liq, SiO2_liq2, prm_SiO2_1]]
for (m0, m1, prm) in mm:
    for P in [1.e5, 10.e9, 100.e9, 200.e9]:
        temperatures = np.linspace(2000., 3500. + P/1.e9*30., 101)
        Cp = np.empty_like(temperatures)
        S0 = np.empty_like(temperatures)
        for i, T in enumerate(temperatures):
            m0.set_state(P, T)
            m1.set_state(P, T)
            
            Cp[i] = m0.heat_capacity_p
            S0[i] = m0.S

        plt.subplot(121)
        plt.title('Heat capacity')
        plt.plot(temperatures, Cp + heat_capacity(temperatures, prm), label=str(P/1.e9)+' GPa')
        plt.subplot(122)
        plt.title('Entropy')
        plt.plot(temperatures, S0 + entropy(temperatures, prm), label=str(P/1.e9)+' GPa')
    plt.legend(loc='lower right')
    plt.show()




