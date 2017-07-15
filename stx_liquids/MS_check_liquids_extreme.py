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


class SiO2_liquid(): # modified to fit melting curve...
    def __init__(self):
        self.prm0 = {'A': -214339.36,
                     'B': 12.148448,
                     'a': 1.1027*np.array([19.960229, 0.e-3, -5.8684512e5, -89.553776, 0.66938861e8])
                    }
        #self.prm0 = {'A': -214339.36,
        #             'B': 12.148448,
        #             'a': np.array([11.4, 5.74e-3, -5.8684512e5, -89.553776, 0.66938861e8])
        #            }
        self.prm1 = self.prm0
        #self.prm1 = {'A': -2.20401092e+05,
        #            'B': 4.78243403e+00,
        #            'a': [1.82964211e+01,   6.76081909e-04, 0., 0., 0.]
        #            }
        
    def set_state(self, P, T):
        if T < 1996.:
            self.prm = self.prm0
        else:
            self.prm = self.prm1
        self.gibbs = 4.184*(enthalpy(T, self.prm) - T*entropy(T, self.prm))
        self.H = 4.184*(enthalpy(T, self.prm))
        self.S = 4.184*(entropy(T, self.prm))
        self.heat_capacity_p = 4.184*(heat_capacity(T, self.prm))

en_liq = DKS_2013_liquids.MgSiO3_liquid()
per_liq = DKS_2013_liquids.MgO_liquid()
SiO2_liq = DKS_2013_liquids.SiO2_liquid()
SiO2_liq2 = SiO2_liquid()

temperatures = np.linspace(500., 3000., 101)
Cp = np.empty_like(temperatures)
Cp2 = np.empty_like(temperatures)
Cp3 = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    per_liq.set_state(1.e5, T)
    en_liq.set_state(1.e5, T)
    SiO2_liq.set_state(1.e5, T)
    SiO2_liq2.set_state(1.e5, T)
    Cp[i] = en_liq.heat_capacity_p - per_liq.heat_capacity_p
    Cp2[i] = SiO2_liq.heat_capacity_p
    Cp3[i] = SiO2_liq2.heat_capacity_p

plt.plot(temperatures, Cp)
plt.plot(temperatures, Cp2)
plt.plot(temperatures, Cp3)
plt.show()

'''
class SiO2_liquid2(): # Orig
    def __init__(self):
        self.prm0 = {'A': -214339.36,
                     'B': 12.148448,
                     'a': [19.960229, 0.e-3, -5.8684512e5, -89.553776, 0.66938861e8]
                    }
        self.prm1 = {'A': -221471.21,
                     'B': 2.3702523,
                     'a': [20.5, 0., 0., 0., 0.]
                    }
    def set_state(self, P, T):
        if T < 1996.:
            self.prm = self.prm0
        else:
            self.prm = self.prm1
            
        self.gibbs = 4.184*(enthalpy(T, self.prm) - T*entropy(T, self.prm))
        self.H = 4.184*(enthalpy(T, self.prm))
        self.S = 4.184*(entropy(T, self.prm))
        self.heat_capacity_p = 4.184*(heat_capacity(T, self.prm))
'''

class MgO_liquid():
    def __init__(self):
        self.prm = {'A': -130340.58,
                    'B': 6.4541207,
                    'a': [17.398557, 0.288e-3, 1.2494063e5, -70.793260, 0.013968958e8]
                    }
    def set_state(self, P, T):
        self.gibbs = 4.184*(enthalpy(T, self.prm) - T*entropy(T, self.prm))
        self.H = 4.184*(enthalpy(T, self.prm))
        self.S = 4.184*(entropy(T, self.prm))
        self.heat_capacity_p = 4.184*(heat_capacity(T, self.prm))

'''
class MgO_liquid2(): # orig
    def __init__(self):
        self.prm = {'A': -130340.58,
                    'B': 6.4541207,
                    'a': [17.398557, -0.751e-3, 1.2494063e5, -70.793260, 0.013968958e8]
                    }
    def set_state(self, P, T):
        self.gibbs = 4.184*(enthalpy(T, self.prm) - T*entropy(T, self.prm))
        self.H = 4.184*(enthalpy(T, self.prm))
        self.S = 4.184*(entropy(T, self.prm))
        self.heat_capacity_p = 4.184*(heat_capacity(T, self.prm))
'''


def linear(x, a, b):
    return a + b*x

def func_Cp(T, a, b, c, d):
    return ( a + b*T + c/(T*T) + d/np.sqrt(T))


MgO_liq = DKS_2013_liquids.MgO_liquid()
SiO2_liq = DKS_2013_liquids.SiO2_liquid()


MgO_liq2 = MgO_liquid()
SiO2_liq2 = SiO2_liquid()

'''
MgO_liq.set_state(1.e5, 2900.)
Cp0 = MgO_liq.heat_capacity_p
MgO_liq.set_state(1.e5, 2901.)
print (MgO_liq.heat_capacity_p - Cp0)/4.184
MgO_liq2.set_state(1.e5, 1800.)
print MgO_liq2.heat_capacity_p
exit()
'''



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
        #S1[i] = 
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

mask = [i for (i, Ti) in enumerate(temperatures) if (Ti > 2250.) and (Ti < 3000.)]
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


mm = [[MgO_liq, MgO_liq2, prm_MgO_1],
      [SiO2_liq, SiO2_liq2, prm_SiO2_1]]
for (m0, m1, prm) in mm:
    for P in [1.e5, 10.e9, 100.e9, 200.e9]:
        temperatures = np.linspace(2000., 3500. + P/1.e9*30., 101)
        for i, T in enumerate(temperatures):
            m0.set_state(P, T)
            m1.set_state(P, T)
            
            Cp0[i] = m0.heat_capacity_p
        plt.plot(temperatures, Cp0 + heat_capacity(temperatures, prm), label=str(P/1.e9)+' GPa')
        
    plt.legend(loc='lower right')
    plt.show()




