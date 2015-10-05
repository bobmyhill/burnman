import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from scipy.optimize import fsolve, curve_fit
from make_intermediate import *

# First, let's define the endmembers
pyrope = minerals.HP_2011_ds62.py()
majorite = minerals.HP_2011_ds62.maj()

# First, let's get some volumes of pyrope and majorite from
# Heinemann et al., 1997
Z = 8
NA = burnman.constants.Avogadro
Atom = 1e-30

p_py_obs = np.array([25.92, 49.85, 75.10, 99.98])/100.
V_obs = np.array([1511.167, 1509.386, 1506.802, 1503.654])/Z*NA*Atom
V_err = np.array([0.024, 0.044, 0.047, 0.052])/Z*NA*Atom

def symmetric(p_py, V_py, V_maj, V_ex):
    return p_py*V_py + (1.-p_py)*V_maj + p_py*(1.-p_py)*V_ex

guesses = [pyrope.params['V_0'], majorite.params['V_0'], 0.0]
popt, pcov = curve_fit(symmetric, p_py_obs, V_obs, guesses, V_err)
pyrope.params['V_0'], majorite.params['V_0'], V_ex_obs  = popt

variable = ['V_py', 'V_maj', 'V_ex']
for i, p in enumerate(popt):
    print variable[i], ':', p, '+/-', np.sqrt(pcov[i][i])

# Tweak majorite properties
print 'Pyrope properties'
print pyrope.params['V_0']
print pyrope.params['K_0']
print pyrope.params['Kprime_0']
print pyrope.params['Kdprime_0']

majorite.params['K_0'] = 160.e9
majorite.params['Kprime_0'] = 4.05
majorite.params['Kdprime_0'] = -majorite.params['Kprime_0']/majorite.params['K_0']

# Make symmetric solid solution
H_ex = 0.
S_ex = 0.
Sconf = 0.
V_ex = V_ex_obs
K_ex = -0.6e9
a_ex = 0.

pymaj_params = [H_ex/4., S_ex/4., Sconf, V_ex/4., K_ex, a_ex]
# First, let's define our intermediates
pymaj = make_intermediate(pyrope, majorite, pymaj_params)()

print 0.5*(pyrope.params['K_0'] + majorite.params['K_0'])
print pymaj.params['K_0']

# Now, let's set up a solid solution model
class pyrope_majorite_binary(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Pyrope-majorite binary'
        self.type='full_subregular'
        self.endmembers = [[pyrope,  'Mg3[Al]2Si3O12'],
                           [majorite, 'Mg3[Mg0.5Si0.5]2Si3O12']]
        self.intermediates=[[[pymaj, pymaj]]]
        
        burnman.SolidSolution.__init__(self, molar_fractions)

garnet = pyrope_majorite_binary()

pressures = np.linspace(1.e5, 10.e9, 101)
excess_volumes = np.empty_like(pressures)

garnet.set_composition([0.5, 0.5])

for i, P in enumerate(pressures):
    garnet.set_state(P, 298.15)
    excess_volumes[i] = garnet.excess_volume


plt.plot(pressures/1.e9, excess_volumes*1.e6)
plt.xlabel('Pressure (GPa)')
plt.ylabel('Excess volume (cm^3/mol)')
plt.show()



# RT from Wang et al
p_py = 0.62

P_obs = np.array([4.195, 6.258, 7.278, 0.0001, 0.0001, 
                  8.567, 8.390, 7.911, 7.845, 7.295,
                  6.663, 5.883, 4.922, 3.681, 2.763, 
                  2.696, 2.345, 2.046, 1.241])
V_obs = np.array([1470.9, 1456.2, 1447.8, 1509.6, 1509.1, 
                  1439.4, 1440.2, 1443.0, 1443.9, 1449.0,
                  1453.2, 1459.2, 1466.9, 1476.6, 1484.4,
                  1485.2, 1488.2, 1491.0, 1498.0])/Z*NA*Atom
V_err = np.array([0.6, 0.8, 0.5, 0.5, 0.3, 
                  0.4, 0.3, 0.2, 0.3, 0.3,
                  0.2, 0.2, 0.3, 0.3, 0.3,
                  0.3, 0.4, 0.3, 0.3])/Z*NA*Atom


volumes = np.empty_like(pressures)
garnet.set_composition([p_py, 1.-p_py])
for i, P in enumerate(pressures):
    garnet.set_state(P, 298.15)
    volumes[i] = garnet.V

plt.plot(pressures/1.e9, volumes)
plt.errorbar(P_obs, V_obs, yerr=V_err, linestyle='None')
plt.show()

# Check KV rule of thumb
garnet.set_composition([0.5, 0.5])
garnet.set_state(1.e5, 298.15)
K_T = garnet.K_T
V = garnet.V

KV_py = pyrope.params['K_0']*pyrope.params['V_0']
KV_maj = majorite.params['K_0']*majorite.params['V_0']

K_average = 0.5*(pyrope.params['K_0'] + majorite.params['K_0'])
KV_average = 0.5*(KV_py + KV_maj)
print 'Bulk modulus excess according to KV:', (KV_average/V - K_average)/1.e9, 'GPa'
print 'Bulk modulus excess is actually', (K_T - K_average)/1.e9, 'GPa'


print (K_T - K_average)/(KV_average/V - K_average)
