from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import inspect

if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman

def debye(args, A):
    m, v = args
    return (A*np.power(m, -0.5))*np.power(v,-0.5)

def debye2(args, A, B):
    s = args[0]
    return 1./(A*s + B)

mineralgroup = burnman.minerals.SLB_2011


avVs = []
avMs = []
Debyes = []
Ss = []
for m in dir(mineralgroup):
    mineral = getattr(mineralgroup, m)
    if inspect.isclass(mineral) and mineral!=burnman.solidsolution.SolidSolution and mineral!=burnman.Mineral and issubclass(mineral, burnman.Mineral) \
       and issubclass(mineral, burnman.mineral_helpers.HelperSpinTransition)==False :
        m1 = mineral()
        if m1.params.has_key('Debye_0'):
            avV = m1.params['V_0']/m1.params['n']
            avM = m1.params['molar_mass']/m1.params['n']
            m1.set_state(1.e5, 300.)
            S = m1.S/m1.params['n']
            avVs.append(avV)
            avMs.append(avM)
            Ss.append(S)
            Debyes.append(m1.params['Debye_0'])

avVs = np.array(avVs)
avMs = np.array(avMs)
Ss = np.array(Ss)
Debyes = np.array(Debyes)

popt, pcov = curve_fit(debye, [avVs, avMs], Debyes)
for i in xrange(len(popt)):
    print popt[i], '+/-', pcov[i][i]
print 'RMS:', np.sqrt(sum((debye([avVs, avMs], *popt) - Debyes)*(debye([avVs, avMs], *popt) - Debyes))/len(Debyes))
plt.plot(Debyes, debye([avVs, avMs], *popt), marker='o', linestyle='None')
plt.show()

# 2
popt, pcov = curve_fit(debye2, [Ss], Debyes)

for i in xrange(len(popt)):
    print popt[i], '+/-', pcov[i][i]

print 'RMS:', np.sqrt(sum((debye2([Ss], *popt) - Debyes)*(debye2([Ss], *popt) - Debyes))/len(Debyes))

plt.plot(debye2([Ss], *popt), Debyes, marker='o', linestyle='None')
plt.plot([0., 1200.], [0., 1200.])
plt.show()

print 1./popt[0], popt[1]/popt[0]


# 1./0.7 is closer to the linear array in SLB2011
def debye_HP2011(S):
    return 1./np.power(np.pi/6., 1./3.) * 10636./(S + 6.44)

plt.plot(debye_HP2011(Ss), Debyes, marker='o', linestyle='None')
plt.plot([0., 1200.], [0., 1200.])
plt.show()
