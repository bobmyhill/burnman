# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

from scipy.optimize import fsolve, curve_fit
import burnman
from HP_convert import *
from listify_xy_file import *
from fitting_functions import *

Fe3S = burnman.minerals.Fe_Si_O.Fe3S()
Z_Fe3S = 8.

data = listify_xy_file('data/Fe3S_room_temperature_volumes.dat')
PT_Fe3S = [data[0]*1.e9, data[1]]
V_Fe3S = data[2]*burnman.constants.Avogadro*1.e-30/Z_Fe3S
Verr_Fe3S = data[3]*burnman.constants.Avogadro*1.e-30/Z_Fe3S

Fe3S.params['Kprime_0'] = 4.0


guesses = [2.8e-5, 150.e9]
popt, pcov = curve_fit(fit_EoS_data(Fe3S, ['V_0', 'K_0']), PT_Fe3S, V_Fe3S, guesses, Verr_Fe3S)
print 'Fe3S V_0:', popt[0], 'm^3/mol'
print 'Fe3S K_0:', popt[1]/1.e9, 'GPa'
print 'Fe3S K\'_0:', 4., '[fixed]'

pressures = np.linspace(1.e5,100.e9, 101) 
volumes = np.empty_like(pressures)

T = 300.
for i, P in enumerate(pressures):
    Fe3S.set_state(P, T)
    volumes[i] = Fe3S.V

plt.plot(pressures, volumes)
plt.plot(PT_Fe3S[0], V_Fe3S, linestyle='None', marker='o')
plt.show()

data = listify_xy_file('data/Fe3S_HT_volumes_Morard_et_al_2008.dat')
P_Fe3S, Pminus, Pplus, T_Fe3S, Terr, aNaCl, aNaClerr, aFe, aFeerr, cFe, cFeerr, aFe3S, aFe3Serr, cFe3S, cFe3Serr = data
PT_Fe3S = [P_Fe3S*1.e9, T_Fe3S]
V_Fe3S = aFe3S*aFe3S*cFe3S*burnman.constants.Avogadro*1.e-30/Z_Fe3S
Verr_Fe3S = V_Fe3S*np.sqrt(2.*(aFe3Serr*aFe3Serr)/(aFe3S*aFe3S) +(cFe3Serr*cFe3Serr)/(cFe3S*cFe3S))

guesses = [4.e-5]
popt, pcov = curve_fit(fit_EoS_data(Fe3S, ['a_0']), PT_Fe3S, V_Fe3S, guesses, Verr_Fe3S)
print 'Fe3S a_0 (Morard et al., 2008):', popt[0]


PT_Fe3S = [(P_Fe3S+Pminus)*1.e9, T_Fe3S]
popt, pcov = curve_fit(fit_EoS_data(Fe3S, ['a_0']), PT_Fe3S, V_Fe3S, guesses, Verr_Fe3S)
print 'Fe3S a_0 min (Morard et al., 2008):', popt[0]

PT_Fe3S = [(P_Fe3S+Pplus)*1.e9, T_Fe3S]
popt, pcov = curve_fit(fit_EoS_data(Fe3S, ['a_0']), PT_Fe3S, V_Fe3S, guesses, Verr_Fe3S)
print 'Fe3S a_0 max (Morard et al., 2008):', popt[0]

# Thermal expansion of Fe3P (28 at % P)
# Data as found in Okamoto, 1990 (Bulletin of Alloy Phase Diagrams, vol. 11, #4)
# Originally from Fasiska and Zwell, 1967 (Trans Met. Soc. AIME.; not accessible online)

# See also Chen et al. (2007). The Fe3P method is preferred as Chen have an excess of Fe in their sample, and use Au as a pressure standard (and have large errors a1 = 3.0 +/- 1.3e-5 K^-1 and a2 = 2.8 +/- 1.5e-8 K^-2).

Tac = [[23., 0.91,   0.44592],
       [414., 0.9137, 0.45062],
       [678., 0.9174, 0.45299]]

Tac = np.array(zip(*Tac))

PT_Fe3P = [1.e5+Tac[0]*0., Tac[0]+273.15]
V_Fe3P = Tac[1]*Tac[2]*burnman.constants.Avogadro*1e-27/Z_Fe3S

guesses = [1.e-5, 4.e-5]
V_0_Fe3S = Fe3S.params['V_0']
popt, pcov = curve_fit(fit_EoS_data(Fe3S, ['V_0', 'a_0']), PT_Fe3P, V_Fe3P, guesses)
print 'Fe3P V_0:', popt[0]
print 'Fe3P a_0 (approximation for Fe3S):', popt[1]


Fe3S.params['V_0'] = V_0_Fe3S

P = 1.e5
temperatures = np.linspace(273.15, 1000.15, 101)
volumes = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    Fe3S.set_state(P, T)
    volumes[i] = Fe3S.V


plt.plot(temperatures, volumes)
plt.plot(PT_Fe3P[1], V_Fe3P, linestyle='None', marker='o')
plt.show()
