# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

# Tallon (1980) suggested that melting of simple substances was associated with an entropy change of
# Sfusion = burnman.constants.gas_constant*np.log(2.) + a*K_T*Vfusion
# Realising also that dT/dP = Vfusion/Sfusion, we can express the entropy 
# and volume of fusion in terms of the melting curve:
# Sfusion = burnman.constants.gas_constant*np.log(2.) / (1. - a*K_T*dTdP)
# Vfusion = Sfusion*dT/dP

from scipy.interpolate import interp1d
from scipy.optimize import fsolve, curve_fit
import burnman
from burnman import minerals
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()

from B20_FeSi import B20_FeSi
from B2_FeSi import B2_FeSi

B20 = B20_FeSi()
B2 = B2_FeSi() # high pressure phase



temperatures = np.linspace(300., 2000., 21)
dV = np.empty_like(temperatures)

for i, T  in enumerate(temperatures):
    P = burnman.tools.equilibrium_pressure([B20, B2], [1.0, -1.0], T)
    dV[i] = (B20.V - B2.V)/B20.V
    print(T, P/1.e9, B20.V/2.*1.e6, B2.V/2.*1.e6)
plt.plot(temperatures, dV)
plt.show()
