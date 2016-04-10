# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))


from scipy.interpolate import interp1d
from scipy.optimize import fsolve, curve_fit
import burnman
from burnman import minerals



from B1_wuestite import B1_wuestite
from liq_wuestite_AA1994 import liq_FeO

from fcc_iron import fcc_iron
from hcp_iron import hcp_iron
from liq_iron_AA1994 import liq_iron

from B2_FeSi import B2_FeSi
from B20_FeSi import B20_FeSi
from liq_FeSi_AA1994 import liq_FeSi

Fe_fcc = fcc_iron()
Fe_hcp = hcp_iron()
Fe_liq  = liq_iron()

FeO_B1 = B1_wuestite()
FeO_liq = liq_FeO()

FeSi_B20 = B20_FeSi()
FeSi_B2 = B2_FeSi()
FeSi_liq = liq_FeSi()

liquid_solid_pairs = [[FeO_B1, FeO_liq],
                      [Fe_fcc, Fe_liq],
                      [Fe_hcp, Fe_liq],
                      [FeSi_B2, FeSi_liq],
                      [FeSi_B20, FeSi_liq]]

pressures = np.linspace(1.e5, 200.e9, 101)



header='Pressure (GPa)'
data=[pressures/1.e9]
for pair in liquid_solid_pairs:
    solid = pair[0]
    liquid = pair[1]
    Tmelt = np.empty_like(pressures)
    Smelt = np.empty_like(pressures)
    dSdT = np.empty_like(pressures)

    P = 100.e9
    temperatures = np.linspace(1000., 5000., 101)
    S = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        liquid.set_state(P, T)
        S[i] = liquid.C_p
    plt.plot(temperatures, S)
    plt.title(liquid.params['name'])
    plt.show()

    '''
    for i, P in enumerate(pressures):
        Tmelt[i] = burnman.tools.equilibrium_temperature([solid, liquid], [1.0, -1.0], P, 2000.)
        Smelt[i] = liquid.S - solid.S

        deltaT = 500.
        liquid.set_state(P, Tmelt[i] - deltaT)
        solid.set_state(P, Tmelt[i] - deltaT)
        deltaG = liquid.gibbs - solid.gibbs
        meanS = deltaG/deltaT

        dSdT[i] = (Smelt[i] - meanS)/(deltaT/2.)

        
    plt.plot(pressures, Smelt, label=solid.params['name'])
    '''
    header = header+', '+solid.params['name']+' T_fusion (K)'
    header = header+', '+solid.params['name']+' S_fusion (J/K/mol)'
    header = header+', '+solid.params['name']+' dSdT_fusion (J/K^2/mol)'
    data.extend([Tmelt, Smelt, dSdT])


np.savetxt(fname='Fe_FeO_FeSi_melts.dat', X=zip(*data), fmt='%.4e', header=header)


plt.legend(loc='lower left')
plt.show()

