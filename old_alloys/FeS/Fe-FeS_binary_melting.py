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

FeS_VI = burnman.minerals.Fe_Si_O.FeS_VI()
FeS_liq = burnman.minerals.Fe_Si_O.FeS_liquid_new()

Fe_liq = burnman.minerals.Myhill_calibration_iron.liquid_iron()
Fe_fcc = burnman.minerals.Myhill_calibration_iron.fcc_iron()
Fe_hcp = burnman.minerals.Myhill_calibration_iron.hcp_iron()

class Fe_FeS_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Fe-FeS liquid'
        self.type='symmetric'
        self.endmembers = [[Fe_liq, '[Fe]'],[FeS_liq, 'Fe[S]']]
        self.enthalpy_interaction=[[0.0]]
        burnman.SolidSolution.__init__(self, molar_fractions)


liq = Fe_FeS_liquid()

# to find equilibrium with metallic iron 
# (ignoring any sulfur incorporation)
# mu Fe = mu Fe

# look over 1500 K from the melting point of pure iron
def Fe_liquidus(X, P, T, Fe_phase):# X = molar fraction Fe
    molar_fractions = [X[0], 1.-X[0]]
    liq.set_composition(molar_fractions)
    liq.set_state(P, T)
    Fe_phase.set_state(P, T)
    return Fe_phase.gibbs - liq.partial_gibbs[0]

def FeS_liquidus(X, P, T, FeS_phase):# X = molar fraction Fe
    if X[0] < 0. or X[0] > 1.:
        X[0] = 0.00001
    molar_fractions = [X[0], 1.-X[0]]
    liq.set_composition(molar_fractions)
    liq.set_state(P, T)
    FeS_phase.set_state(P, T)
    return FeS_phase.gibbs - liq.partial_gibbs[1]


molar_mass_S = 32.065
molar_mass_Fe = 55.845

# Find the Fe liquidus
pressures = np.linspace(30.e9, 120.e9, 7)
for P in pressures:
    T_hcp_liq = fsolve(eqm_temperature([Fe_liq, Fe_hcp], [1.0, -1.0]), 1400., args=(P))[0]
    print 'hcp:', P/1.e9, T_hcp_liq-273 , Fe_liq.S - Fe_hcp.S
    T_fcc_liq = fsolve(eqm_temperature([Fe_liq, Fe_fcc], [1.0, -1.0]), 1400., args=(P))[0]
    print 'fcc:', P/1.e9, T_fcc_liq -273, Fe_liq.S -Fe_fcc.S

    if T_fcc_liq > T_hcp_liq:
        T_hcp_fcc = fsolve(eqm_temperature([Fe_hcp, Fe_fcc], [1.0, -1.0]), 1400., args=(P))[0]
    else:
        T_hcp_fcc = 100000.

    Tmelt = max(T_fcc_liq, T_hcp_liq)
    temperatures = np.linspace(Tmelt-1500., Tmelt, 101.)
    X_Fe = np.empty_like(temperatures)
    wt_percent_S = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        if T < T_hcp_fcc:
            stable_Fe_phase = Fe_hcp
        else:
            stable_Fe_phase = Fe_fcc
        X_Fe[i] = fsolve(Fe_liquidus, [0.001], args=(P, T, stable_Fe_phase))[0]
        wt_percent_S[i] = 100.*(1.-X_Fe[i])*molar_mass_S/(molar_mass_Fe + (1.-X_Fe[i])*molar_mass_S)

    plt.plot(wt_percent_S, temperatures, label=str(P/1.e9)+'GPa')

# Find the FeS liquidus
for P in pressures:
    Tmelt = fsolve(eqm_temperature([FeS_liq, FeS_VI], [1.0, -1.0]), 1400., args=(P))[0]
    print Tmelt
    temperatures = np.linspace(Tmelt-1500., Tmelt, 101.)
    X_Fe = np.empty_like(temperatures)
    wt_percent_S = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        X_Fe[i] = fsolve(FeS_liquidus, [0.999], args=(P, T, FeS_VI))[0]
        wt_percent_S[i] = 100.*(1.-X_Fe[i])*molar_mass_S/(molar_mass_Fe + (1.-X_Fe[i])*molar_mass_S)
    plt.plot(wt_percent_S, temperatures, label=str(P/1.e9)+'GPa')





plt.xlabel('S (wt %)')
plt.xlim(0., 36.47)
plt.ylabel('T (K)')
plt.legend(loc='lower right')
plt.show()


