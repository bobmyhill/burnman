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



from B1_wuestite import B1_wuestite
from liq_wuestite_AA1994 import liq_FeO
from fcc_iron import fcc_iron
from hcp_iron import hcp_iron
from liq_iron_AA1994 import liq_iron


class Fe_FeO_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Fe-FeO solution'
        self.type='full_subregular'
        self.P_0=1.e5
        self.T_0=1650.
        self.n_atoms=1.
        self.endmembers = [[liq_iron(), '[Fe]'], [liq_FeO(), 'Fe[O]']]
        self.energy_interaction = [[[00.e3, 30.e3]]]
        self.volume_interaction = [[[0., 0.]]]
        burnman.SolidSolution.__init__(self, molar_fractions)



fcc = fcc_iron()
hcp = hcp_iron()
wus = B1_wuestite()
liq = Fe_FeO_liquid()

'''
fper=burnman.minerals.SLB_2011.ferropericlase()
fper.set_composition([1.0, 0.])
fper.set_state(1.e5, 300.)
print np.sqrt(fper.K_S/fper.rho)
print fper.K_T/1.e9
exit()
'''

def eqm_XT(args, P, Fe_phase, FeO_phase, liq_phase):
    X_FeO, T = args
    Fe_phase.set_state(P, T)
    FeO_phase.set_state(P, T)
    liq_phase.set_composition([1. - X_FeO, X_FeO])
    liq_phase.set_state(P, T)

    return [Fe_phase.gibbs - liq_phase.partial_gibbs[0],
            FeO_phase.gibbs - liq_phase.partial_gibbs[1]]



def Fe_eqm_T(arg, X_FeO, P, Fe_phase, liq_phase):
    T = arg[0]
    Fe_phase.set_state(P, T)
    liq_phase.set_composition([1. - X_FeO, X_FeO])
    liq_phase.set_state(P, T)

    return [Fe_phase.gibbs - liq_phase.partial_gibbs[0]]

def FeO_eqm_T(arg, X_FeO, P, FeO_phase, liq_phase):
    T = arg[0]
    
    FeO_phase.set_state(P, T)
    liq_phase.set_composition([1. - X_FeO, X_FeO])
    liq_phase.set_state(P, T)

    return [FeO_phase.gibbs - liq_phase.partial_gibbs[1]]



pressures = np.linspace(100.e9, 360.e9, 14)
temperatures = np.empty_like(pressures)
X_FeOs = np.empty_like(pressures)
for i, P in enumerate(pressures):
    print P/1.e9
    X, T = fsolve(eqm_XT, [0.1, 2000.], args=(P, hcp, wus, liq))
    X_FeOs[i] = X
    temperatures[i] = T

    
Fe_FeO_stability = burnman.tools.array_from_file('data/Ozawa_et_al_2011_Fe_FeO_phase_stability.dat')
plt.plot(pressures/1.e9, temperatures)
plt.errorbar(Fe_FeO_stability[0], Fe_FeO_stability[1], yerr=Fe_FeO_stability[2], marker='o', linestyle='None')
plt.show()

plt.plot(pressures/1.e9, X_FeOs)
plt.show()



P = 50.e9
X_FeO_eutectic, T = fsolve(eqm_XT, [0.1, 2000.], args=(P, fcc, wus, liq))



X_FeOs = np.linspace(0., X_FeO_eutectic, 11)
temperatures_hcp = np.empty_like(X_FeOs)
temperatures_fcc = np.empty_like(X_FeOs)

for i, X_FeO in enumerate(X_FeOs):
    temperatures_hcp[i] = fsolve(Fe_eqm_T, [1000.], args=(X_FeO, P, hcp, liq))[0]
    temperatures_fcc[i] = fsolve(Fe_eqm_T, [1000.], args=(X_FeO, P, fcc, liq))[0]

   
X_FeOs_2 = np.linspace(X_FeO_eutectic, 1.0, 11)
temperatures_wus = np.empty_like(X_FeOs_2)

for i, X_FeO in enumerate(X_FeOs_2):
    temperatures_wus[i] = fsolve(FeO_eqm_T, [1000.], args=(X_FeO, P, wus, liq))[0]

    
plt.plot(X_FeOs, temperatures_hcp)
plt.plot(X_FeOs, temperatures_fcc)
plt.plot(X_FeOs_2, temperatures_wus)

def wtO_to_mole_FeO(wtO):
    molar_mass_O =15.9994
    molar_mass_Fe = 55.845

    moles_O = wtO/molar_mass_O
    moles_Fe = (100.-wtO)/molar_mass_Fe

    moles_FeO = moles_O
    moles_Fe = moles_Fe - moles_O

    return moles_FeO/(moles_FeO + moles_Fe)

    
data = burnman.tools.array_from_file('data/Seagle_2008_low_eutectic_bounds.dat')
plt.plot(wtO_to_mole_FeO(data[2]), data[1], marker='o', linestyle='None', label="solid")
data = burnman.tools.array_from_file('data/Seagle_2008_high_eutectic_bounds.dat')
plt.plot(wtO_to_mole_FeO(data[2]), data[1], marker='o', linestyle='None', label="solid+liquid")
data = burnman.tools.array_from_file('data/Seagle_2008_high_liquidus_bounds.dat')
plt.plot(wtO_to_mole_FeO(data[2]), data[1], marker='o', linestyle='None', label="liquid")
plt.legend(loc='upper left')
plt.show()
