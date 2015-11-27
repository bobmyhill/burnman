# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

from scipy.interpolate import interp1d
from scipy.optimize import fsolve, curve_fit
import burnman
from burnman import minerals
from listify_xy_file import *
from fitting_functions import *
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()

class liq_iron (burnman.Mineral):
    def __init__(self):
        formula='Fe'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Liquid iron',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': -4165.,
            'V_0': 7.2e-6, # 6.733e-6 ,
            'K_0': 128.0e9, # 166.e9 ,
            'Kprime_0': 5.2, # 5.32 ,
            'Debye_0': 250. ,
            'grueneisen_0': 1.72 ,
            'q_0': 0.0 ,
            'Cv_el': 2.7, # 2.7,
            'T_el': 6500., # 6500.
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        burnman.Mineral.__init__(self)


from fcc_iron import fcc_iron
from hcp_iron import hcp_iron

bcc = minerals.HP_2011_ds62.iron()
fcc = fcc_iron()
hcp = hcp_iron()
liq = liq_iron()


hcp.set_state(97.e9, 3615.)
liq.set_state(97.e9, 3615.)
liq.params['F_0'] = liq.params['F_0'] + (hcp.gibbs - liq.gibbs)

temperatures = np.linspace(1800., 3000., 101)
Cps = np.empty_like(temperatures)
volumes = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    liq.set_state(1.e9, T)
    Cps[i] = liq.C_p
    volumes[i] = liq.V

plt.plot(temperatures, Cps)
plt.show()
plt.plot(temperatures, volumes)
plt.show()

# Tallon (1980) suggested that melting of simple substances was associated with an entropy change of
# Sfusion = burnman.constants.gas_constant*np.log(2.) + a*K_T*Vfusion
# Realising also that dT/dP = Vfusion/Sfusion, we can express the entropy 
# and volume of fusion in terms of the melting curve:
# Sfusion = burnman.constants.gas_constant*np.log(2.) / (1. - a*K_T*dTdP)
# Vfusion = Sfusion*dT/dP


melting_curve_data = listify_xy_file('data/Anzellini_2013_Fe_melting_curve.dat')
melting_temperature = interp1d(melting_curve_data[0]*1.e9, 
                               melting_curve_data[1], 
                               kind='cubic')


# Now we plot the entropy and volume of the liquid phase along the melting curve
pressures = np.linspace(1.e5, 250.e9, 26)
Sfusion = np.empty_like(pressures)
Vfusion = np.empty_like(pressures)
Smelt = np.empty_like(pressures)
Smelt_model = np.empty_like(pressures)
Vmelt = np.empty_like(pressures)
Vmelt_model = np.empty_like(pressures)
Tmelt_model = np.empty_like(pressures)

Cp_1bar = 46.024
S_1bar = 99.765
Tm_1bar = 1809.

for i, P in enumerate(pressures):
    if P > 97.e9:
        Fe_phase = hcp
    else:
        Fe_phase = fcc

    dP = 100. # Pa
    dT = melting_temperature(P + dP/2.) - melting_temperature(P - dP/2.)
    dTdP = dT/dP
    T = melting_temperature(P)
    Fe_phase.set_state(P, T)
    aK_T = Fe_phase.alpha*Fe_phase.K_T
    Sfusion[i] = burnman.constants.gas_constant*np.log(2.) / (1. - aK_T*dTdP)
    Vfusion[i] = Sfusion[i]*dTdP

    Smelt[i] = Fe_phase.S + Sfusion[i]
    Vmelt[i] = Fe_phase.V + Vfusion[i]

    liq.set_state(P, T)
    Smelt_model[i] = liq.S
    Vmelt_model[i] = liq.V
    
    print int(P/1.e9), Sfusion[i], Vfusion[i]*1.e6, aK_T/1.e9, Fe_phase.S, (Fe_phase.K_T - liq.K_T)/1.e9
    Tmelt_model[i] = burnman.tools.equilibrium_temperature([Fe_phase, liq], [1.0, -1.0], P, 1800.)
    


plt.plot(pressures/1.e9, Smelt, marker='o', linestyle='None')
plt.plot(pressures/1.e9, Smelt_model)
plt.show()


plt.plot(pressures/1.e9, Vmelt, marker='o', linestyle='None')
plt.plot(pressures/1.e9, Vmelt_model)
plt.show()


fig1 = mpimg.imread('data/Anzellini_2013_Fe_melting.png')  # Uncomment these two lines if you want to overlay the plot on a screengrab from SLB2011
plt.imshow(fig1, extent=[0., 230., 1200., 5200.], aspect='auto')
plt.plot(pressures/1.e9, melting_temperature(pressures), marker='o', linestyle='None')
plt.plot(pressures/1.e9, Tmelt_model)
plt.show()




exit()


