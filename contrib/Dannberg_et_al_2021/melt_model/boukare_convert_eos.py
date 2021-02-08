from __future__ import absolute_import

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))

import burnman
from burnman import Mineral
from burnman.processchemistry import dictionarize_formula, formula_mass

from model_parameters import *
from eos_and_averaging import thermodynamic_properties

Pref = 100.e9
class mpv (Mineral):
    def __init__(self):
        formula = 'MgSiO3'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'mpv',
            'formula': formula,
            'equation_of_state': 'mod_hp_tmt',
            'Pref': Pref,
            'H_Pref': -1442310.0,
            'S_Pref': 62.6,
            'Cp_Pref': [0., 0., 0., 0.],
            'V_0': 2.445e-05,
            'a_0': 1.87e-05,
            'K_0': 2.51e+11,
            'Kprime_0': 4.14,
            'Kdprime_0': -1.6e-11,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}
        Mineral.__init__(self)


class fpv (Mineral):
    def __init__(self):
        formula = 'FeSiO3'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'fpv',
            'formula': formula,
            'equation_of_state': 'mod_hp_tmt',
            'Pref': Pref,
            'H_Pref': -1082910.0,
            'S_Pref': 95.0,
            'Cp_Pref': [0., 0., 0., 0.],
            'V_0': 2.534e-05,
            'a_0': 1.87e-05,
            'K_0': 2.81e+11,
            'Kprime_0': 4.14,
            'Kdprime_0': -1.6e-11,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}
        Mineral.__init__(self)

class per (Mineral):
    def __init__(self):
        formula = 'MgO'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'per',
            'formula': formula,
            'equation_of_state': 'mod_hp_tmt',
            'Pref': Pref,
            'H_Pref': -601570.0,
            'S_Pref': 26.5,
            'Cp_Pref': [0., 0., 0., 0.],
            'V_0': 1.125e-05,
            'a_0': 3.11e-05,
            'K_0': 1.616e+11,
            'Kprime_0': 3.95,
            'Kdprime_0': -2.4e-11,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}
        Mineral.__init__(self)


class fper (Mineral):
    def __init__(self):
        formula = 'FeO'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'fper',
            'formula': formula,
            'equation_of_state': 'mod_hp_tmt',
            'Pref': Pref,
            'H_Pref': -262240.0,
            'S_Pref': 58.6,
            'Cp_Pref': [0., 0., 0., 0.],
            'V_0': 1.206e-05,
            'a_0': 3.22e-05,
            'K_0': 1.52e+11,
            'Kprime_0': 4.9,
            'Kdprime_0': -3.2e-11,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}
        Mineral.__init__(self)


b_wus = burnman.minerals.boukare.fe_bridgmanite_boukare()
b_per = burnman.minerals.boukare.mg_bridgmanite_boukare()
b_mpv = burnman.minerals.boukare.periclase_boukare()
b_fpv = burnman.minerals.boukare.wuestite_boukare()


#s_wus = burnman.minerals.SLB_2011.fe_bridgmanite()
#s_per = burnman.minerals.SLB_2011.mg_bridgmanite()
#s_mpv = burnman.minerals.SLB_2011.periclase()
#s_fpv = burnman.minerals.SLB_2011.wuestite()

hp_wus = burnman.minerals.HHPH_2013.fper()
hp_per = burnman.minerals.HHPH_2013.per()
hp_mpv = burnman.minerals.HHPH_2013.mpv()
hp_fpv = burnman.minerals.HHPH_2013.fpv()


wus = fper()
per = per()
mpv = mpv()
fpv = fpv()

from scipy.optimize import curve_fit
func_Cp = lambda T, a, b, c, d: a + b*T + c/T/T + d/np.sqrt(T)
temperatures = np.linspace(300., 2000., 101)
pressures = Pref + 0.*temperatures
print('We start by taking the heat capacities from the Holland et al., 2013 dataset'
      'and fit the new heat capacity terms using them.')
print('New heat capacities at a reference pressure of {0} GPa:'.format(Pref/1.e9))
for m, m_mod in [[hp_mpv, mpv],
                 [hp_fpv, fpv],
                 [hp_per, per],
                 [hp_wus, wus]]:
    heat_capacities = m.evaluate(['C_p'], pressures, temperatures)[0]

    guesses = m.params['Cp']
    popt, _ = curve_fit(func_Cp, temperatures, heat_capacities, guesses)
    m_mod.params['Cp_Pref'] = popt
    print('{0} {1}'.format(m.name, popt))
print('(Calibrated between {0} and {1} K)'.format(temperatures[0], temperatures[-1]))


mbr_MgO = [0.,  0.581]
mbr_FeO = [0.908, 0.]
mbr_SiO2 = [0.092, 0.419]



# Now for the fitting
# First, the Fe endmember
x_FeO = mbr_FeO[0]
x_SiO2 = mbr_SiO2[0]

n_fpv = x_SiO2
n_wus = (x_FeO - x_SiO2)

with warnings.catch_warnings(record=True) as w:
    burnman.tools.check_eos_consistency(fpv, P=80.e9, T=4000., verbose=True)


hp_fe_mantle = burnman.CombinedMineral([hp_fpv, hp_wus], [n_fpv, n_wus])
fe_mantle = burnman.CombinedMineral([fpv, wus], [n_fpv, n_wus])
b_fe_mantle = burnman.CombinedMineral([b_fpv, b_wus], [n_fpv, n_wus])

temperatures = np.linspace(300., 7000., 101)
pressures = 100.e9 + 0.*temperatures

heat_capacities = hp_fe_mantle.evaluate(['C_p'], pressures, temperatures)[0]
plt.plot(temperatures, heat_capacities, linestyle=':', color='blue', label='Mg endmember, HHPH2013 dataset (Mod. Tait)')
heat_capacities = fe_mantle.evaluate(['C_p'], pressures, temperatures)[0]
plt.plot(temperatures, heat_capacities, linestyle='-', color='blue', label='Fe endmember, New HP Modified Tait')
#heat_capacities = b_fe_mantle.evaluate(['C_p'], pressures, temperatures)[0]
#plt.plot(temperatures, heat_capacities, linestyle='-')



# Now the Mg endmember
x_MgO = mbr_MgO[1]
x_SiO2 = mbr_SiO2[1]

n_mpv = x_SiO2
n_per = (x_MgO - x_SiO2)

hp_mg_mantle = burnman.CombinedMineral([hp_mpv, hp_per], [n_mpv, n_per])
mg_mantle = burnman.CombinedMineral([mpv, per], [n_mpv, n_per])
b_mg_mantle = burnman.CombinedMineral([b_mpv, b_per], [n_mpv, n_per])


heat_capacities = hp_mg_mantle.evaluate(['C_p'], pressures, temperatures)[0]
plt.plot(temperatures, heat_capacities, linestyle=':', color='red', label='Mg endmember, HHPH2013 dataset (Mod. Tait)')
heat_capacities = mg_mantle.evaluate(['C_p'], pressures, temperatures)[0]
plt.plot(temperatures, heat_capacities, linestyle='-', color='red', label='Mg endmember, New HP Modified Tait')
#heat_capacities = b_mg_mantle.evaluate(['C_p'], pressures, temperatures)[0]
#plt.plot(temperatures, heat_capacities, linestyle='-')

plt.ylim(0., )
plt.legend()
plt.show()

"""
for m in [mpv, fpv, per, wus]:
    print(m.params)
"""

rxn = burnman.CombinedMineral([mpv, per, fpv, wus], [-1., 1., 1., -1.])
for T in np.linspace(2000., 6000., 5):
    pressures = np.linspace(100.e9, 150.e9, 101)
    temperatures = T + 0.*pressures
    G = rxn.evaluate(['gibbs'], pressures, temperatures)[0]
    KD = np.exp(-G/(burnman.constants.gas_constant*T))
    plt.plot(pressures, KD, label='{0} K'.format(T))
plt.legend()
plt.show()


# Finally, let's create our combined mantle endmembers:
melting_reference_pressure = 120.e9 # Pa
melting_temperatures=np.array([4821.2, 3470.0]) # K
melting_entropies=np.array([34.33, 33.77]) # J/K/mol-cations
melting_volumes=np.array([9.29e-08, 1.51e-07]) # m^3/mol-cations


formula = mg_mantle.formula
dP = 1.e3
Ks = mg_mantle.evaluate(['K_T'], [1.e5-dP, 1.e5, 1.e5+dP], [298.15, 298.15, 298.15])[0]
Kprime = (Ks[2] - Ks[0])/(2.*dP)

dP = 1.e6
Ks = mg_mantle.evaluate(['K_T'], [1.e5-dP, 1.e5, 1.e5+dP], [298.15, 298.15, 298.15])[0]
Kdprime = (Ks[2] - 2.*Ks[1] + Ks[0])/(dP*dP) + 4.1e-12

mg_mantle.set_state(1.e5, 298.15)
mg_melt = burnman.Mineral(params = {'name': 'mg mantle',
                                   'formula': formula,
                                   'equation_of_state': 'mod_hp_tmt',
                                   'Pref': Pref,
                                   'H_Pref': mpv.params['H_Pref']*n_mpv + per.params['H_Pref']*n_per,
                                   'S_Pref': mpv.params['S_Pref']*n_mpv + per.params['S_Pref']*n_per,
                                   'Cp_Pref': mpv.params['Cp_Pref']*n_mpv + per.params['Cp_Pref']*n_per,
                                   'V_0': mg_mantle.V,
                                   'a_0': mg_mantle.alpha,
                                   'K_0': mg_mantle.K_T,
                                   'Kprime_0': Kprime,
                                   'Kdprime_0': Kdprime,
                                   'n': sum(formula.values()),
                                   'molar_mass': formula_mass(formula)})

print(formula)

pressures = np.linspace(140.e9, 150.e9, 101)
temperatures = 300. + 0.*pressures
plt.plot(pressures, mg_mantle.evaluate(['V'], pressures, temperatures)[0])
plt.plot(pressures, mg_melt.evaluate(['V'], pressures, temperatures)[0])
plt.show()

mg_mantle.set_state(melting_reference_pressure, melting_temperatures[0])
mg_melt.set_state(melting_reference_pressure, melting_temperatures[0])

mg_melt.params['V_0'] *= (mg_mantle.V + melting_volumes[0])/mg_melt.V

mg_melt.set_state(melting_reference_pressure, melting_temperatures[0])
mg_melt.params['S_Pref'] += (mg_mantle.S + melting_entropies[0]) - mg_melt.S

mg_melt.set_state(melting_reference_pressure, melting_temperatures[0])
mg_melt.params['H_Pref'] += mg_mantle.gibbs - mg_melt.gibbs


mg_melt.set_state(melting_reference_pressure, melting_temperatures[0])
print('Melting temperature: {1} K ({0} GPa)'.format(mg_melt.pressure/1.e9, mg_melt.temperature))
print(f'V_fusion: {mg_melt.V - mg_mantle.V}')
print(f'S_fusion: {mg_melt.S - mg_mantle.S}')
print(f'Check eqm (should be 0): {mg_melt.gibbs - mg_mantle.gibbs}')



# AND THE SAME FOR FE!!

formula = fe_mantle.formula
dP = 1.e3
Ks = fe_mantle.evaluate(['K_T'], [1.e5-dP, 1.e5, 1.e5+dP], [298.15, 298.15, 298.15])[0]
Kprime = (Ks[2] - Ks[0])/(2.*dP)

dP = 1.e6
Ks = fe_mantle.evaluate(['K_T'], [1.e5-dP, 1.e5, 1.e5+dP], [298.15, 298.15, 298.15])[0]
Kdprime = (Ks[2] - 2.*Ks[1] + Ks[0])/(dP*dP) + 4.1e-12


# NOTE: The commented out line is what we should strictly use
# However, the Fe_melt and Mg_melt have similar Cps over the range of temperatures
# considered in this study, so we here choose to make the melt Cp independent of composition at Pref.
fe_mantle.set_state(1.e5, 298.15)
fe_melt = burnman.Mineral(params = {'name': 'fe mantle',
                                   'formula': formula,
                                   'equation_of_state': 'mod_hp_tmt',
                                   'Pref': Pref,
                                   'H_Pref': 0,
                                   'S_Pref': 0,
                                   'Cp_Pref': mpv.params['Cp_Pref']*n_mpv + per.params['Cp_Pref']*n_per,
                                   #'Cp_Pref': fpv.params['Cp_Pref']*n_fpv + wus.params['Cp_Pref']*n_wus,
                                   'V_0': fe_mantle.V,
                                   'a_0': fe_mantle.alpha,
                                   'K_0': fe_mantle.K_T,
                                   'Kprime_0': Kprime,
                                   'Kdprime_0': Kdprime,
                                   'n': sum(formula.values()),
                                   'molar_mass': formula_mass(formula)})

print(formula)

pressures = np.linspace(140.e9, 150.e9, 101)
temperatures = 300. + 0.*pressures
plt.plot(pressures, fe_mantle.evaluate(['V'], pressures, temperatures)[0])
plt.plot(pressures, fe_melt.evaluate(['V'], pressures, temperatures)[0])
plt.show()

fe_mantle.set_state(melting_reference_pressure, melting_temperatures[1])
fe_melt.set_state(melting_reference_pressure, melting_temperatures[1])

fe_melt.params['V_0'] *= (fe_mantle.V + melting_volumes[1])/fe_melt.V

fe_melt.set_state(melting_reference_pressure, melting_temperatures[1])
fe_melt.params['S_Pref'] += (fe_mantle.S + melting_entropies[1]) - fe_melt.S

fe_melt.set_state(melting_reference_pressure, melting_temperatures[1])
fe_melt.params['H_Pref'] += fe_mantle.gibbs - fe_melt.gibbs


fe_melt.set_state(melting_reference_pressure, melting_temperatures[1])
print('Melting temperature: {1} K ({0} GPa)'.format(fe_melt.pressure/1.e9, fe_melt.temperature))
print(f'V_fusion: {fe_melt.V - fe_mantle.V}')
print(f'S_fusion: {fe_melt.S - fe_mantle.S}')
print(f'Check eqm (should be 0): {fe_melt.gibbs - fe_mantle.gibbs}')





temperatures = np.linspace(300., 7000., 101)
pressures = 100.e9 + 0.*temperatures
heat_capacities = fe_melt.evaluate(['C_p'], pressures, temperatures)[0]
plt.plot(temperatures, heat_capacities, linestyle='--', color='blue', linewidth=3)
heat_capacities = mg_melt.evaluate(['C_p'], pressures, temperatures)[0]
plt.plot(temperatures, heat_capacities, linestyle='--', color='red', linewidth=3)



heat_capacities = fe_mantle.evaluate(['C_p'], pressures, temperatures)[0]
plt.plot(temperatures, heat_capacities, linestyle='-', color='blue')

heat_capacities = mg_mantle.evaluate(['C_p'], pressures, temperatures)[0]
plt.plot(temperatures, heat_capacities, linestyle='-', color='red')

#plt.ylim(0., )
plt.show()


mg_melt.set_state(100.e9, 3000.)

print('Compare the molar isobaric heat capacities for the Mg-bearing melt')
print(f'New melt model (standalone function): {thermodynamic_properties(100.e9, 3000., mg_mantle_melt_params)["molar_C_p"]}')
print(f'New melt model (burnman): {mg_melt.C_p}')

Cp_sol_eqv = (n_mpv*thermodynamic_properties(100.e9, 3000., mpv_params)["molar_C_p"] +
              n_per*thermodynamic_properties(100.e9, 3000., per_params)["molar_C_p"])
print(f'Solids: {Cp_sol_eqv}')



# Finally, let's print the new parameter dictionaries:
print('Melt endmember parameter dictionaries:')
for m in [fe_melt, mg_melt]:
    print(m.params)

print('Solid endmember parameter dictionaries:')
print(mpv.params)
print(fpv.params)
print(per.params)
print(wus.params)
