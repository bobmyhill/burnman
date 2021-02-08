from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter
from warnings import catch_warnings

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../../'))

import burnman
from burnman import Mineral
from burnman.processchemistry import dictionarize_formula, formula_mass

from eos_and_averaging import thermodynamic_properties

gas_constant = 8.31446

mpv = burnman.Mineral(params={'name': 'mpv',
                              'a_0': 1.87e-05,
                              'K_0': 251.0e9,
                              'Pref': 100.0e9,
                              'Cp_Pref': np.array([ 1.61546581e+02, -3.31714290e-03, -3.57533814e+06, -1.11254791e+03]),
                              'H_Pref': -1442310.0,
                              'Kprime_0': 4.14,
                              'T_0': 298.15,
                              'T_einstein': 561.0,
                              'Kdprime_0': -1.6e-11,
                              'V_0': 2.445e-05,
                              'molar_mass': 0.1003887,
                              'S_Pref': 62.6,
                              'equation_of_state': 'mod_hp_tmt',
                              'n': 5.0,
                              'formula': {'Mg': 1.0, 'Si': 1.0, 'O': 3.0}})

# Check for consistency with burnman code
#mpv.set_state(60.e9, 3000.)
#print(mpv.gibbs, thermodynamic_properties(60.e9, 3000., mpv.params)['gibbs'])
#print(mpv.S, thermodynamic_properties(60.e9, 3000., mpv.params)['S'])
#exit()

fpv = burnman.Mineral(params={'a_0': 1.87e-05,
                              'K_0': 281.0e9,
                              'Pref': 100.0e9,
                              'Cp_Pref': np.array([ 1.39546209e+02,  6.36191292e-03, -4.13886524e+06, -4.64775577e+02]),
                              'H_Pref': -1082910.0,
                              'Kprime_0': 4.14,
                              'T_0': 298.15,
                              'T_einstein': 418.1,
                              'Kdprime_0': -1.6e-11,
                              'V_0': 2.534e-05,
                              'name': 'fpv',
                              'molar_mass': 0.1319287,
                              'S_Pref': 95.0,
                              'equation_of_state': 'mod_hp_tmt',
                              'n': 5.0,
                              'formula': {'Si': 1.0, 'Fe': 1.0, 'O': 3.0}})


per = burnman.Mineral(params={'a_0': 3.11e-05,
                              'K_0': 161.6e9,
                              'Pref': 100.0e9,
                              'Cp_Pref': np.array([ 7.31147154e+01, -6.35318887e-03, -7.33679285e+05, -5.92994207e+02]),
                              'H_Pref': -601570.0,
                              'Kprime_0': 3.95,
                              'T_0': 298.15,
                              'T_einstein': 540.2,
                              'Kdprime_0': -2.4e-11,
                              'V_0': 1.125e-05,
                              'name': 'per',
                              'molar_mass': 0.0403044,
                              'S_Pref': 26.5,
                              'equation_of_state': 'mod_hp_tmt',
                              'n': 2.0,
                              'formula': {'Mg': 1.0, 'O': 1.0}})


wus = burnman.Mineral(params={'a_0': 3.22e-05,
                              'K_0': 152.0e9,
                              'Pref': 100.0e9,
                              'Cp_Pref': np.array([ 5.20016403e+01, 3.36163516e-03, -1.19540964e+06,  2.55067110e+01]),
                              'H_Pref': -262240.0,
                              'Kprime_0': 4.9,
                              'T_0': 298.15,
                              'T_einstein': 297.6,
                              'Kdprime_0': -3.2e-11,
                              'V_0': 1.206e-05,
                              'name': 'fper',
                              'molar_mass': 0.0718444,
                              'S_Pref': 58.6,
                              'equation_of_state': 'mod_hp_tmt',
                              'n': 2.0,
                              'formula': {'Fe': 1.0, 'O': 1.0}})

fe_mantle_melt = burnman.Mineral(params={'a_0': 2.9614332469401705e-05,
                                         'K_0': 166652774642.11273,
                                         'Pref': 100.e9,
                                         'Cp_Pref': np.array([ 7.95326013e+01, -2.41909947e-03, -1.61692272e+06, -5.62222634e+02]),
                                         'H_Pref': -195245.49100022088,
                                         'Kprime_0': 5.0802472229003905,
                                         'T_0': 298.15,
                                         'T_einstein': 505.75,
                                         'Kdprime_0': -3.9742163085937504e-11,
                                         'V_0': 1.2325484447664221e-05,
                                         'name': 'fe mantle',
                                         'molar_mass': 0.0707624708,
                                         'S_Pref': 95.0299295525918,
                                         'equation_of_state': 'mod_hp_tmt',
                                         'n': 2.092,
                                         'formula': {'O': 1.092, 'Fe': 0.908, 'Si': 0.092}})

mg_mantle_melt = burnman.Mineral(params={'a_0': 2.0572748142847914e-05,
                                         'K_0': 231645314972.72287,
                                         'Pref': 100.e9,
                                         'Cp_Pref': np.array([ 7.95326013e+01, -2.41909947e-03, -1.61692272e+06, -5.62222634e+02]),
                                         'H_Pref': -538009.8593335259,
                                         'Kprime_0': 4.252832366943359,
                                         'T_0': 298.15,
                                         'T_einstein': 558.0924045503805,
                                         'Kdprime_0': -2.1381292724609374e-11,
                                         'V_0': 1.2180438865657191e-05,
                                         'name': 'mg mantle',
                                         'molar_mass': 0.048592178,
                                         'S_Pref': 64.88469713598576,
                                         'equation_of_state': 'mod_hp_tmt',
                                         'n': 2.419,
                                         'formula': Counter({'O': 1.419, 'Mg': 0.581, 'Si': 0.419})})



def liq_sol_molar_compositions(P, T,
                               melting_reference_pressure,
                               melting_temperatures,
                               melting_entropies,
                               melting_volumes,
                               n_mole_mix):

    dG = ((melting_temperatures - T) * melting_entropies +
          (P - melting_reference_pressure) * melting_volumes)

    Xls = 1.0 - ((1.0 - np.exp(dG[1]/(n_mole_mix[1]*gas_constant*T))) /
                 (np.exp(dG[0]/(n_mole_mix[0]*gas_constant*T)) -
                  np.exp(dG[1]/(n_mole_mix[1]*gas_constant*T))))

    Xss = Xls * np.exp(dG[1]/(n_mole_mix[1]*gas_constant*T))

    return Xls, Xss

def calculate_xfe_in_fper_and_pv(pressure, temperature,
                                 x_Fe_in_solid, c_mantle):
    """
    Calculates the proportion of wus in fper, fpv in pv, and
    the MASS fraction of pv in the solid given
    the Fe number in the solid.
    """
    x_MgO = (1. - x_Fe_in_solid)*c_mantle[0]['MgO']
    x_FeO = x_Fe_in_solid*c_mantle[1]['FeO']
    x_SiO2 = (1. - x_Fe_in_solid)*c_mantle[0]['SiO2'] + x_Fe_in_solid*c_mantle[1]['SiO2']

    norm = x_MgO + x_FeO
    f_FeO = x_FeO/norm
    f_SiO2 = x_SiO2/norm

    for m in [mpv, fpv, per, wus]:
        m.set_state(pressure, temperature)

    gibbs_rxn = mpv.gibbs + wus.gibbs - fpv.gibbs - per.gibbs
    KD = np.exp(gibbs_rxn/(gas_constant*temperature))

    # Solving equation 6 in Nakajima et al., 2012 for X_Fe_fp and X_Fe_pv
    # Solved using the definition of the distribution coefficient to define X_Fe_fp as a function of X_Fe_pv

    num_to_sqrt = ((-4. * f_FeO * (KD - 1.) * KD * f_SiO2) +
                   (np.power(1. + (f_FeO * (KD - 1)) + ((KD - 1.) * f_SiO2), 2.)))

    p_fpv = ((-1. + f_FeO - (f_FeO * KD) + f_SiO2 - (f_SiO2 * KD) + np.sqrt(num_to_sqrt)) /
               (2. * f_SiO2 * (1. - KD)))

    p_wus = p_fpv / (((1. - p_fpv) * KD) + p_fpv)

    f_pv = f_SiO2 # MOLAR mass of bridgmanite in the solid

    # finally, convert to mass fraction of bridgmanite in the solid
    molar_mass_fper = p_wus*wus.molar_mass + (1. - p_wus)*per.molar_mass
    molar_mass_pv = p_fpv*fpv.molar_mass + (1. - p_fpv)*mpv.molar_mass

    mass_fraction_pv = f_pv*molar_mass_pv/(f_pv*molar_mass_pv + (1. - f_pv)*molar_mass_fper)

    return p_wus, p_fpv, mass_fraction_pv # f_pv is the molar fraction of pv


def calculate_xfe_in_solid_from_molar_proportions(p_wus, p_fpv, mass_fraction_pv):
    """
    Calculates the Fe number in the solid given
    the proportion of wus in fper, fpv in pv, and
    the molar fraction of pv in the solid.
    """

    molar_mass_fper = p_wus*wus.molar_mass + (1. - p_wus)*per.molar_mass
    molar_mass_pv = p_fpv*fpv.molar_mass + (1. - p_fpv)*mpv.molar_mass

    f_pv = ((mass_fraction_pv/molar_mass_pv)/
            ((mass_fraction_pv/molar_mass_pv) + (1. - mass_fraction_pv)/molar_mass_fper)) # molar fraction of bridgmanite

    return f_pv*p_fpv + (1. - f_pv)*p_wus

################# BEGIN PARAMETERS ###################
# All the lists are Mg endmember first, then Fe endmember
c_mantle = [{'MgO': 0.581, 'SiO2': 0.419},
            {'FeO': 0.908, 'SiO2': 0.092}] # molar_proportions

melting_reference_pressure = 120.e9 # Pa
melting_temperatures=np.array([4821.2, 3470.0]) # K
melting_entropies=np.array([34.33, 33.77]) # J/K/mol-cations
melting_volumes=np.array([9.29e-08, 1.51e-07]) # m^3/mol-cations
n_mole_mix = np.array([0.52, 0.56]) # having different "n"s would be incorrect for a solid solution, but this does a slightly better job than assuming the same number of moles mixing for each "endmember"... this model is not strictly thermodynamically correct anyway.


################# END PARAMETERS ###################

x_Fes = np.linspace(1.e-12, 1.-1.e-12, 101)
p_wus = np.empty_like(x_Fes)
p_fpv = np.empty_like(x_Fes)
mass_f_pv = np.empty_like(x_Fes)

pressure = 100.e9
temperature = 3000.
for i, x_Fe_in_solid in enumerate(x_Fes):

    p_wus[i], p_fpv[i], mass_f_pv[i] = calculate_xfe_in_fper_and_pv(pressure, temperature,
                                                                    x_Fe_in_solid, c_mantle)

plt.plot(x_Fes, p_wus, label='p_wus')
plt.plot(x_Fes, p_fpv, label='p_fpv')
plt.plot(x_Fes, mass_f_pv, label='mass fraction pv', linestyle=':')
plt.legend()
plt.show()



P = 120.e9
temperatures = np.linspace(melting_temperatures[0],
                           melting_temperatures[1], 101)

Xls = np.empty_like(temperatures)
Xss = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    Xls[i], Xss[i] = liq_sol_molar_compositions(P, T,
                                                melting_reference_pressure,
                                                melting_temperatures,
                                                melting_entropies,
                                                melting_volumes,
                                                n_mole_mix)


fig = plt.figure()
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
ax[0].plot(Xls, temperatures, label='liquid composition')
ax[0].plot(Xss, temperatures, label='solid composition')
ax[0].legend()

xs = np.linspace(0., 1., 101)
ax[1].plot(xs, melting_entropies.dot(np.array([1 - xs, xs])))
ax[2].plot(xs, melting_volumes.dot(np.array([1 - xs, xs]))*1.e6)

plt.show()






# Now for the fitting
# First, the Fe endmember
print(c_mantle)
x_FeO = c_mantle[1]['FeO']
x_SiO2 = c_mantle[1]['SiO2']

n_fpv = x_SiO2
n_wus = (x_FeO - x_SiO2)

with catch_warnings(record=True) as w:
    burnman.tools.check_eos_consistency(fpv, P=80.e9, T=4000., verbose=True)


temperatures = np.linspace(300., 7000., 101)
pressures = 100.e9 + 0.*temperatures

fe_mantle = burnman.CombinedMineral([fpv, wus], [n_fpv, n_wus])
heat_capacities = fe_mantle.evaluate(['C_p'], pressures, temperatures)[0]
plt.plot(temperatures, heat_capacities, linestyle='-', color='red', label='Fe-endmember')



# Now the Mg endmember
x_MgO = c_mantle[0]['MgO']
x_SiO2 = c_mantle[0]['SiO2']

n_mpv = x_SiO2
n_per = (x_MgO - x_SiO2)

mg_mantle = burnman.CombinedMineral([mpv, per], [n_mpv, n_per])
heat_capacities = mg_mantle.evaluate(['C_p'], pressures, temperatures)[0]
plt.plot(temperatures, heat_capacities, linestyle='-', color='blue', label='Mg-endmember')

plt.ylim(0., )

plt.xlabel('Temperature (K)')
plt.ylabel('Cp (J/K/mol)')
plt.legend()
plt.show()
