from __future__ import absolute_import
import numpy as np

gas_constant = 8.31446

#################### BEGIN EOS PARAMETERS #######################
mpv_params={'name': 'mpv',
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
            'formula': {'Mg': 1.0, 'Si': 1.0, 'O': 3.0}}


fpv_params={'a_0': 1.87e-05,
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
            'formula': {'Si': 1.0, 'Fe': 1.0, 'O': 3.0}}


per_params={'a_0': 3.11e-05,
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
            'formula': {'Mg': 1.0, 'O': 1.0}}


wus_params={'a_0': 3.22e-05,
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
            'formula': {'Fe': 1.0, 'O': 1.0}}

fe_mantle_melt_params={'a_0': 2.9614332469401705e-05,
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
                       'formula': {'O': 1.092, 'Fe': 0.908, 'Si': 0.092}}

mg_mantle_melt_params={'a_0': 2.0572748142847914e-05,
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
                       'formula': {'O': 1.419, 'Mg': 0.581, 'Si': 0.419}}


volatile_mantle_melt_params={'a_0': 2.0572748142847914e-05,
                             'K_0': 231645314972.72287,
                             'Pref': 100.e9,
                             'Cp_Pref': np.array([ 7.95326013e+01, -2.41909947e-03, -1.61692272e+06, -5.62222634e+02]),
                             'H_Pref': -538009.8593335259,
                             'Kprime_0': 4.252832366943359,
                             'T_0': 298.15,
                             'T_einstein': 558.0924045503805,
                             'Kdprime_0': -2.1381292724609374e-11,
                             'V_0': 1.2180438865657191e-05,
                             'name': 'volatile mantle component',
                             'molar_mass': 0.048592178,
                             'S_Pref': 64.88469713598576,
                             'equation_of_state': 'mod_hp_tmt',
                             'n': 3.,
                             'formula': {'O': 2., 'C': 1.}}

print('WARNING! volatile melt parameters not yet chosen!!')
##################### END EOS PARAMETERS ########################

################# BEGIN MELT MODEL PARAMETERS ###################
# All the lists are Mg endmember first, then Fe endmember
c_mantle = [{'MgO': 0.581, 'SiO2': 0.419},
            {'FeO': 0.907, 'SiO2': 0.093}] # molar_proportions

melting_reference_pressure = 120.e9 # Pa
melting_temperatures=np.array([4821.2, 3424.5]) # K 
melting_entropies=np.array([34.33, 33.77]) # J/K/mol-cations
melting_volumes=np.array([9.29e-08, 1.51e-07]) # m^3/mol-cations
n_mole_mix = np.array([0.62, 0.48]) # having different "n"s would be incorrect for a solid solution, but this does a slightly better job than assuming the same number of moles mixing for each "endmember"... this model is not strictly thermodynamically correct anyway.


################# END MELT MODEL PARAMETERS ###################


########### NOTE FOR BULK COMPOSITION OF PYROLITE #############

x = 0.93
c_pyrolite = {'FeO': 0.908*(1. - x), 'MgO': 0.581*x, 'SiO2': 0.419*x + 0.092*(1. - x)}

#print('Fe/Si: {0}'.format(c_pyrolite['FeO']/c_pyrolite['SiO2']))
#print('Mg/Si: {0}'.format(c_pyrolite['MgO']/c_pyrolite['SiO2']))
print('Fe/Mg: {0} ({1})'.format(c_pyrolite['FeO']/c_pyrolite['MgO'], 5.8/50.))
