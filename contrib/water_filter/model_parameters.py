import numpy as np

R = 8.31446  # gas constant

liq_sp = {'H_poly': -94000.,
          'S_poly': -75.,
          'r': 4.}  # 4 oxygens per Mg2SiO4

# See calculate_KD_energies.py for derivation of feESV params
olfeESV = np.array([1.94536022e+04, 0.00000000e+00, 3.00000000e-07])
wadfeESV = np.array([ 1.11505465e+04, -2.80395144e+00,  8.96713987e-08])
ringfeESV = np.array([ 3.18360572e+03, -5.71706066e-01,  3.09190317e-07])
lmfeESV = np.array([ 2.38453875e+03, -5.59492003e+00,  4.17619638e-08])

ol = {'E': -16982.,
      'S': 4.6,
      'V': 1.89e-6,
      'hyE': 74000.,
      'hyS': 15.,
      'hyV': -2.e-6,
      'feE': olfeESV[0],
      'feS': olfeESV[1],
      'feV': olfeESV[2]}

wad = {'E': 0.,
       'S': 0.,
       'V': 0.,
       'hyE': 34500.,
       'hyS': 15.,
       'hyV': 0.,
       'feE': wadfeESV[0],
       'feS': wadfeESV[1],
       'feV': wadfeESV[2]}

ring = {'E': 7060.,
        'S': -4.8,
        'V': -8.12e-7,
        'hyE': 32500.,
        'hyS': 15.,
        'hyV': 0.,
        'feE': ringfeESV[0],
        'feS': ringfeESV[1],
        'feV': ringfeESV[2]}

lm = {'E': 77858.,
      'S': -0.645,
      'V': -3.53e-6,
      'hyE': 100000.,  # NEED TO CHANGE
      'hyS': -12,  # NEED TO CHANGE
      'hyV': 0.,  # NEED TO CHANGE
      'feE': lmfeESV[0],
      'feS': lmfeESV[1],
      'feV': lmfeESV[2]}

melt = {'E': 175553.,
        'S': 100.3,
        'V': -3.277e-6,
        'a': 2.60339693e-06,
        'b': 2.64753089e-11,
        'c': 1.18703511e+00}


olwad = {'Delta_E': ol['E'] - wad['E'],
         'Delta_S': ol['S'] - wad['S'],
         'Delta_V': ol['V'] - wad['V'],
         'halfPint0': 0.976e9,  # estimated from SLB2011 Figure
         'dhalfPintdT': -4.05e5}  # estimated from SLB2011 Figure

wadring = {'Delta_E': wad['E'] - ring['E'],
           'Delta_S': wad['S'] - ring['S'],
           'Delta_V': wad['V'] - ring['V'],
           'halfPint0': 2.286e9,   # estimated from SLB2011 Figure
           'dhalfPintdT': -8.85e5}  # estimated from SLB2011 Figure

ringlm = {'Delta_E': ring['E'] - lm['E'],
          'Delta_S': ring['S'] - lm['S'],
          'Delta_V': ring['V'] - lm['V'],
          'halfPint0': 0.1e9,  # Something small, could increase this to simulate garnet decomposition.
          'dhalfPintdT': 0.}  # 0.

# Component equation of state parameters
Mg2SiO4_params = {'name': 'Mg2SiO4 (based on Mg-wad)',
                  'equation_of_state': 'mod_hp_tmt',
                  'T_0': 298.15,
                  'Pref': 17500000000.0,
                  'V_0': 4.073795492729634e-05,
                  'K_0': 157218159278.60214,
                  'Kprime_0': 4.747568746830502,
                  'Kdprime_0': -4.235205844957051e-11,
                  'a_0': 1.0724967177696426e-05,
                  'T_einstein': 1142.5572613375452,
                  'H_Pref': -1324992.8391624494,
                  'S_Pref': 81.1060010156774,
                  'Cp_Pref': np.array([1.49312734e+02, 9.66981406e-03,
                                       -1.06823762e+07, 7.20496606e+02]),
                  'molar_mass': 0.14069310000000002,
                  'n': 7.0,
                  'formula': {'Mg': 2.0, 'Si': 1.0, 'O': 4.0}}

Fe2SiO4_params = {'name': 'Fe2SiO4 (based on Fe-wad)',
                  'equation_of_state': 'mod_hp_tmt',
                  'T_0': 298.15,
                  'Pref': 17500000000.0,
                  'V_0': 4.302047981460708e-05,
                  'K_0': 157940753600.1616,
                  'Kprime_0': 4.716105207436653,
                  'Kdprime_0': -4.062575119500919e-11,
                  'a_0': 1.221163276444518e-05,
                  'T_einstein': 1029.050289141727,
                  'H_Pref': -616069.0953829046,
                  'S_Pref': 133.85124728860626,
                  'Cp_Pref': np.array([1.51964362e+02, 8.80493315e-03,
                                       -7.74825617e+06, 6.51972863e+02]),
                  'molar_mass': 0.20377309999999998,
                  'n': 7.0,
                  'formula': {'Fe': 2.0, 'Si': 1.0, 'O': 4.0}}

MgSiO3_params = {'name': 'MgSiO3 (based on majorite)',
                 'equation_of_state': 'mod_hp_tmt',
                 'T_0': 298.15,
                 'Pref': 17500000000.0,
                 'V_0': 2.8648145222703425e-05,
                 'K_0': 160125971765.10178,
                 'Kprime_0': 4.38070845314276,
                 'Kdprime_0': -2.834813574330987e-11,
                 'a_0': 1.3063847563146157e-05,
                 'T_einstein': 889.9701738863345,
                 'H_Pref': -927677.8250983527,
                 'S_Pref': 57.849598336365794,
                 'Cp_Pref': np.array([1.17851662e+02, 3.67545477e-03,
                                      -5.58729690e+06, 1.86118367e+02]),
                 'molar_mass': 0.1003887,
                 'n': 5.0,
                 'formula': {'O': 3.0, 'Mg': 1.0, 'Si': 1.0}}


H2O_params = {'name': 'H2O (based on water)',
              'equation_of_state': 'mod_hp_tmt',
              'T_0': 298.15,
              'Pref': 17500000000.0,
              'V_0': 1.3230849810056659e-05,
              'K_0': 23855375995.545673,
              'Kprime_0': 3.1271291860824606,
              'Kdprime_0': -5.950260351896275e-11,
              'a_0': 0.00022967262033577235,
              'T_einstein': 0.22907580091146804,
              'H_Pref': -63479.066549524505,
              'S_Pref': -28.056180880343014,
              'Cp_Pref': np.array([7.31268765e+01,  1.14832240e-03,
                                   1.91210537e+07, -1.02859341e+03]),
              'molar_mass': 0.01801528, 'n': 3.0,
              'formula': {'H': 2.0, 'O': 1.0}}
