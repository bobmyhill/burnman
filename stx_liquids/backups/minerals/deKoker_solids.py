
class stishovite():
    params={'formula': {'Si': 1., 'O': 2.},
            'V_0': 0.1513000000E+02 * 1e-6, # F_cl (1,1) # should be m^3/mol, Vref
            'T_0': 0.3000000000E+04, # F_cl (1,2) # K, Tref
            'E_0': -.2274840214E+04 * 1e3, # F_cl (1,3) # J/mol, Eref
            'S_0': 0.1668222552E+00 * 1e3, # F_cl (1,4) # J/K/mol, Sref
            'n': 2., # called fsn in param file
            'p_fit': [0.2471304763E+05, 0.4793020138E+05],
            'ak_fit': [0.1389501259E+01, 0.1332025550E+01],
            'C_V': 0.7794230433E-01 * 1e3 # cvfit in param file
    }
    # PV parameters (truncated at 3rd order)
    # de Koker p_fit parameters correspond to the following expression for F_cmp: 
    # for i, p in enumerate(params['p_fit']):
    #     intp += p * np.power(f, i+2) / (i+2.)
    # Therefore
    # p[0]/2 = 9 * K_0 * V_0 / 2
    # p[1]/3 = 9 * K_0 * V_0 * a3 / 6
    params['K_0'] = params['p_fit'][0]/(9.*params['V_0']) * 1.e3
    a3 = 2.*params['p_fit'][1]/(9.*params['K_0']*params['V_0']) * 1.e3
    params['K_prime_0'] = 4. + (a3/3.)
    params['K_dprime_0'] = (-143./9. - params['K_prime_0']*(params['K_prime_0'] - 7.))/params['K_0']
    # PVT parameters
    params['gamma_0'] = params['ak_fit'][0]
    params['q_0'] = params['ak_fit'][1]


class perovskite():
    params={'formula': {'Mg': 1., 'Si': 1., 'O': 3.},
            'V_0': 0.2705000000E+02 * 1e-6, # F_cl (1,1) # should be m^3/mol, Vref
            'T_0': 0.3000000000E+04, # F_cl (1,2) # K, Tref
            'E_0': -.3355012782E+04 * 1e3, # F_cl (1,3) # J/mol, Eref
            'S_0': 0.3384574347E+00 * 1e3, # F_cl (1,4) # J/K/mol, Sref
            'n': 2., # called fsn in param file
            'p_fit': [0.4067243956E+05, 0.1177159096E+05],
            'ak_fit': [0.1893754815E+01, 0.1487809730E+01],
            'C_V': 0.1338111589E+00 * 1e3 # cvfit in param file
    }
    params['K_0'] = params['p_fit'][0]/(9.*params['V_0']) * 1.e3
    a3 = 2.*params['p_fit'][1]/(9.*params['K_0']*params['V_0']) * 1.e3
    params['K_prime_0'] = 4. + (a3/3.)
    params['K_dprime_0'] = (-143./9. - params['K_prime_0']*(params['K_prime_0'] - 7.))/params['K_0']
    # PVT parameters
    params['gamma_0'] = params['ak_fit'][0]
    params['q_0'] = params['ak_fit'][1]


class periclase():
    params={'formula': {'Mg': 1., 'O': 1.},
            'V_0': 0.1223000000E+02 * 1e-6, # F_cl (1,1) # should be m^3/mol, Vref
            'T_0': 0.2000000000E+04, # F_cl (1,2) # K, Tref
            'E_0': -.1164949141E+04 * 1e3, # F_cl (1,3) # J/mol, Eref
            'S_0': 0.1198358648E+00 * 1e3, # F_cl (1,4) # J/K/mol, Sref
            'n': 2., # called fsn in param file
            'p_fit': [0.1208938157E+05, 0.1133765229E+05],
            'ak_fit': [0.1412003694E+01, 0.6317609916E+00],
            'C_V': 0.4904715075E-01 * 1e3 # cvfit in param file
    }
    params['K_0'] = params['p_fit'][0]/(9.*params['V_0']) * 1.e3
    a3 = 2.*params['p_fit'][1]/(9.*params['K_0']*params['V_0']) * 1.e3
    params['K_prime_0'] = 4. + (a3/3.)
    params['K_dprime_0'] = (-143./9. - params['K_prime_0']*(params['K_prime_0'] - 7.))/params['K_0']
    # PVT parameters
    params['gamma_0'] = params['ak_fit'][0]
    params['q_0'] = params['ak_fit'][1]
