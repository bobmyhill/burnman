from scipy.optimize import fsolve
from eos_and_averaging import thermodynamic_properties
from model_parameters import *

def partitioning(args, composition, redox_on=1.):
    xCa, xFe, xMg, xAl, xSi, xO = composition

    #for i, m in enumerate(args):
    #        if m < 0.:
    #        args[i] = -args[i]

    CaSiO3, MgO, FeO, MgSiO3, FeSiO3, FeAlO3, AlAlO3, iron = args

    KDapp = 0.4

    bdg_total = MgSiO3 + FeSiO3 + FeAlO3 + AlAlO3
    Al_bdg = (FeAlO3 + 2.*AlAlO3)
    Fe_bdg = (FeSiO3 + FeAlO3)

    Al_bdg_apfu = Al_bdg/bdg_total
    Fe_bdg_apfu = Fe_bdg/bdg_total
    Nakajima_expr = FeAlO3/bdg_total - redox_on*(-0.13*Al_bdg_apfu + 3.1*(Al_bdg_apfu**2) + 0.39*Fe_bdg_apfu) # note typo in Frost and Myhill (2016)


    eqns = np.array([CaSiO3 - xCa,
                     FeO + FeSiO3 + FeAlO3 + iron - xFe,
                     MgO + MgSiO3 - xMg,
                     FeAlO3 + AlAlO3*2. - xAl,
                     CaSiO3 + MgSiO3 + FeSiO3 - xSi,
                     3.*(CaSiO3 + bdg_total) + (FeO + MgO) - xO,
                     (Fe_bdg*MgO)/(MgSiO3*FeO) - KDapp,
                     Nakajima_expr])

    return eqns


def CFMASO_mineralogy(composition=np.array([1.3, 2.4, 20.0, 1.9, 16.0, 58.4]), redox_on=1.):


    mins = ['CaSiO3', 'MgO', 'FeO', 'MgSiO3', 'FeSiO3', 'FeAlO3', 'AlAlO3', 'iron']
    c = composition
    guess =[1.3, 7.00695968, 0.5, 15.30647257, 0.5, 0.41597479, 0.7420126,   0. ]
    sol = fsolve(partitioning, guess, args = (composition, redox_on))
    CaSiO3, MgO, FeO, MgSiO3, FeSiO3, FeAlO3, AlAlO3, iron = sol

    d = {}
    d['x_cpv'] = CaSiO3
    d['x_iron'] = iron
    d['x_fper'] = MgO+FeO
    d['x_bdg'] = MgSiO3 + FeSiO3 + FeAlO3 + AlAlO3
    d['x_total'] = (d['x_cpv'] + d['x_iron'] + d['x_fper'] + d['x_bdg'])
    d['f_cpv'] = d['x_cpv'] / d['x_total']
    d['f_iron'] = d['x_iron'] / d['x_total']
    d['f_fper'] = d['x_fper']  / d['x_total']
    d['f_bdg'] = d['x_bdg'] / d['x_total']

    d['bdg_fractions'] = {'mbdg': MgSiO3/d['x_bdg'],
                          'fbdg': FeSiO3/d['x_bdg'],
                          'fabdg': FeAlO3/d['x_bdg'],
                          'abdg': AlAlO3/d['x_bdg']}
    d['fper_fractions'] = {'per': MgO/d['x_fper'],
                           'wus': FeO/d['x_fper']}

    return d

def density(pressure, temperature, prps, mins):
    volume = 0.
    mass = 0.
    for i in range(len(mins)):
        V_m = thermodynamic_properties(pressure, temperature, mins[i])['V']
        M_m = mins[i]['molar_mass']

        volume += V_m*prps[i]
        mass += M_m*prps[i]

    return mass/volume

def aluminous_density_difference(P, T, X_Fe_endmember, K_D_fper_melt = 0.24):
    FMS = np.array([c_mantle[1]['FeO']*X_Fe_endmember,
                    c_mantle[0]['MgO']*(1.-X_Fe_endmember),
                    c_mantle[1]['SiO2']*X_Fe_endmember + c_mantle[0]['SiO2']*(1-X_Fe_endmember)])

    FMS *= 2.4+20.0+16.0
    Ca = 1.3
    Al = 1.9
    O = Ca + Al*3./2. + FMS[0] + FMS[1] + FMS[2]*2.
    CFMASO_composition = np.array([Ca, FMS[0], FMS[1], 1.9, FMS[2], O])

    a = CFMASO_mineralogy(CFMASO_composition, redox_on=1.) # turn off redox

    # Aluminous pyrolite (without cpv, iron)
    sol_prps = [a['x_bdg']*a['bdg_fractions']['mbdg'],
                a['x_bdg']*a['bdg_fractions']['fbdg'],
                a['x_bdg']*a['bdg_fractions']['fabdg'],
                a['x_bdg']*a['bdg_fractions']['abdg'],
                a['x_fper']*a['fper_fractions']['per'],
                a['x_fper']*a['fper_fractions']['wus']]
    sol_mins = [mpv_params, fpv_params, fapv_params, apv_params, per_params, wus_params]

    Fe_Mg_melt_ratio = a['fper_fractions']['wus']/a['fper_fractions']['per']/K_D_fper_melt

    # Fe+Mg+Si = 1
    # Fe = c_mantle[1]['FeO']*x_Fe_member
    # Mg = c_mantle[0]['MgO']*(1 - x_Fe_member)

    # Thus
    # Fe/Mg = c_mantle[1]['FeO']*x_Fe_member/(c_mantle[0]['MgO']*(1 - x_Fe_member))
    # Rearranging:
    # 1/(1 + 1/(Fe_Mg_melt_ratio*(c_mantle[0]['MgO']/c_mantle[1]['FeO']))) = x_Fe_member
    x_Fe_member = 1./(1. + 1./(Fe_Mg_melt_ratio*(c_mantle[0]['MgO']/c_mantle[1]['FeO'])))

    c_melt = np.array([x_Fe_member*c_mantle[1]['FeO'],
                       (1 - x_Fe_member)*c_mantle[0]['MgO'],
                       x_Fe_member*c_mantle[1]['SiO2'] + (1. - x_Fe_member)*c_mantle[0]['SiO2']])

    assert Fe_Mg_melt_ratio == c_melt[0]/c_melt[1]

    print(FMS[0]/FMS[1]/Fe_Mg_melt_ratio, c_melt)


    print(c_mantle[1]['FeO']*X_Fe_endmember/c_melt[0])
    #print(CFMASO_composition[1]/np.sum(CFMASO_composition[:5])/c_melt[0])
    # Melt in eqm with aluminous pyrolite (without Ca, Al, Fe3+)
    liq_prps = [1. - x_Fe_member, x_Fe_member]
    liq_mins = [mg_mantle_melt_params, fe_mantle_melt_params]

    rho_sol = density(P, T, sol_prps, sol_mins)
    rho_liq = density(P, T, liq_prps, liq_mins)

    return rho_liq - rho_sol


def melt_solid_density_using_K_D(P, T, X_Fe_endmember, K_D_sol_melt = 0.43):
    FMS = np.array([c_mantle[1]['FeO']*X_Fe_endmember,
                    c_mantle[0]['MgO']*(1.-X_Fe_endmember),
                    c_mantle[1]['SiO2']*X_Fe_endmember + c_mantle[0]['SiO2']*(1-X_Fe_endmember)])

    Fe_Mg_melt_ratio = (FMS[0]/FMS[1])/K_D_sol_melt

    # Fe = c_mantle[1]['FeO']*x_Fe_member
    # Mg = c_mantle[0]['MgO']*(1 - x_Fe_member)

    # Thus
    # Fe/Mg = c_mantle[1]['FeO']*x_Fe_member/(c_mantle[0]['MgO']*(1 - x_Fe_member))
    # Rearranging:
    # 1/(1 + 1/(Fe_Mg_melt_ratio*(c_mantle[0]['MgO']/c_mantle[1]['FeO']))) = x_Fe_member
    x_Fe_member = 1./(1. + 1./(Fe_Mg_melt_ratio*(c_mantle[0]['MgO']/c_mantle[1]['FeO'])))

    c_melt = np.array([x_Fe_member*c_mantle[1]['FeO'],
                       (1 - x_Fe_member)*c_mantle[0]['MgO'],
                       x_Fe_member*c_mantle[1]['SiO2'] + (1. - x_Fe_member)*c_mantle[0]['SiO2']])


    FMS *= 2.4+20.0+16.0
    Ca = 1.3
    Al = 1.9
    O = Ca + Al*3./2. + FMS[0] + FMS[1] + FMS[2]*2.
    CFMASO_composition = np.array([Ca, FMS[0], FMS[1], 1.9, FMS[2], O])

    a = CFMASO_mineralogy(CFMASO_composition)

    # Aluminous pyrolite (without cpv, iron)
    sol_prps = [a['x_bdg']*a['bdg_fractions']['mbdg'],
                a['x_bdg']*a['bdg_fractions']['fbdg'],
                a['x_bdg']*a['bdg_fractions']['fabdg'],
                a['x_bdg']*a['bdg_fractions']['abdg'],
                a['x_fper']*a['fper_fractions']['per'],
                a['x_fper']*a['fper_fractions']['wus']]
    sol_mins = [mpv_params, fpv_params, fapv_params, apv_params, per_params, wus_params]

    #print(CFMASO_composition[1]/np.sum(CFMASO_composition[:5])/c_melt[0])
    # Melt in eqm with aluminous pyrolite (without Ca, Al, Fe3+)
    liq_prps = [1. - x_Fe_member, x_Fe_member]
    liq_mins = [mg_mantle_melt_params, fe_mantle_melt_params]

    rho_sol = density(P, T, sol_prps, sol_mins)
    rho_liq = density(P, T, liq_prps, liq_mins)

    return rho_liq, rho_sol
