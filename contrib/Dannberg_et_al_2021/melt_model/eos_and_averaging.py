import numpy as np
from model_parameters import *

eps = np.finfo(np.float).eps
gas_constant = 8.31446

def tait_constants(params):
    """
    returns parameters for the modified Tait equation of state
    derived from K_T and its two first pressure derivatives
    EQ 4 from Holland and Powell, 2011
    """
    a = (1. + params['Kprime_0']) / (
        1. + params['Kprime_0'] + params['K_0'] * params['Kdprime_0'])
    b = params['Kprime_0'] / params['K_0'] - \
        params['Kdprime_0'] / (1. + params['Kprime_0'])
    c = (1. + params['Kprime_0'] + params['K_0'] * params['Kdprime_0']) / (
        params['Kprime_0'] * params['Kprime_0'] + params['Kprime_0'] - params['K_0'] * params['Kdprime_0'])
    return a, b, c

def thermal_energy(T, einstein_T, n):
    """
    calculate the thermal energy of a substance.  Takes the temperature,
    the Einstein temperature, and n, the number of atoms per molecule.
    Returns thermal energy in J/mol
    """
    if T <= eps:
        return 3. * n * gas_constant * einstein_T * 0.5  # zero point energy
    x = einstein_T / T
    E_th = 3. * n * gas_constant * einstein_T * \
        (0.5 + 1. / (np.exp(x) - 1.0))  # include the zero point energy
    return E_th


def molar_heat_capacity_v(T, einstein_T, n):
    """
    Heat capacity at constant volume.  In J/K/mol
    """
    if T <= eps:
        return 0.
    x = einstein_T / T
    C_v = 3.0 * n * gas_constant * \
        (x * x * np.exp(x) / np.power(np.exp(x) - 1.0, 2.0))
    return C_v


def __thermal_pressure(T, params):
    """
    Returns thermal pressure [Pa] as a function of T [K]
    EQ 12 - 1 of Holland and Powell, 2011
    """

    # This is basically the mie-gruneisen equation of state for thermal
    # pressure using an Einstein model for heat capacity.  The additional
    # assumption that they make is that alpha*K/Cv, (or gamma / V) is
    # constant over a wide range of compressions.
    
    # Note that the xi function in HP2011 is just the Einstein heat capacity
    # divided by 3nR. This function is *not* used to calculate the
    # heat capacity - Holland and Powell (2011) prefer the additional
    # freedom provided by their polynomial expression.
    
    E_th = thermal_energy(T, params['T_einstein'], params['n'])
    C_V0 = molar_heat_capacity_v(
        params['T_0'], params['T_einstein'], params['n'])
    P_th = params['a_0'] * params['K_0'] / C_V0 * E_th
    return P_th

def __relative_thermal_pressure(T, params):
    """
    Returns relative thermal pressure [Pa] as a function of T-params['T_0'] [K]
    EQ 12 - 1 of Holland and Powell, 2011
    """
    return __thermal_pressure(T, params) - \
        __thermal_pressure(params['T_0'], params)

def __intCpdT(temperature, params):
    """
    Returns the thermal addition to the standard state enthalpy [J/mol]
    at the reference pressure
    """
    return ((params['Cp_Pref'][0] * temperature +
             0.5 * params['Cp_Pref'][1] * np.power(temperature, 2.) -
             params['Cp_Pref'][2] / temperature +
             2. * params['Cp_Pref'][3] * np.sqrt(temperature)) -
            (params['Cp_Pref'][0] * params['T_0'] +
             0.5 * params['Cp_Pref'][1] * params['T_0'] * params['T_0'] -
             params['Cp_Pref'][2] / params['T_0'] +
             2.0 * params['Cp_Pref'][3] * np.sqrt(params['T_0'])))

def __intCpoverTdT(temperature, params):
    """
    Returns the thermal addition to the standard state entropy [J/K/mol]
    at the reference pressure
    """
    return ((params['Cp_Pref'][0] * np.log(temperature) +
             params['Cp_Pref'][1] * temperature -
             0.5 * params['Cp_Pref'][2] / np.power(temperature, 2.) -
             2.0 * params['Cp_Pref'][3] / np.sqrt(temperature)) -
            (params['Cp_Pref'][0] * np.log(params['T_0']) +
             params['Cp_Pref'][1] * params['T_0'] -
             0.5 * params['Cp_Pref'][2] / (params['T_0'] * params['T_0']) -
             2.0 * params['Cp_Pref'][3] / np.sqrt(params['T_0'])))


    
def thermodynamic_properties(pressure, temperature, params):
    """
    Returns the gibbs free energy, the entropy, the volume and the density as a function of pressure, temperature and the mineral parameters

    The changes between this equation of state and the Holland and Powell thermal equation of state are as follows:
    1) The reference pressure for all of the volumetric parameters is strictly [0 Pa]
    2) The reference pressure for all of the thermal parameters (including H, S) is params['Pref'], given in [Pa]
    """
    
    a, b, c = tait_constants(params)
    Pth = __relative_thermal_pressure(temperature, params)
    
    ksi_over_ksi_0 = molar_heat_capacity_v(temperature, params['T_einstein'], params['n']) / molar_heat_capacity_v(params['T_0'], params['T_einstein'], params['n'])
    
    # Integrate the gibbs free energy along the isobaric path from (Pref, T_ref) to (Pref, T_final)
    G_Pref_Tf = params['H_Pref'] + __intCpdT(temperature, params) - temperature * (params['S_Pref'] + __intCpoverTdT(temperature, params))
    
    # Integrate the gibbs free energy along the isothermal path from (Pref, T_final) to (P_final, T_final)
    if pressure != params['Pref']: # EQ 13
        intVdP = params['V_0'] * ((pressure - params['Pref'])*(1. - a) +
                                  (a * (np.power((1. + b * (params['Pref'] - Pth)), 1. - c) -
                                        np.power((1. + b * (pressure - Pth)), 1. - c)) /
                                   (b * (c - 1.))))

        dintVdpdT = (params['V_0'] * params['a_0'] * params['K_0'] * a * ksi_over_ksi_0) * (
            np.power((1. + b * (pressure - Pth)), 0. - c) - np.power((1. + b * (params['Pref'] - Pth)), 0. - c))
                    
    else:
        intVdP = 0.
        dintVdpdT = 0.

        
    gibbs = G_Pref_Tf + intVdP
    entropy =  params['S_Pref'] + __intCpoverTdT(temperature, params) + dintVdpdT
    volume = params['V_0']*(1 - a * (1. - np.power((1. + b * (pressure - Pth)), -1.0 * c)))
    density = params['molar_mass']/volume

    # NEW STUFF FOR K_T, alpha, Cp
    C_V0 = molar_heat_capacity_v(params['T_0'], params['T_einstein'], params['n'])
    C_V = molar_heat_capacity_v(temperature, params['T_einstein'], params['n'])
    isothermal_bulk_modulus = (params['K_0'] * (1. + b * (pressure - Pth)) *
                               (a + (1. - a) * np.power((1. + b * (pressure - Pth)), c)))
                               
    thermal_expansivity = (params['a_0'] *
                           (C_V / C_V0) *
                           1. / ((1. + b * (pressure - Pth)) *
                                 (a + (1. - a) *
                                  np.power((1 + b * (pressure - Pth)), c))))

    Cp_ref = (params['Cp_Pref'][0] + params['Cp_Pref'][1] * temperature +
              params['Cp_Pref'][2] * np.power(temperature, -2.) +
              params['Cp_Pref'][3] * np.power(temperature, -0.5))
    
    dSdT0 = params['V_0'] * params['K_0'] * np.power((ksi_over_ksi_0 * params['a_0']), 2.0) * \
            (np.power((1. + b * (pressure - Pth)), -1. - c) -
             np.power((1. + b * (params['Pref']-Pth)), -1. - c))
    
    x = params['T_einstein']/temperature
    dSdT = dSdT0 + dintVdpdT * ( 1 - 2./x + 2./(np.exp(x) - 1.) ) * x/temperature
    
    heat_capacity_p = Cp_ref + temperature * dSdT
    return {'gibbs': gibbs, # molar
            'S': entropy, # S here is a molar quantity. Divide through by molar mass to get J/K/kg.
            'V': volume, # molar
            'rho': density,
            'alpha': thermal_expansivity,
            'beta_T': 1./isothermal_bulk_modulus,
            'molar_C_p': heat_capacity_p,
            'C_p_per_kilogram': heat_capacity_p/params['molar_mass']} # C_p here is a molar quantity. Divide through by molar mass to get J/K/kg.



############### PROPERTY AVERAGING ####################
# 4 parameters:
# proportions of fe perovskite, wuestite and fe_endmember melt in their respective phases (p_fpv, p_wus, p_feliq),
# mass fraction perovskite (mass_f_pv)
# porosity (phi)


def average_solid_properties(pressure, temperature, p_fpv, p_wus, mass_f_pv):
    n_mol_pv = mass_f_pv/(p_fpv*fpv_params['molar_mass'] + (1. - p_fpv)*mpv_params['molar_mass'])
    n_mol_fper = (1. - mass_f_pv)/(p_wus*wus_params['molar_mass'] + (1. - p_wus)*per_params['molar_mass'])

    molar_f_pv =  n_mol_pv/(n_mol_pv + n_mol_fper)

    # mpv, fpv, per, wus
    fractions = [molar_f_pv*(1. - p_fpv),
                 molar_f_pv*p_fpv,
                 (1. - molar_f_pv)*(1. - p_wus),
                 (1. - molar_f_pv)*p_wus]
    
    properties = [thermodynamic_properties(pressure, temperature, mpv_params),
                  thermodynamic_properties(pressure, temperature, fpv_params),
                  thermodynamic_properties(pressure, temperature, per_params),
                  thermodynamic_properties(pressure, temperature, wus_params)]

    molar_masses = [mpv_params['molar_mass'],
                    fpv_params['molar_mass'],
                    per_params['molar_mass'],
                    wus_params['molar_mass']]
                    
    alphas = [prp['alpha'] for prp in properties]
    volumes = [prp['V'] for prp in properties]
    beta_Ts = [prp['beta_T'] for prp in properties]
    C_ps = [prp['molar_C_p'] for prp in properties]

    V_molar = np.sum([fractions[i]*volumes[i] for i in range(4)])
    M_molar = np.sum([fractions[i]*molar_masses[i] for i in range(4)])
    return {'molar mass': M_molar,
            'V': V_molar,
            'alpha': 1./V_molar*np.sum([fractions[i]*alphas[i]*volumes[i] for i in range(4)]),
            'rho': M_molar/V_molar,
            'beta_T': 1./V_molar*np.sum([fractions[i]*beta_Ts[i]*volumes[i] for i in range(4)]),
            'molar_C_p': np.sum([fractions[i]*C_ps[i] for i in range(4)]),
            'C_p_per_kilogram': np.sum([fractions[i]*C_ps[i] for i in range(2)])/M_molar}
    
    

def average_melt_properties(pressure, temperature, p_feliq):    
    # mg_mantle_melt, fe_melt_melt
    fractions = [(1. - p_feliq), p_feliq]
    
    properties = [thermodynamic_properties(pressure, temperature, mg_mantle_melt_params),
                  thermodynamic_properties(pressure, temperature, fe_mantle_melt_params)]

    molar_masses = [mg_mantle_melt_params['molar_mass'],
                    fe_mantle_melt_params['molar_mass']]
                    
    alphas = [prp['alpha'] for prp in properties]
    volumes = [prp['V'] for prp in properties]
    beta_Ts = [prp['beta_T'] for prp in properties]
    C_ps = [prp['molar_C_p'] for prp in properties]

    V_molar = np.sum([fractions[i]*volumes[i] for i in range(2)])
    M_molar = np.sum([fractions[i]*molar_masses[i] for i in range(2)])
    return {'molar mass': M_molar,
            'V': V_molar,
            'alpha': 1./V_molar*np.sum([fractions[i]*alphas[i]*volumes[i] for i in range(2)]),
            'rho': M_molar/V_molar,
            'beta_T': 1./V_molar*np.sum([fractions[i]*beta_Ts[i]*volumes[i] for i in range(2)]),
            'molar_C_p': np.sum([fractions[i]*C_ps[i] for i in range(2)]),
            'C_p_per_kilogram': np.sum([fractions[i]*C_ps[i] for i in range(2)])/M_molar}



def average_composite_properties(pressure, temperature, p_fpv, p_wus, p_feliq, mass_f_pv, phi):

    # solid, liquid
    properties = [average_solid_properties(pressure, temperature, p_fpv, p_wus, mass_f_pv),
                  average_melt_properties(pressure, temperature, p_feliq)]

                    
    molar_masses = [prp['molar mass'] for prp in properties]
    alphas = [prp['alpha'] for prp in properties]
    volumes = [prp['V'] for prp in properties]
    beta_Ts = [prp['beta_T'] for prp in properties]
    C_ps = [prp['molar_C_p'] for prp in properties]

    
    n_moles = (1. - phi)/volumes[0] + phi/volumes[1]
    fractions = [(1. - phi)/volumes[0]/n_moles, phi/volumes[1]/n_moles]

    
    V_molar = np.sum([fractions[i]*volumes[i] for i in range(2)])
    M_molar = np.sum([fractions[i]*molar_masses[i] for i in range(2)])
    return {'molar mass': M_molar,
            'V': V_molar,
            'alpha': 1./V_molar*np.sum([fractions[i]*alphas[i]*volumes[i] for i in range(2)]),
            'rho': M_molar/V_molar,
            'beta_T': 1./V_molar*np.sum([fractions[i]*beta_Ts[i]*volumes[i] for i in range(2)]),
            'molar_C_p': np.sum([fractions[i]*C_ps[i] for i in range(2)]),
            'C_p_per_kilogram': np.sum([fractions[i]*C_ps[i] for i in range(2)])/M_molar} 
