from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
from eos_and_averaging import thermodynamic_properties
from model_parameters import *

def liq_sol_molar_compositions(P, T,
                               melting_reference_pressure,
                               melting_temperatures,
                               melting_entropies,
                               melting_volumes,
                               n_mole_mix):

    dG = ((melting_temperatures - T) * melting_entropies +
          (P - melting_reference_pressure) * melting_volumes) # this is a linearised version (we don't use the melt and solid endmember equations of state directly).

    Xls = 1.0 - ((1.0 - np.exp(dG[1]/(n_mole_mix[1]*gas_constant*T))) /
                 (np.exp(dG[0]/(n_mole_mix[0]*gas_constant*T)) -
                  np.exp(dG[1]/(n_mole_mix[1]*gas_constant*T))))

    Xss = Xls * np.exp(dG[1]/(n_mole_mix[1]*gas_constant*T))

    return Xls, Xss

from scipy.optimize import fsolve
def solidus_liquidus(P, X_Fe):


    diff_Xl = lambda T: liq_sol_molar_compositions(P, T,
                                                   melting_reference_pressure,
                                                   melting_temperatures,
                                                   melting_entropies,
                                                   melting_volumes,
                                                   n_mole_mix)[0] - X_Fe
    diff_Xs = lambda T: liq_sol_molar_compositions(P, T,
                                                   melting_reference_pressure,
                                                   melting_temperatures,
                                                   melting_entropies,
                                                   melting_volumes,
                                                   n_mole_mix)[1] - X_Fe
    return fsolve(diff_Xs, 4000.)[0], fsolve(diff_Xl, 4000.)[0]


def melting_enthalpy(P, T, X_bulk,
                     melting_reference_pressure,
                     melting_temperatures,
                     melting_entropies,
                     melting_volumes,
                     n_mole_mix):

    # arrays are Mg-rich endmember first
    dG = ((melting_temperatures - T) * melting_entropies +
          (P - melting_reference_pressure) * melting_volumes) # this is a linearised version (we don't use the melt and solid endmember equations of state directly).

    Xls = 1.0 - ((1.0 - np.exp(dG[1]/(n_mole_mix[1]*gas_constant*T))) /
                 (np.exp(dG[0]/(n_mole_mix[0]*gas_constant*T)) -
                  np.exp(dG[1]/(n_mole_mix[1]*gas_constant*T))))

    Xss = Xls * np.exp(dG[1]/(n_mole_mix[1]*gas_constant*T))


    dH = (melting_temperatures * melting_entropies +
          (P - melting_reference_pressure) * melting_volumes)
    if X_bulk < Xss:
        H = 0.
    elif X_bulk > Xls:
        H = np.array([1. - X_bulk, X_bulk]).dot(dH)
    else:
        H = (X_bulk - Xss)/(Xls - Xss)*np.array([1. - Xls, Xls]).dot(dH)
    return H

def calculate_endmember_proportions_volumes_masses(pressure, temperature,
                                                   x_Fe_bearing_endmember, x_Fe_bearing_endmember_in_melt, porosity,
                                                   c_mantle):
    """
    Calculates the endmember molar fractions in the composite,
    along with the solid and melt molar masses and molar volumes

    Note: the equilibrium compositions of the solid phases
    (bridgmanite and periclase) are calculated in this function,
    but the coexisting melt is not necessarily in equilibrium with those solid phases.
    A separate function is required to equilibrate the solid and melt.
    """

    # 1) Molar composition of the solid
    x_MgO_solid = (1. - x_Fe_bearing_endmember)*c_mantle[0]['MgO']
    x_FeO_solid = x_Fe_bearing_endmember*c_mantle[1]['FeO']
    x_SiO2_solid = (1. - x_Fe_bearing_endmember)*c_mantle[0]['SiO2'] + x_Fe_bearing_endmember*c_mantle[1]['SiO2']

    norm = x_MgO_solid + x_FeO_solid
    f_FeO = x_FeO_solid/norm # note that f_FeO ("Fe number, or Fe/(Mg+Fe)") is NOT x_Fe
    f_SiO2 = x_SiO2_solid/norm

    # 2) Fe-Mg partitioning between bridgmanite and periclase in the solid
    mpv_gibbs = thermodynamic_properties(pressure, temperature, mpv_params)['gibbs']
    fpv_gibbs = thermodynamic_properties(pressure, temperature, fpv_params)['gibbs']
    per_gibbs = thermodynamic_properties(pressure, temperature, per_params)['gibbs']
    wus_gibbs = thermodynamic_properties(pressure, temperature, wus_params)['gibbs']

    gibbs_rxn = mpv_gibbs + wus_gibbs - fpv_gibbs - per_gibbs
    KD = np.exp(gibbs_rxn/(gas_constant*temperature))

    # Solving equation 6 in Nakajima et al., 2012 for x_Fe_fp and x_Fe_pv
    # Solved using the definition of the distribution coefficient
    # to define x_Fe_fp as a function of x_Fe_pv
    num_to_sqrt = ((-4. * f_FeO * (KD - 1.) * KD * f_SiO2) +
                   (np.power(1. + (f_FeO * (KD - 1.)) + ((KD - 1.) * f_SiO2), 2.)))
    p_fpv = ((-1. + f_FeO - (f_FeO * KD) + f_SiO2 - (f_SiO2 * KD) + np.sqrt(num_to_sqrt)) /
               (2. * f_SiO2 * (1. - KD)))
    p_wus = p_fpv / (((1. - p_fpv) * KD) + p_fpv)
    f_pv = f_SiO2 # MOLAR fraction of bridgmanite IN THE SOLID (on a formula unit basis)

    # 3) Molar mass of the solution phases
    molar_mass_fper = p_wus*wus_params['molar_mass'] + (1. - p_wus)*per_params['molar_mass']
    molar_mass_pv = p_fpv*fpv_params['molar_mass'] + (1. - p_fpv)*mpv_params['molar_mass']
    molar_mass_melt = (x_Fe_bearing_endmember_in_melt * fe_mantle_melt_params['molar_mass'] +
                       (1. - x_Fe_bearing_endmember_in_melt) * mg_mantle_melt_params['molar_mass'])

    # 5) Molar volume of the solid phases
    mpv_volume = thermodynamic_properties(pressure, temperature, mpv_params)['V']
    fpv_volume = thermodynamic_properties(pressure, temperature, fpv_params)['V']
    per_volume = thermodynamic_properties(pressure, temperature, per_params)['V']
    wus_volume = thermodynamic_properties(pressure, temperature, wus_params)['V']

    molar_volume_fper = p_wus*wus_volume + (1. - p_wus)*per_volume # 1 cation
    molar_volume_pv = p_fpv*fpv_volume + (1. - p_fpv)*mpv_volume # 2 cations

    # 6) Molar volume of the solid on a formula unit basis
    molar_volume_solid = (molar_volume_fper*(1. - f_pv) + molar_volume_pv*f_pv)
    molar_mass_solid = (molar_mass_fper*(1. - f_pv) + molar_mass_pv*f_pv)

    # 7) Molar volume of the liquid on a formula-unit (one-cation) basis
    # We can calculate this in two ways, using the full EOS for the liquid,
    # or the two-parameter volume of melting (from which the full EOS is built).
    # Here I use the full EoS
    mg_melt_volume = thermodynamic_properties(pressure, temperature, mg_mantle_melt_params)['V']
    fe_melt_volume = thermodynamic_properties(pressure, temperature, fe_mantle_melt_params)['V']
    molar_volume_melt = (1. - x_Fe_bearing_endmember_in_melt)*mg_melt_volume + x_Fe_bearing_endmember_in_melt*fe_melt_volume

    # Here I use the two parameter version
    # This is slightly more efficient than querying the melt equations of state

    # for Mg-melt endmember, on a one-cation basis:
    # x_per = x_MgO - x_SiO2
    #       = 1. - 2.*x_SiO2
    # x_mpv = x_SiO2
    #molar_volume_melt = ((1. - x_Fe_bearing_endmember_in_melt)*((1. - 2.*c_mantle[0]['SiO2'])*per_volume +
    #                                          (c_mantle[0]['SiO2']*mpv_volume) +
    #                                          melting_volumes[0]) +
    #                     x_Fe_bearing_endmember_in_melt*((1. - 2.*c_mantle[1]['SiO2'])*wus_volume +
    #                                   c_mantle[1]['SiO2']*fpv_volume +
    #                                   melting_volumes[1]))

    # 8) Use the porosity to calculate the molar fraction of melt and solid
    n_moles_melt = porosity/molar_volume_melt
    n_moles_solid = (1. - porosity)/molar_volume_solid
    molar_fraction_solid = n_moles_solid/(n_moles_melt + n_moles_solid)
    molar_fraction_melt = 1. - molar_fraction_solid

    # 9) Endmember molar fractions in the solid-melt composite (on a formula-unit basis)
    molar_fractions_in_composite = {'per': molar_fraction_solid * (1. - f_pv) * (1. - p_wus),
                                    'wus': molar_fraction_solid * (1. - f_pv) * p_wus,
                                    'mpv': molar_fraction_solid * f_pv * (1. - p_fpv),
                                    'fpv': molar_fraction_solid * f_pv * p_fpv,
                                    'mg_melt': molar_fraction_melt * (1. - x_Fe_bearing_endmember_in_melt),
                                    'fe_melt': molar_fraction_melt * x_Fe_bearing_endmember_in_melt}

    molar_volumes = {'melt': molar_volume_melt,
                     'solid': molar_volume_solid}
    molar_masses = {'melt': molar_mass_melt,
                    'solid': molar_mass_solid}

    return [molar_fractions_in_composite, molar_volumes, molar_masses]


def calculate_xfe_in_solid_from_molar_proportions(endmember_proportions, c_mantle):
    """
    Calculates x_Fe in the solid given
    the endmember proportions in the solid.
    """

    molar_FeO = ((1.*endmember_proportions['fpv'] + endmember_proportions['wus'])/
                 (2.*endmember_proportions['fpv'] + endmember_proportions['wus'] +
                  2.*endmember_proportions['mpv'] + endmember_proportions['per']))

    x_Fe = molar_FeO / c_mantle[1]['FeO']
    return x_Fe


################# BEGIN PLOTS ######################
plt.rc('font', family='DejaVu sans', size=15.)


# 1) Plot phase proportions as a function of the 1D compositional parameter x_Fe
x_Fes = np.linspace(0., 1., 101)
ppns = []

pressure = 100.e9
temperature = 3000.
for i, x_Fe_bearing_endmember in enumerate(x_Fes):
    ppns.append(calculate_endmember_proportions_volumes_masses(pressure, temperature,
                                                               x_Fe_bearing_endmember,
                                                               x_Fe_bearing_endmember_in_melt=0.0, porosity=0.0,
                                                               c_mantle=c_mantle)[0])



fig = plt.figure(figsize=(20, 8))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

for mbr in ['per', 'wus', 'mpv', 'fpv', 'mg_melt', 'fe_melt']:
    ax[0].plot(x_Fes, [ppn[mbr] for ppn in ppns], label='{0} in composite'.format(mbr))

ax[1].plot(x_Fes, [calculate_xfe_in_solid_from_molar_proportions(ppn, c_mantle) for ppn in ppns],
           label='x_Fe in solid (converted from mbr fractions; should be 1:1)',
           linewidth=3.)

ax[1].plot(x_Fes, [((ppn['wus'] + ppn['fpv'])/
                    (ppn['per'] + ppn['wus'] + ppn['mpv'] + ppn['fpv']))
                   for ppn in ppns], label='Fe/(Mg+Fe) in solid')

ax[1].plot(x_Fes, [((ppn['per'] + ppn['mpv'])/
                    (ppn['per'] + ppn['wus'] + 2.*ppn['mpv'] + 2.*ppn['fpv']))
                   for ppn in ppns],
           linestyle=':' , label='MgO in solid')
ax[1].plot(x_Fes, [((ppn['wus'] + ppn['fpv'])/
                    (ppn['per'] + ppn['wus'] + 2.*ppn['mpv'] + 2.*ppn['fpv']))
                   for ppn in ppns],
           linestyle=':' , label='FeO in solid')
ax[1].plot(x_Fes, [((ppn['fpv'] + ppn['mpv'])/
                    (ppn['per'] + ppn['wus'] + 2.*ppn['mpv'] + 2.*ppn['fpv']))
                   for ppn in ppns],
           linestyle=':', label='SiO2 in solid')

for i in range(2):
    ax[i].set_xlabel('molar proportion of Fe bearing endmember')
    ax[i].set_ylabel('Molar fractions')
    ax[i].legend()
plt.show()


# 2) Plot melt curves and melting entropies/volumes as a function of the 1D compositional parameter x_Fe
fig = plt.figure()
fig.set_size_inches(18.0, 12.0)

ax = [fig.add_subplot(2, 2, i) for i in range(1, 4)]
for P in [120.e9, 130.e9, 140.e9]:
    dT = melting_volumes/melting_entropies*(P - melting_reference_pressure) # dT/dP = DV/DS
    T_melt = melting_temperatures + dT
    temperatures = np.linspace(T_melt[0], T_melt[1], 101)


    Xls = np.empty_like(temperatures)
    Xss = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        Xls[i], Xss[i] = liq_sol_molar_compositions(P, T,
                                                    melting_reference_pressure,
                                                    melting_temperatures,
                                                    melting_entropies,
                                                    melting_volumes,
                                                    n_mole_mix)


    ax[0].plot(Xls, temperatures, label='liquid composition, {0} GPa'.format(P/1.e9))
    ax[0].plot(Xss, temperatures, label='solid composition, {0} GPa'.format(P/1.e9))

xs = np.linspace(0., 1., 101)

ax[1].plot(xs, melting_entropies.dot(np.array([1 - xs, xs])), label='melting entropies (linear)')
ax[2].plot(xs, melting_volumes.dot(np.array([1 - xs, xs]))*1.e6, label='melting volumes (linear)')

melt_volumes = np.empty_like(xs)
for j, x_Fe_bearing_endmember in enumerate(xs):
    n_cations_solid = 1./((1. - x_Fe_bearing_endmember)*c_mantle[0]['MgO'] + x_Fe_bearing_endmember*c_mantle[1]['FeO'])
    molar_volumes = calculate_endmember_proportions_volumes_masses(pressure = 100.e9, temperature = 3600.,
                                                                   x_Fe_bearing_endmember=x_Fe_bearing_endmember,
                                                                   x_Fe_bearing_endmember_in_melt=x_Fe_bearing_endmember, # again, the proportion of the Fe-rich composition
                                                                   porosity=0.0,
                                                                   c_mantle=c_mantle)[1]

    melt_volumes[j] = molar_volumes['melt'] - molar_volumes['solid']/n_cations_solid

ax[2].plot(xs, melt_volumes*1.e6, linestyle=':', label='computed melting volumes (not used, but should be close to the line fit)')
ax[2].set_ylim(0.,)

for i in range(3):
    ax[i].set_xlim(0., 1.)
    ax[i].set_xlabel('molar proportion of Fe bearing endmember')
    ax[i].legend()

ax[0].set_ylabel('Temperature (K)')
ax[1].set_ylabel('$\Delta S_{melting}$ (J/K/mol_cations)')
ax[2].set_ylabel('$\Delta V_{melting}$ (cm$^3$/mol_cations)')

plt.show()


##########################################
############### BENCHMARK ################
##########################################
X_Fe = 0.2
P = 120.e9

print('At {0} GPa and X_Fe = {1}:'.format(P/1.e9, X_Fe))
print('deltaH_fusion = {0} J/mol'.format(np.array([1. - X_Fe, X_Fe]).dot((melting_temperatures *
                                                                            melting_entropies +
                                                                            (P - melting_reference_pressure) * melting_volumes))))

molar_mass_melt = calculate_endmember_proportions_volumes_masses(P, 3000.,
                                                                 X_Fe, X_Fe,
                                                                 0., c_mantle)[2]['melt']
print('molar_mass of melt is {0} kg/mol'.format(molar_mass_melt))
print('Thus deltaH_fusion = {0} J/kg'.format(np.array([1. - X_Fe, X_Fe]).dot((melting_temperatures * melting_entropies +
                                                                          (P - melting_reference_pressure) * melting_volumes))/
                                         molar_mass_melt))

solidus, liquidus = solidus_liquidus(P, X_Fe)

print('the solidus and liquidus are at {0} and {1} K'.format(solidus, liquidus))

temperatures = np.linspace(solidus, liquidus, 101)
heat_capacities = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    p, v, m = calculate_endmember_proportions_volumes_masses(P, T,
                                                             X_Fe, X_Fe,  0., # (completely solid)
                                                             c_mantle)

    heat_capacities[i] = (p['mpv']*thermodynamic_properties(P, T, mpv_params)['molar_C_p'] +
                          p['fpv']*thermodynamic_properties(P, T, fpv_params)['molar_C_p'] +
                          p['per']*thermodynamic_properties(P, T, per_params)['molar_C_p'] +
                          p['wus']*thermodynamic_properties(P, T, wus_params)['molar_C_p'])/m['solid']

heat_capacities2 = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    p, v, m = calculate_endmember_proportions_volumes_masses(P, T,
                                                             X_Fe, X_Fe,  1., # (completely liq)
                                                             c_mantle)

    heat_capacities2[i] = (p['mg_melt']*thermodynamic_properties(P, T, mg_mantle_melt_params)['molar_C_p'] +
                          p['fe_melt']*thermodynamic_properties(P, T, fe_mantle_melt_params)['molar_C_p'])/m['melt']


plt.plot(temperatures, heat_capacities)
plt.plot(temperatures, heat_capacities2)
plt.show()
print('')
print('The heat capacity of the solid is ~{0} J/K/kg across the melting interval'.format(heat_capacities[51]))
print('This corresponds to an enthalpy change of {0} J/kg.'.format(np.trapz(heat_capacities, temperatures)))



temperatures = np.linspace(solidus-1., liquidus+1., 1001)
enthalpies = np.empty_like(temperatures)

#for X_Fe in [0.2, 0.4, 0.6, 0.8]:
for i, T in enumerate(temperatures):
    enthalpies[i] = melting_enthalpy(P, T, X_Fe,
                                     melting_reference_pressure,
                                     melting_temperatures,
                                     melting_entropies,
                                     melting_volumes,
                                     n_mole_mix)

plt.plot(temperatures, np.gradient(enthalpies, temperatures))
plt.show()
