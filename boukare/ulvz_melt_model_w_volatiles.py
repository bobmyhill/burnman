from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
from eos_and_averaging import thermodynamic_properties
from model_parameters import *
import ternary

def liq_sol_molar_compositions(P, T, x_volatile_in_melt,
                               melting_reference_pressure,
                               melting_temperatures,
                               melting_entropies,
                               melting_volumes,
                               n_mole_mix):
    """
    At a given pressure, temperature and volatile fraction *in the melt*,
    returns the amount of iron-bearing endmember in the solid and the liquid
    """
    dG = ((melting_temperatures - T) * melting_entropies +
          (P - melting_reference_pressure) * melting_volumes) # this is a linearised version (we don't use the melt and solid endmember equations of state directly).

    p0 = np.exp(dG[0]/(n_mole_mix[0]*gas_constant*T))
    p1 = np.exp(dG[1]/(n_mole_mix[1]*gas_constant*T))

    # Xls is the proportion of X_Fe in the liquid
    # dG is the delta gibbs of the Mg and then Fe endmember
    Xls = (1.0 - (1. - x_volatile_in_melt)*p0) / (p1 - p0)

    # Xss is the proportion of X_Fe in the solid
    Xss = Xls * p1

    return Xls, Xss


from scipy.optimize import fsolve
def solidus_liquidus_dry(P, X_Fe):


    diff_Xl = lambda T: liq_sol_molar_compositions(P, T, 0.,
                                                   melting_reference_pressure,
                                                   melting_temperatures,
                                                   melting_entropies,
                                                   melting_volumes,
                                                   n_mole_mix)[0] - X_Fe
    diff_Xs = lambda T: liq_sol_molar_compositions(P, T, 0.,
                                                   melting_reference_pressure,
                                                   melting_temperatures,
                                                   melting_entropies,
                                                   melting_volumes,
                                                   n_mole_mix)[1] - X_Fe
    return fsolve(diff_Xs, 4000.)[0], fsolve(diff_Xl, 4000.)[0]

def solve_quadratic(a, b, c, scale): # scale provides positive or negative
    f = b*b - 4.*a*c
    if np.abs(scale) != 1:
        raise Exception('scale factor must be -1 or 1')
    if f < 0.:
        raise Exception('ummm, we\'re looking for a real root, and this equation doesn\'t seem to have one (value inside sqrt is {0})'.format(b*b - 4.*a*c))

    return (-b + scale*np.sqrt(f))/(2*a)

def liq_sol_molar_compositions_from_bulk(P, T, x_Fe_bulk, x_vol_bulk,
                                         melting_reference_pressure,
                                         melting_temperatures,
                                         melting_entropies,
                                         melting_volumes,
                                         n_mole_mix):
    """
    At a given pressure, temperature and X_V fraction in the melt,
    returns the amount of iron-bearing endmember in the solid and the liquid
    """
    dG = ((melting_temperatures - T) * melting_entropies +
          (P - melting_reference_pressure) * melting_volumes) # this is a linearised version (we don't use the melt and solid endmember equations of state directly).

    p0 = np.exp(dG[0]/(n_mole_mix[0]*gas_constant*T))
    p1 = np.exp(dG[1]/(n_mole_mix[1]*gas_constant*T))

    # Xls is the proportion of X_Fe in the liquid
    Xls = solve_quadratic(a= (p1 - p0)*p1/p0,
                          b= p1*(1. - 1./p0) - x_Fe_bulk*(p1 - p0)/p0 + x_vol_bulk*(1. - p1),
                          c=-x_Fe_bulk*(1. - 1./p0),
                          scale=1)

    # Xss is the proportion of X_Fe in the solid
    Xss = Xls * p1
    if x_vol_bulk == 0.:
        Xlv = 0.
    else:
        Xlv = x_vol_bulk*(Xls - Xss)/(x_Fe_bulk - Xss)

    if Xlv > 0.:
        molar_fraction_liquid = x_vol_bulk/Xlv
    else:
        molar_fraction_liquid = (x_Fe_bulk - Xss)/(Xls - Xss)

    if molar_fraction_liquid >= -1.e-6 and molar_fraction_liquid < 1.e-6:
        Xls, Xss, Xlv, molar_fraction_liquid = [np.nan, X_Fe_bulk,
                                                np.nan, 0.]
    if molar_fraction_liquid < -1.e-6 or molar_fraction_liquid > 1.:
        Xls, Xss, Xlv, molar_fraction_liquid = [X_Fe_bulk, np.nan,
                                                x_vol_bulk, 1.]

    return Xls, Xss, Xlv, molar_fraction_liquid



def melting_enthalpy(P, T, x_Fe_bulk, x_vol_bulk,
                     melting_reference_pressure,
                     melting_temperatures,
                     melting_entropies,
                     melting_volumes,
                     n_mole_mix):


    Xls, Xss, Xlv, molar_fraction_liquid = liq_sol_molar_compositions_from_bulk(P, T, x_Fe_bulk, x_vol_bulk,
                                                                                melting_reference_pressure,
                                                                                melting_temperatures,
                                                                                melting_entropies,
                                                                                melting_volumes,
                                                                                n_mole_mix)

    dH = (melting_temperatures * melting_entropies +
          (P - melting_reference_pressure) * melting_volumes)

    print('Need to check this!')
    if molar_fraction_liquid < 1.e-6:
        H = 0.
    elif molar_fraction_liquid > 1. - 1.e-6:
        H = np.array([1. - x_Fe_bulk - x_vol_bulk, x_Fe_bulk]).dot(dH)
    else:
        H = molar_fraction_liquid*np.array([1. - Xls - Xlv, Xls]).dot(dH)
    return H

def calculate_endmember_proportions_volumes_masses(pressure, temperature,
                                                   x_Fe_bearing_endmember_in_solid,
                                                   x_Fe_bearing_endmember_in_melt,
                                                   x_volatile_endmember_in_melt,
                                                   porosity,
                                                   c_mantle):

    # Calculates the endmember molar fractions in the composite,
    # along with the solid and melt molar masses and molar volumes

    # Note: the equilibrium compositions of the solid phases
    # (bridgmanite and periclase) are calculated in this function,
    # but the coexisting melt is not necessarily in equilibrium with those solid phases.
    # A separate function is required to equilibrate the solid and melt.


    # 1) Molar composition of the solid
    x_MgO_solid = (1. - x_Fe_bearing_endmember_in_solid)*c_mantle[0]['MgO']
    x_FeO_solid = x_Fe_bearing_endmember_in_solid*c_mantle[1]['FeO']
    x_SiO2_solid = (1. - x_Fe_bearing_endmember_in_solid)*c_mantle[0]['SiO2'] + x_Fe_bearing_endmember_in_solid*c_mantle[1]['SiO2']

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
                       x_volatile_endmember_in_melt * volatile_mantle_melt_params['molar_mass'] +
                       (1. - x_Fe_bearing_endmember_in_melt - x_volatile_endmember_in_melt) * mg_mantle_melt_params['molar_mass'])

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
    volatile_melt_volume = thermodynamic_properties(pressure, temperature, volatile_mantle_melt_params)['V']

    molar_volume_melt = ((1. - x_Fe_bearing_endmember_in_melt)*mg_melt_volume +
                         x_Fe_bearing_endmember_in_melt*fe_melt_volume +
                         x_volatile_endmember_in_melt*volatile_melt_volume)

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
                                    'fe_melt': molar_fraction_melt * x_Fe_bearing_endmember_in_melt,
                                    'volatile_melt': molar_fraction_melt * x_volatile_endmember_in_melt}

    molar_volumes = {'melt': molar_volume_melt,
                     'solid': molar_volume_solid}
    molar_masses = {'melt': molar_mass_melt,
                    'solid': molar_mass_solid}

    return [molar_fractions_in_composite, molar_volumes, molar_masses]


def calculate_xfe_in_solid_from_molar_proportions(endmember_proportions, c_mantle):

    # Calculates x_Fe in the solid given
    # the endmember proportions in the solid.


    molar_FeO = ((1.*endmember_proportions['fpv'] + endmember_proportions['wus'])/
                 (2.*endmember_proportions['fpv'] + endmember_proportions['wus'] +
                  2.*endmember_proportions['mpv'] + endmember_proportions['per']))

    x_Fe = molar_FeO / c_mantle[1]['FeO']
    return x_Fe


################# BEGIN PLOTS ######################

# 2a) Run some different bulk compositions through the melting process
P = 136.e9
temperatures = np.linspace(3000., 5000., 51)

fig = plt.figure()
ax = [fig.add_subplot(2, 3, i) for i in range(1, 7)]

tfig, tax = ternary.figure()
fontsize = 10
tax.ax.axis("off")
tax.gridlines(color="grey", linestyle='-', multiple=0.1)
tax.ticks(multiple=0.2,offset=0.02, tick_formats="%.1f")
tax.right_axis_label("Fe-endmember", fontsize=fontsize, offset=0.14)
tax.left_axis_label("Mg-endmember", fontsize=fontsize, offset=0.14)
tax.bottom_axis_label("volatile", fontsize=fontsize, offset=0.14)

FFM_plot = 0.07
for Fe_over_Fe_plus_Mg in np.linspace(0.01, 0.1, 10):
    for X_vol_bulk in np.linspace(0.1, 0., 6):

        X_Fe_bulk = Fe_over_Fe_plus_Mg*(1. - X_vol_bulk)

        Xls = np.empty_like(temperatures)
        Xss = np.empty_like(temperatures)
        Xlv = np.empty_like(temperatures)
        molar_f_liq = np.empty_like(temperatures)

        for i, T in enumerate(temperatures):
            Xls[i], Xss[i], Xlv[i], molar_f_liq[i] = liq_sol_molar_compositions_from_bulk(P, T, X_Fe_bulk, X_vol_bulk,
                                                                                          melting_reference_pressure,
                                                                                          melting_temperatures,
                                                                                          melting_entropies,
                                                                                          melting_volumes,
                                                                                          n_mole_mix)

        if Fe_over_Fe_plus_Mg == FFM_plot:
            ax[0].plot(temperatures, molar_f_liq, label='$x^{{bulk}}_{{volatile}}$: {0:.2f}'.format(X_vol_bulk))
            ax[1].plot(temperatures, Xls/(1.-Xlv), label='$x^{{bulk}}_{{volatile}}$: {0:.2f}'.format(X_vol_bulk))
            ax[2].plot(temperatures, Xss, label='$x^{{bulk}}_{{volatile}}$: {0:.2f}'.format(X_vol_bulk))
            ax[3].plot(temperatures, Xlv, label='$x^{{bulk}}_{{volatile}}$: {0:.2f}'.format(X_vol_bulk))
            ax[4].plot(Xlv, Xls/(1.-Xlv), label='$x^{{bulk}}_{{volatile}}$: {0:.2f}'.format(X_vol_bulk))
            ax[5].plot(Xlv[0], Xls[0]/(1.-Xlv[0]), label='$x^{{bulk}}_{{volatile}}$: {0:.2f}'.format(X_vol_bulk))

            tax.plot(np.array([Xlv, Xls, 1. - Xls - Xlv]).T, linewidth=2.0,
                              label='$x^{{bulk}}_{{volatile}}$: {0:.2f}'.format(X_vol_bulk))

Fe_over_Fe_plus_Mg = FFM_plot

temperatures = np.linspace(3250., 5000., 8)
X_vol_bulks = np.linspace(0.1, 0.005, 101)

for j, T in enumerate(temperatures):

    Xls = np.empty_like(X_vol_bulks)
    Xss = np.empty_like(X_vol_bulks)
    Xlv = np.empty_like(X_vol_bulks)
    molar_f_liq = np.empty_like(X_vol_bulks)

    for i, X_vol_bulk in enumerate(X_vol_bulks):
        X_Fe_bulk = Fe_over_Fe_plus_Mg*(1. - X_vol_bulk)
        Xls[i], Xss[i], Xlv[i], molar_f_liq[i] = liq_sol_molar_compositions_from_bulk(P, T, X_Fe_bulk, X_vol_bulk,
                                                                                      melting_reference_pressure,
                                                                                      melting_temperatures,
                                                                                      melting_entropies,
                                                                                      melting_volumes,
                                                                                      n_mole_mix)

    print(np.array([Xlv, Xls, 1. - Xls - Xlv]).T)
    tax.plot(np.array([Xlv, Xls, 1. - Xls - Xlv]).T, linewidth=2.0, color='black')

tax.legend()
ax[5].legend()
for i in range(3):
    ax[i].set_xlabel('Temperatures (K)')

for i in range(5):
    ax[i].set_ylim(0., 1.)
ax[2].set_ylim(0., 0.1)

ax[0].set_ylabel('Molar fraction of liquid')
ax[1].set_ylabel('Fe/(Fe+Mg) in liquid')
ax[2].set_ylabel('Fe in solid')
ax[3].set_ylabel('Volatile in liquid')

ax[4].set_xlabel('Volatile in liquid')
ax[4].set_ylabel('Fe/(Fe+Mg) in liquid')
fig.tight_layout()

tfig.savefig('ternary_liquid_plots.pdf')
fig.savefig('liquid_plots.pdf')
plt.show()

exit()
################# BEGIN PLOTS ######################
plt.rc('font', family='DejaVu sans', size=15.)

# 1) Plot phase proportions as a function of the 1D compositional parameter x_Fe
x_Fes = np.linspace(0., 1., 101)
ppns = []


pressure = 100.e9
temperature = 3000.
for i, x_Fe_bearing_endmember in enumerate(x_Fes):
    ppns.append(calculate_endmember_proportions_volumes_masses(pressure, temperature,
                                                               x_Fe_bearing_endmember_in_solid=x_Fe_bearing_endmember,
                                                               x_Fe_bearing_endmember_in_melt=0.0, porosity=0.0,
                                                               x_volatile_endmember_in_melt=0.0,
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


# 2a) Run some different bulk compositions through the melting process
P = 136.e9
temperatures = np.linspace(2000., 5500., 1001)

for X_Fe_bulk in np.linspace(0.01, 0.1, 10):
    for X_vol_bulk in np.linspace(0.9, 0., 10):
        Xls = np.empty_like(temperatures)
        Xss = np.empty_like(temperatures)
        Xlv = np.empty_like(temperatures)
        molar_f_liq = np.empty_like(temperatures)

        for i, T in enumerate(temperatures):
            Xls[i], Xss[i], Xlv[i], molar_f_liq[i] = liq_sol_molar_compositions_from_bulk(P, T, X_Fe_bulk, X_vol_bulk,
                                                                                          melting_reference_pressure,
                                                                                          melting_temperatures,
                                                                                          melting_entropies,
                                                                                          melting_volumes,
                                                                                          n_mole_mix)
        if X_Fe_bulk == 0.01:
            plt.plot(temperatures, molar_f_liq, label='$x^{{bulk}}_{{volatile}}$: {0}'.format(X_vol_bulk))
        else:
            plt.plot(temperatures, molar_f_liq)

plt.legend()
plt.xlabel('Temperatures (K)')
plt.ylabel('Molar fraction of liquid')
plt.show()


# 2b) Plot melt curves and melting entropies/volumes as a function of the 1D compositional parameter x_Fe
fig = plt.figure()
fig.set_size_inches(18.0, 12.0)

ax = [fig.add_subplot(2, 2, i) for i in range(1, 4)]
for P in [120.e9, 130.e9, 140.e9]:
    dT = melting_volumes/melting_entropies*(P - melting_reference_pressure) # dT/dP = DV/DS
    T_melt = melting_temperatures + dT
    temperatures = np.linspace(2000, 5000., 1001)

    Xls = np.empty_like(temperatures)
    Xss = np.empty_like(temperatures)
    X_Fe_bulk = np.empty_like(temperatures)
    X_vol_bulk = np.empty_like(temperatures)
    molar_f_liq = np.empty_like(temperatures)
    for X_vol_melt in [0., 0.2, 0.4, 0.6, 0.8]:
        for i, T in enumerate(temperatures):
            Xls_, Xss_ = liq_sol_molar_compositions(P, T, X_vol_melt,
                                                    melting_reference_pressure,
                                                    melting_temperatures,
                                                    melting_entropies,
                                                    melting_volumes,
                                                    n_mole_mix)

            #print('X_vol_melt should be {0}'.format(X_vol_melt))
            # Now let the molar melt fraction be 0.5 and calculate the bulk composition
            f_liq = 0.5

            X_Fe_bulk[i] = Xls_*f_liq + Xss_*(1. - f_liq)
            X_vol_bulk[i] = X_vol_melt*f_liq


            Xls[i], Xss[i], Xlv_, molar_f_liq[i] = liq_sol_molar_compositions_from_bulk(P, T, X_Fe_bulk[i], X_vol_bulk[i],
                                                                                        melting_reference_pressure,
                                                                                        melting_temperatures,
                                                                                        melting_entropies,
                                                                                        melting_volumes,
                                                                                        n_mole_mix)

        mask = [True if molar_f_liq[i] < 1. and molar_f_liq[i] > 0.
                and X_Fe_bulk[i] < 1. and X_Fe_bulk[i] > 0.
                and X_vol_bulk[i] < 1. and X_vol_bulk[i] > 0. else False
                for i in range(len(molar_f_liq))]

        ax[0].plot((Xls/(1. - X_vol_melt))[mask], temperatures[mask], label='liquid composition, {0} GPa'.format(P/1.e9))
        ax[0].plot(Xss[mask], temperatures[mask], label='solid composition, {0} GPa'.format(P/1.e9))



ax[0].set_xlim(0., 1.)

xs = np.linspace(0., 1., 101)

ax[1].plot(xs, melting_entropies.dot(np.array([1 - xs, xs])), label='melting entropies (linear)')
ax[2].plot(xs, melting_volumes.dot(np.array([1 - xs, xs]))*1.e6, label='melting volumes (linear)')

melt_volumes = np.empty_like(xs)
for j, x_Fe_bearing_endmember in enumerate(xs):
    n_cations_solid = 1./((1. - x_Fe_bearing_endmember)*c_mantle[0]['MgO'] + x_Fe_bearing_endmember*c_mantle[1]['FeO'])
    molar_volumes = calculate_endmember_proportions_volumes_masses(pressure = 100.e9, temperature = 3600.,
                                                                   x_Fe_bearing_endmember_in_solid=x_Fe_bearing_endmember,
                                                                   x_Fe_bearing_endmember_in_melt=x_Fe_bearing_endmember, # again, the proportion of the Fe-rich composition
                                                                   x_volatile_endmember_in_melt=0.,
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
                                                                 0.,
                                                                 0., c_mantle)[2]['melt']
print('molar_mass of melt is {0} kg/mol'.format(molar_mass_melt))
print('Thus deltaH_fusion = {0} J/kg'.format(np.array([1. - X_Fe, X_Fe]).dot((melting_temperatures * melting_entropies +
                                                                          (P - melting_reference_pressure) * melting_volumes))/
                                         molar_mass_melt))

solidus, liquidus = solidus_liquidus_dry(P, X_Fe)

print('the dry solidus and liquidus are at {0} and {1} K'.format(solidus, liquidus))

temperatures = np.linspace(solidus, liquidus, 101)
heat_capacities = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    p, v, m = calculate_endmember_proportions_volumes_masses(P, T,
                                                             X_Fe, X_Fe,
                                                             0.,
                                                             0., # (completely solid)
                                                             c_mantle)

    heat_capacities[i] = (p['mpv']*thermodynamic_properties(P, T, mpv_params)['molar_C_p'] +
                          p['fpv']*thermodynamic_properties(P, T, fpv_params)['molar_C_p'] +
                          p['per']*thermodynamic_properties(P, T, per_params)['molar_C_p'] +
                          p['wus']*thermodynamic_properties(P, T, wus_params)['molar_C_p'])/m['solid']

heat_capacities2 = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    p, v, m = calculate_endmember_proportions_volumes_masses(P, T,
                                                             X_Fe, X_Fe,
                                                             0.,
                                                             1., # (completely liq)
                                                             c_mantle)

    heat_capacities2[i] = (p['mg_melt']*thermodynamic_properties(P, T, mg_mantle_melt_params)['molar_C_p'] +
                          p['fe_melt']*thermodynamic_properties(P, T, fe_mantle_melt_params)['molar_C_p'])/m['melt']


plt.plot(temperatures, heat_capacities)
plt.plot(temperatures, heat_capacities2)
plt.show()
print('')
print('The heat capacity of the solid is ca. {0} J/K/kg across the melting interval'.format(heat_capacities[51]))
print('This corresponds to an enthalpy change of {0} J/kg.'.format(np.trapz(heat_capacities, temperatures)))



temperatures = np.linspace(2000., 5000., 1001)
enthalpies = np.empty_like(temperatures)

fig = plt.figure(figsize=(20, 8))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
for X_vol_bulk in np.linspace(0.0, 0.8, 9):
    for i, T in enumerate(temperatures):
        enthalpies[i] = melting_enthalpy(P, T, X_Fe, X_vol_bulk,
                                         melting_reference_pressure,
                                         melting_temperatures,
                                         melting_entropies,
                                         melting_volumes,
                                         n_mole_mix)

    ax[0].plot(temperatures, enthalpies/1000., label='$x^{{bulk}}_{{volatile}}$: {0}'.format(X_vol_bulk))
    ax[1].plot(temperatures, np.gradient(enthalpies, temperatures), label='$x^{{bulk}}_{{volatile}}$: {0}'.format(X_vol_bulk))

for i in range(2):
    ax[i].set_xlabel('Temperature (K)')
    ax[i].legend()

ax[0].set_ylabel('Excess enthalpy (kJ/mol)')
ax[1].set_ylabel('Excess heat capacity (J/K/mol)')
plt.show()
