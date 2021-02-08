from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
from eos_and_averaging import thermodynamic_properties
from model_parameters import *
from pyrolite_partitioning import aluminous_density_difference, melt_solid_density_using_K_D

from ulvz_melt_model import liq_sol_molar_compositions
from ulvz_melt_model import solidus_liquidus
from ulvz_melt_model import melting_enthalpy
from ulvz_melt_model import calculate_endmember_proportions_volumes_masses
from ulvz_melt_model import calculate_xfe_in_solid_from_molar_proportions


if __name__ == '__main__':
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
    fig.set_size_inches(12.0, 6.0)

    ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

    fig2 = plt.figure()
    fig2.set_size_inches(6.0, 4.0)
    ax2 = [fig2.add_subplot(1, 1, i) for i in range(1, 2)]
    for P in [130.e9]:

        # Calculate temperatures between pure MgO-SiO2 melting and pure FeO-SiO2 melting
        dT = melting_volumes/melting_entropies*(P - melting_reference_pressure) # dT/dP = DV/DS
        T_melt = melting_temperatures + dT
        temperatures = np.linspace(T_melt[0]+1, T_melt[1]-1, 101)


        Xls = np.empty_like(temperatures)
        Xss = np.empty_like(temperatures)
        solrhos = np.empty_like(temperatures)
        meltrhos = np.empty_like(temperatures)
        meltrhos_mod = np.empty_like(temperatures)
        melt2rhos = np.empty_like(temperatures)
        melt3rhos = np.empty_like(temperatures)
        melt4rhos = np.empty_like(temperatures)
        melt5rhos = np.empty_like(temperatures)
        sol3rhos = np.empty_like(temperatures)
        sol4rhos = np.empty_like(temperatures)
        sol5rhos = np.empty_like(temperatures)

        ratio_Fe_solid_liquid = np.empty_like(temperatures)
        KDs = np.empty_like(temperatures)
        KDs_fper_liq = np.empty_like(temperatures)
        for i, T in enumerate(temperatures):
            Xls[i], Xss[i] = liq_sol_molar_compositions(P, T,
                                                        melting_reference_pressure,
                                                        melting_temperatures,
                                                        melting_entropies,
                                                        melting_volumes,
                                                        n_mole_mix)

            sol, vols, masses = calculate_endmember_proportions_volumes_masses(P, T,
                                                                 x_Fe_bearing_endmember=Xss[i],
                                                                 x_Fe_bearing_endmember_in_melt=Xss[i],
                                                                 porosity=0., c_mantle=c_mantle)

            solrhos[i] = masses['solid']/vols['solid']

            melt, vols, masses = calculate_endmember_proportions_volumes_masses(P, T,
                                                                  x_Fe_bearing_endmember=Xls[i],
                                                                  x_Fe_bearing_endmember_in_melt=Xls[i],
                                                                  porosity=1., c_mantle=c_mantle)
            meltrhos[i] = masses['melt']/vols['melt']


            # MODIFY PARAMETERS
            old_V = fe_mantle_melt_params['V_0']
            fe_mantle_melt_params['V_0'] = 2.e-05
            melt, vols, masses = calculate_endmember_proportions_volumes_masses(P, T,
                                                                  x_Fe_bearing_endmember=Xls[i],
                                                                  x_Fe_bearing_endmember_in_melt=Xls[i],
                                                                  porosity=1., c_mantle=c_mantle)
            meltrhos_mod[i] = masses['melt']/vols['melt']
            fe_mantle_melt_params['V_0'] = old_V

            melt2, vols, masses = calculate_endmember_proportions_volumes_masses(P, T,
                                                                   x_Fe_bearing_endmember=Xss[i],
                                                                   x_Fe_bearing_endmember_in_melt=Xss[i],
                                                                   porosity=1., c_mantle=c_mantle)

            melt2rhos[i] = masses['melt']/vols['melt']

            D_sol = (sol['fpv'] + sol['wus'])/(sol['per'] + sol['mpv'])
            D_bulk_solidus = (Xss[i]*c_mantle[1]['FeO'])/((1 - Xss[i])*c_mantle[0]['MgO'])
            assert np.abs(D_bulk_solidus - D_sol) < 1.e-9 # these should be identical

            D_liq = (melt['fe_melt']/melt['mg_melt'])
            D_fper = (sol['wus']/sol['per'])
            ratio_Fe_solid_liquid[i] = (Xss[i]*c_mantle[1]['FeO'])/(Xls[i]*c_mantle[1]['FeO'])

            KDs[i] = D_sol/D_liq
            KDs_fper_liq[i] = D_fper/D_liq

            # 0.33, 0.43, 0.53 corresponds to D_sol_melt of 0.38, 0.47, 0.56 at a pyrolite composition (X_Fe = 0.07)
            melt3rhos[i], sol3rhos[i] = melt_solid_density_using_K_D(P, T, Xss[i], 0.33)
            melt4rhos[i], sol4rhos[i] = melt_solid_density_using_K_D(P, T, Xss[i], 0.43)
            melt5rhos[i], sol5rhos[i] = melt_solid_density_using_K_D(P, T, Xss[i], 0.53)

            # x_Fe*V_Fe + (1 - x_Fe)*V_Mg = V
            # V_Fe = (V - (1 - x_Fe)*V_Mg)/x_Fe
            V_equiv = masses['melt']/melt4rhos[i]

            liq_Mg, vols_Mg, masses_Mg = calculate_endmember_proportions_volumes_masses(P, T,
                                                                 x_Fe_bearing_endmember=0.,
                                                                 x_Fe_bearing_endmember_in_melt=0.,
                                                                 porosity=1., c_mantle=c_mantle)
            liq_Fe, vols_Fe, masses_Fe = calculate_endmember_proportions_volumes_masses(P, T,
                                                                  x_Fe_bearing_endmember=1.,
                                                                  x_Fe_bearing_endmember_in_melt=1.,
                                                                  porosity=1., c_mantle=c_mantle)

            V_equiv_Fe = (vols_Mg['melt'] - (1. - Xls[i])*V_equiv)/Xls[i]
            V_equiv_Mg = (vols_Fe['melt'] - Xls[i]*V_equiv)/(1. - Xls[i])
            #print(Xss[i], Xls[i], V_equiv_Fe - vols_Fe['melt'], V_equiv_Mg - vols_Mg['melt'])


        mask = [i for i in range(len(temperatures)) if Xss[i] < 0.6]
        ax[0].plot(Xls[mask], temperatures[mask], label='liquid composition, {0} GPa'.format(P/1.e9))
        ax[0].plot(Xss[mask], temperatures[mask], label='solid composition, {0} GPa'.format(P/1.e9))


        ax[1].plot(Xss[mask], solrhos[mask], label='solid (solidus)'.format(P/1.e9), linestyle='--')
        ax[1].plot(Xss[mask], melt2rhos[mask], label='isochem. liquid (solidus)'.format(P/1.e9), linestyle=':')
        ax[1].plot(Xss[mask], meltrhos[mask], label='eqm liquid (solidus,mod1)'.format(P/1.e9))
        ax[1].plot(Xss[mask], meltrhos_mod[mask], label='eqm liquid (solidus,mod2)'.format(P/1.e9))


        ax[1].fill_between(Xss[mask], melt3rhos[mask], melt5rhos[mask], label='eqm liquid (solidus,expt)'.format(P/1.e9), alpha=0.2)

        #print('NOTE: this estimate of solid density does not include metallic iron or CaPv')
        #ax[1].plot(Xss[mask], sol4rhos[mask], label='solid at solidus (expt)'.format(P/1.e9))

        ax2[0].semilogy(Xss[mask], KDs[mask], label='X = solid')
        ax2[0].semilogy(Xss[mask], KDs_fper_liq[mask], label='X = fper')
        #ax[3].plot(Xss, KDs)

    xs = np.linspace(0., 1., 101)

    #ax[1].plot(xs, melting_entropies.dot(np.array([1 - xs, xs])), label='melting entropies (linear)')
    #ax[2].plot(xs, melting_volumes.dot(np.array([1 - xs, xs]))*1.e6, label='melting volumes (linear)')

    melt_volumes = np.empty_like(xs)
    for j, x_Fe_bearing_endmember in enumerate(xs):
        n_cations_solid = 1./((1. - x_Fe_bearing_endmember)*c_mantle[0]['MgO'] + x_Fe_bearing_endmember*c_mantle[1]['FeO'])
        molar_volumes = calculate_endmember_proportions_volumes_masses(pressure = 100.e9, temperature = 3600.,
                                                                       x_Fe_bearing_endmember=x_Fe_bearing_endmember,
                                                                       x_Fe_bearing_endmember_in_melt=x_Fe_bearing_endmember, # again, the proportion of the Fe-rich composition
                                                                       porosity=0.0,
                                                                       c_mantle=c_mantle)[1]

        melt_volumes[j] = molar_volumes['melt'] - molar_volumes['solid']/n_cations_solid

    #ax[2].plot(xs, melt_volumes*1.e6, linestyle=':', label='computed melting volumes (not used, but should be close to the line fit)')
    #ax[2].set_ylim(0.,)

    ax[0].plot([0.07, 0.07], [3000., 5000.], label='model mantle composition', color='black')
    ax[0].set_ylim(3000., 5000.)

    ax[1].plot([0.07, 0.07], [4500., 8000.], label='model mantle composition', color='black')
    ax[1].set_ylim(4500., 8000.)

    for i in range(2):
        ax[i].set_xlim(0., 1.)
        ax[i].set_xlabel('molar proportion of Fe bearing endmember')
        ax[i].legend()


    ax[0].set_xlim(0., 0.6)
    ax[1].set_xlim(0., 0.6)

    ax2[0].set_xlim(0., 0.6)
    ax2[0].set_xlabel('molar proportion of Fe bearing endmember')
    #ax2[0].set_ylabel('$f_{FeO,sol}$ / $f_{FeO,liq}$')

    ax2[0].set_ylabel('$K_D$ (($f_{Fe,X}f_{Mg,liq}$) / ($f_{Fe,liq}f_{Mg,X}$))')

    ax[0].set_ylim(3000., 5000.)
    #ax[1].set_ylim(3000., 5000.)
    ax[0].set_ylabel('Temperature (K)')
    #ax[1].set_ylabel('$\Delta S_{melting}$ (J/K/mol_cations)')
    ax[1].set_ylabel('Density (kg/m$^3$)')
    #ax[2].set_ylabel('$\Delta V_{melting}$ (cm$^3$/mol_cations)')

    ax2[0].plot([0.07, 0.07], [0.1, 0.3], label='model mantle', color='black')
    #ax2[0].set_ylim(0.1, 0.3)
    ax2[0].legend()
    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig('output_figures/default_model_densities.pdf')
    fig2.savefig('output_figures/KD_Fe_Mg_sol_liq.pdf')
    plt.show()
