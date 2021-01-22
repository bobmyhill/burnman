import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from scipy.optimize import fsolve
from make_child_solutions import make_child_solutions

from datasets import Frost_2003_fper_ol_wad_rw
from datasets import Matsuzaka_et_al_2000_rw_wus_stv

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))

import burnman


def chain_plotter(sampler, labels):
    samples = sampler.get_chain()
    n_params = samples.shape[2]
    n_params_per_plot = 10
    for j in range(int(np.ceil(n_params/n_params_per_plot))):
        if j == int(np.ceil(n_params/n_params_per_plot) - 1):
            n_plots = n_params - n_params_per_plot*j
        else:
            n_plots = n_params_per_plot
        fig, axes = plt.subplots(n_plots, figsize=(10, n_plots), sharex=True)
        for i in range(n_plots):
            ax = axes[i]
            ax.plot(samples[:, :, j*n_params_per_plot + i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[j*n_params_per_plot + i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        plt.show()


def eqm_pressures(m1, m2, temperatures):
    composition = m1.formula
    assemblage = burnman.Composite([m1, m2])
    assemblage.set_state(1.e5, temperatures[0])
    equality_constraints = [('T', temperatures),
                            ('phase_proportion', (m1, 0.0))]
    sols, prm = burnman.equilibrate(composition, assemblage,
                                    equality_constraints,
                                    initial_state_from_assemblage=True,
                                    store_iterates=False)
    if type(sols) is list:
        return np.array([sol.x[0] for sol in sols])
    else:
        return sols.x[0]


# FPER-OL POLYMORPH (OR GARNET) PARTITIONING
def affinity_ol_fper(v, x_ol, G, T, W_ol, W_fper):
    # G is deltaG = G_per + G_fa/2. - G_fper - G_fo/2.
    x_fper = v[0]
    if np.abs(np.abs(x_ol - 0.5) - 0.5) < 1.e-10:
        v[0] = x_ol
        return 0.
    else:
        KD = ((x_ol*(1. - x_fper))
              / (x_fper*(1. - x_ol)))
        if KD < 0.:
            KD = 1.e-12
        return (G - W_ol*(2.*x_ol - 1.) - W_fper*(1 - 2.*x_fper)
                + burnman.constants.gas_constant*T*np.log(KD))


def plots(dataset, storage):
    endmembers = dataset['endmembers']
    solutions = dataset['solutions']
    child_solutions = make_child_solutions(solutions)

    # A few images
    ol_polymorph_img = mpimg.imread('figures/ol_polymorphs.png')
    ol_polymorph_img_1200C = mpimg.imread('figures/Akimoto_1987_fo_fa_phase_diagram_1200C.png')
    ol_polymorph_img_1000C = mpimg.imread('figures/Akimoto_1987_fo_fa_phase_diagram_1000C.png')
    ol_polymorph_img_800C = mpimg.imread('figures/Akimoto_1987_fo_fa_phase_diagram_800C.png')

    ol_fper_img = mpimg.imread('figures/ol_fper_RTlnKD.png')
    wad_fper_img = mpimg.imread('figures/wad_fper_RTlnKD.png')
    rw_fper_img = mpimg.imread('figures/ring_fper_gt_KD.png')
    rw_fper_part_img = mpimg.imread('figures/ring_fper_partitioning.png')


    fo_phase_diagram = mpimg.imread('figures/Mg2SiO4_phase_diagram_Jacobs_2017.png')
    fa_phase_diagram = mpimg.imread('figures/Fe2SiO4_phase_diagram_Yagi_1987.png')
    fa_phase_diagram2 = mpimg.imread('figures/Fe2SiO4_phase_diagram_Jacobs_2001.png')

    """
    # Plot mrw EoS
    pressures = np.linspace(1.e5, 25.e9, 101)
    plt.imshow(mrw_volume_diagram, extent=[0., 25., 35.5,40.5], aspect='auto')
    for T in [300., 700., 1100., 1500., 1900.]:
        temperatures = pressures*0. + T
        plt.plot(pressures/1.e9, mrw.evaluate(['V'], pressures, temperatures)[0]*1.e6)
    plt.show()


    fig = plt.figure(figsize=(30, 15))
    ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
    ax[0].imshow(fo_phase_diagram, extent=[1000., 2600., 5, 30], aspect='auto')

    temperatures = np.linspace(1000., 2600., 21)
    ax[0].plot(temperatures, eqm_pressures(fo, mwd, temperatures)/1.e9, linewidth=4.)
    ax[0].plot(temperatures, eqm_pressures(mwd, mrw, temperatures)/1.e9, linewidth=4.)


    #ax[1].imshow(fa_phase_diagram, extent=[3., 7., 550.+273.15, 1350.+273.15], aspect='auto')
    #temperatures = np.linspace(550.+273.15, 1350.+273.15, 21)
    #ax[1].plot(eqm_pressures(fa, frw, temperatures)/1.e9, temperatures, linewidth=4.)
    #ax[1].plot(eqm_pressures(fa, fwd, temperatures)/1.e9, temperatures, linestyle=':', linewidth=4.)


    ax[1].imshow(fa_phase_diagram2, extent=[700., 1900., 0., 10.], aspect='auto')
    temperatures = np.linspace(700., 1900., 21)
    ax[1].plot(temperatures, eqm_pressures(fa, frw, temperatures)/1.e9, linewidth=4., label='fa-frw')
    ax[1].plot(temperatures, eqm_pressures(fa, fwd, temperatures)/1.e9, linestyle=':', linewidth=4., label='fa-fwd (should be metastable)')


    ax[0].set_xlabel('T (K)')
    ax[0].set_ylabel('P (GPa)')
    ax[1].set_xlabel('P (GPa)')
    ax[1].set_ylabel('T (K)')
    ax[1].legend()
    plt.show()

    # PLOT VOLUMES COMPARED WITH OTHER DATASETS
    class Murnaghan_EOS(object):
        def __init__(self, V0, K, Kprime):
            self.V0 = V0
            self.K = K
            self.Kprime = Kprime
            self.V = lambda P: self.V0*np.power(1. + P*(self.Kprime/self.K),
                                                -1./self.Kprime)


    M_fo  = Murnaghan_EOS(4.6053e-5, 95.7e9, 4.6)
    M_mwd = Murnaghan_EOS(4.2206e-5, 146.2544e9, 4.21)
    M_mrw = Murnaghan_EOS(4.1484e-5, 145.3028e9, 4.4)
    M_per = Murnaghan_EOS(1.1932e-5, 125.9e9, 4.1)
    M_py  = Murnaghan_EOS(11.8058e-5, 129.0e9, 4.)

    M_fa = Murnaghan_EOS(4.8494e-5, 99.8484e9, 4.)
    M_fwd = Murnaghan_EOS(4.4779e-5, 139.9958e9, 4.)
    M_frw = Murnaghan_EOS(4.3813e-5, 160.781e9, 5.)
    M_fper = Murnaghan_EOS(1.2911e-5, 152.6e9, 4.)
    M_alm = Murnaghan_EOS(12.1153e-5, 120.7515e9, 5.5)

    from burnman.minerals import HHPH_2013, SLB_2011
    dss = [[M_per, M_fper, M_fo, M_fa, M_mwd, M_fwd, M_mrw, M_frw, 'Frost (2003)'],
           [HHPH_2013.per(), HHPH_2013.fper(),
            HHPH_2013.fo(), HHPH_2013.fa(),
            HHPH_2013.mwd(), HHPH_2013.fwd(),
            HHPH_2013.mrw(), HHPH_2013.frw(), 'HHPH'],
           [SLB_2011.periclase(), SLB_2011.wuestite(),
            SLB_2011.forsterite(), SLB_2011.fayalite(),
            SLB_2011.mg_wadsleyite(), SLB_2011.fe_wadsleyite(),
            SLB_2011.mg_ringwoodite(), SLB_2011.fe_ringwoodite(), 'SLB'],
           [per,  wus,  fo,  fa,  mwd,  fwd,  mrw,  frw, 'this study']]


    fig = plt.figure(figsize=(24,12))
    ax = [fig.add_subplot(2, 4, i) for i in range(1, 9)]
    pressures = np.linspace(1.e5, 24.e9, 101)
    T=1673.15
    temperatures = pressures*0. + T
    for i, ds in enumerate(dss):
        if i == 0:
            Vs = [m.V(pressures) for m in ds[0:-1]]
        else:
            Vs = [m.evaluate(['V'], pressures, temperatures)[0] for m in ds[0:-1]]
        if i==3:
            linewidth=3.
        else:
            linewidth=1.

        for j, V in enumerate(Vs):
            try:
                ax[j].plot(pressures/1.e9, Vs[j]*1.e6, label=ds[j].name+' '+ds[-1],
                           linewidth=linewidth)
            except:
                ax[j].plot(pressures/1.e9, Vs[j]*1.e6, label=ds[-1],
                           linewidth=linewidth)

    for i in range(0, 8):
        ax[i].legend(loc='best')
    plt.show()

    """
    viridis = cm.get_cmap('viridis', 101)

    fig = plt.figure(figsize=(30, 10))
    ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

    # OLIVINE
    ax[0].imshow(ol_fper_img, extent=[0.0, 0.8, -45000., -5000.],
                 aspect='auto')

    Pmin = 0.e9
    Pmax = 15.e9

    T = 1673.15
    mins = [endmembers[m]
            for m in ['per', 'wus', 'fo', 'fa', 'mwd', 'fwd', 'mrw', 'frw']]
    for P in [1.e5, 5.e9, 10.e9, 15.e9]:
        for m in mins:
            m.set_state(P, T)
        G = (endmembers['per'].gibbs - endmembers['wus'].gibbs
             - endmembers['fo'].gibbs/2. + endmembers['fa'].gibbs/2.)
        W_ol = (solutions['ol'].solution_model.We[0][1]
                + solutions['ol'].solution_model.Wv[0][1] * P) / 2.  # 1 cation
        W_fper = (solutions['mw'].solution_model.We[0][1]
                  + solutions['mw'].solution_model.Wv[0][1] * P)

        x_ols = np.linspace(0.00001, 0.99999, 101)
        x_fpers = np.array([fsolve(affinity_ol_fper, [x_ol],
                                   args=(x_ol, G, T, W_ol, W_fper))[0]
                            for x_ol in x_ols])
        KDs = ((x_ols*(1. - x_fpers))
               / (x_fpers*(1. - x_ols)))
        ax[0].plot(x_ols, burnman.constants.gas_constant*T*np.log(KDs),
                   color=viridis((P - Pmin) / (Pmax - Pmin)),
                   linewidth=3., label='{0} GPa'.format(P/1.e9))

    Frost_2003_assemblages = Frost_2003_fper_ol_wad_rw.get_assemblages(dataset)
    P_Xol_RTlnKDs = []
    for assemblage in Frost_2003_assemblages:
        if solutions['ol'] in assemblage.phases:
            idx_ol = assemblage.phases.index(solutions['ol'])
            idx_mw = assemblage.phases.index(solutions['mw'])
            T = assemblage.nominal_state[1]
            x_ol = assemblage.stored_compositions[idx_ol][0][1]
            x_fper = assemblage.stored_compositions[idx_mw][0][1]
            RTlnKD = (burnman.constants.gas_constant * T
                      * np.log((x_ol*(1. - x_fper))
                               / (x_fper*(1. - x_ol))))
            P_Xol_RTlnKDs.append([assemblage.nominal_state[0],
                                  x_ol, RTlnKD])

    pressures, x_ols, RTlnKDs = np.array(P_Xol_RTlnKDs).T
    ax[0].scatter(x_ols, RTlnKDs, c=pressures, s=80., label='data',
                  cmap=viridis, vmin=Pmin, vmax=Pmax)

    ax[0].set_xlim(0., 0.8)
    ax[0].legend(loc='best')

    # WADSLEYITE
    ax[1].imshow(wad_fper_img, extent=[0.0, 0.4, -25000., -5000.],
                 aspect='auto')

    viridis = cm.get_cmap('viridis', 101)
    Pmin = 10.e9
    Pmax = 18.e9

    T = 1673.15
    for P in [10.e9, 12.e9, 14.e9, 16.e9, 18.e9]:
        for m in mins:
            m.set_state(P, T)
        G = (endmembers['per'].gibbs - endmembers['wus'].gibbs
             - endmembers['mwd'].gibbs/2. + endmembers['fwd'].gibbs/2.)
        W_wad = (solutions['wad'].solution_model.We[0][1]
                 + solutions['wad'].solution_model.Wv[0][1] * P) / 2.  # 1 cation
        W_fper = (solutions['mw'].solution_model.We[0][1]
                  + solutions['mw'].solution_model.Wv[0][1] * P)

        x_wads = np.linspace(0.00001, 0.99999, 101)
        x_fpers = np.array([fsolve(affinity_ol_fper, [x_wad],
                                   args=(x_wad, G, T, W_wad, W_fper))[0]
                            for x_wad in x_wads])
        KDs = ((x_wads*(1. - x_fpers))
               / (x_fpers*(1. - x_wads)))
        ax[1].plot(x_wads, burnman.constants.gas_constant*T*np.log(KDs),
                   color=viridis((P-Pmin)/(Pmax-Pmin)),
                   linewidth=3., label='{0} GPa'.format(P/1.e9))

    P_Xwad_RTlnKDs = []
    for assemblage in Frost_2003_assemblages:
        if solutions['wad'] in assemblage.phases:
            idx_wad = assemblage.phases.index(solutions['wad'])
            idx_mw = assemblage.phases.index(solutions['mw'])
            x_wad = assemblage.stored_compositions[idx_wad][0][1]
            x_fper = assemblage.stored_compositions[idx_mw][0][1]
            T = assemblage.nominal_state[1]
            RTlnKD = (burnman.constants.gas_constant * T
                      * np.log((x_wad*(1. - x_fper))
                               / (x_fper*(1. - x_wad))))
            P_Xwad_RTlnKDs.append([assemblage.nominal_state[0],
                                   x_wad, RTlnKD])

    pressures, x_wads, RTlnKDs = np.array(P_Xwad_RTlnKDs).T
    ax[1].scatter(x_wads, RTlnKDs, c=pressures, s=80., label='data',
                  cmap=viridis, vmin=Pmin, vmax=Pmax)

    ax[1].set_xlim(0., 0.4)
    ax[1].legend(loc='best')

    # RINGWOODITE
    # ax[2].imshow(rw_fper_part_img, extent=[0.0, 1., 0., 1.], aspect='auto')

    viridis = cm.get_cmap('viridis', 101)
    Pmin = 10.e9
    Pmax = 24.e9

    T = 1673.15
    for P in [10.e9, 12.5e9, 15.e9, 17.5e9, 20.e9]:
        for m in mins:
            m.set_state(P, T)
        G = (endmembers['per'].gibbs - endmembers['wus'].gibbs
             - endmembers['mrw'].gibbs / 2. + endmembers['frw'].gibbs / 2.)
        W_rw = ((child_solutions['ring'].solution_model.We[0][1]
                 + child_solutions['ring'].solution_model.Wv[0][1] * P)
                / 2.)  # 1 cation
        W_fper = (solutions['mw'].solution_model.We[0][1]
                  + solutions['mw'].solution_model.Wv[0][1] * P)

        x_rws = np.linspace(0.00001, 0.99999, 101)
        x_fpers = np.array([fsolve(affinity_ol_fper, [x_rw],
                                   args=(x_rw, G, T, W_rw, W_fper))[0]
                            for x_rw in x_rws])

        ax[2].plot(x_rws, x_fpers, color=viridis((P-Pmin)/(Pmax-Pmin)),
                   linewidth=3., label=P/1.e9)

    P_Xrw_Xfper = []
    for assemblage in Frost_2003_assemblages:
        if solutions['sp'] in assemblage.phases:
            idx_rw = assemblage.phases.index(solutions['sp'])
            idx_mw = assemblage.phases.index(solutions['mw'])
            P_Xrw_Xfper.append([assemblage.nominal_state[0],
                                assemblage.stored_compositions[idx_rw][0][4],  # frw is the 5th element
                                assemblage.stored_compositions[idx_mw][0][1]])  # fper is the 2nd element

    pressures, x_rws, x_fpers = np.array(P_Xrw_Xfper).T
    c = ax[2].scatter(x_rws, x_fpers, c=pressures, s=80., label='data',
                      cmap=viridis, vmin=Pmin, vmax=Pmax)

    ax[2].set_xlim(0., 1.)
    ax[2].set_ylim(0., 1.)
    ax[2].legend(loc='best')
    plt.show()

    # BINARY PHASE DIAGRAM

    # plt.imshow(ol_polymorph_img, extent=[0., 1., 6., 20.], aspect='auto')
    # plt.imshow(ol_polymorph_img_1000C, extent=[-0.01, 1.005, 4., 21.],
    # aspect='auto')

    for (T0, color) in [(1273.15, 'blue'),
                        (1673.15, 'orange'),
                        (2073.15, 'purple')]:

        x_m1 = 0.3

        composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1.-x_m1), 'Si': 1., 'O': 4.}
        solutions['wad'].guess = np.array([1. - x_m1, x_m1])
        solutions['ol'].guess = np.array([1. - x_m1, x_m1])
        child_solutions['ring'].guess = np.array([0.15, 0.85])
        assemblage = burnman.Composite([solutions['ol'],
                                        solutions['wad'],
                                        child_solutions['ring']])
        assemblage.set_state(14.e9, T0)
        equality_constraints = [('T', T0), ('phase_proportion',
                                            (solutions['ol'], 0.0))]
        sol, prm = burnman.equilibrate(composition, assemblage,
                                       equality_constraints,
                                       store_iterates=False,
                                       initial_state_from_assemblage=True)
        P_inv = assemblage.pressure
        x_ol_inv = assemblage.phases[0].molar_fractions[1]
        x_wad_inv = assemblage.phases[1].molar_fractions[1]
        x_rw_inv = assemblage.phases[2].molar_fractions[1]
        for (m1, m2) in [(solutions['wad'], solutions['ol']),
                         (solutions['wad'], child_solutions['ring']),
                         (solutions['ol'], child_solutions['ring'])]:
            composition = {'Fe': 0., 'Mg': 2., 'Si': 1., 'O': 4.}
            assemblage = burnman.Composite([m1.endmembers[0][0],
                                            m2.endmembers[0][0]])
            equality_constraints = [('T', T0), ('phase_proportion',
                                                (m1.endmembers[0][0], 0.0))]
            sol, prm = burnman.equilibrate(composition, assemblage,
                                           equality_constraints,
                                           store_iterates=False)
            P1 = assemblage.pressure
            composition = {'Fe': 2., 'Mg': 0., 'Si': 1., 'O': 4.}
            assemblage = burnman.Composite([m1.endmembers[1][0],
                                            m2.endmembers[1][0]])
            equality_constraints = [('T', T0), ('phase_proportion',
                                                (m1.endmembers[1][0], 0.0))]
            sol, prm = burnman.equilibrate(composition, assemblage,
                                           equality_constraints,
                                           store_iterates=False)
            P0 = assemblage.pressure
            print(P0/1.e9, P1/1.e9)
            if m1 is solutions['wad']:
                x_m1s = np.linspace(0.001, x_wad_inv, 21)
            else:
                x_m1s = np.linspace(x_ol_inv, 0.999, 21)

            pressures = np.empty_like(x_m1s)
            x_m2s = np.empty_like(x_m1s)
            m1.guess = np.array([1. - x_m1s[0], x_m1s[0]])
            m2.guess = np.array([1. - x_m1s[0], x_m1s[0]])
            for i, x_m1 in enumerate(x_m1s):
                composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1. - x_m1),
                               'Si': 1., 'O': 4.}
                assemblage = burnman.Composite([m1, m2])
                assemblage.set_state(P1*(1 - x_m1) + P0*x_m1, T0)
                m1.set_composition([1. - x_m1, x_m1])
                m2.set_composition(m2.guess)
                assemblage.n_moles = 1.
                assemblage.set_fractions([1., 0.])
                equality_constraints = [('T', T0), ('phase_proportion', (m2, 0.0))]
                sol, prm = burnman.equilibrate(composition, assemblage,
                                               equality_constraints,
                                               initial_state_from_assemblage=True,
                                               initial_composition_from_assemblage=True,
                                               store_iterates=False)

                m2.guess = m2.molar_fractions
                x_m2s[i] = m2.molar_fractions[1]
                pressures[i] = assemblage.pressure

            plt.plot(x_m1s, pressures/1.e9, linewidth=3., color=color)
            plt.plot(x_m2s, pressures/1.e9, linewidth=3.,
                     color=color, label='{0} K'.format(T0))
        plt.plot([x_ol_inv, x_rw_inv], [P_inv/1.e9, P_inv/1.e9], linewidth=3.,
                 color=color)

        """
        # bdg + fper
        x_m1s = []
        pressures = []
        x_m2s = []
        for x_m1 in np.linspace(0.3, 0.5, 51):
            composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1. - x_m1), 'Si': 1., 'O': 4.}
            child_solutions['mg_fe_bdg'].guess = np.array([1. - x_m1, x_m1])
            fper.guess = np.array([1. - x_m1, x_m1])
            assemblage = burnman.Composite([child_solutions['mg_fe_bdg'],
                                            fper, stv])
            assemblage.set_state(30.e9, T0)
            equality_constraints = [('T', T0),
                                    ('phase_proportion', (stv, 0.))]
            sol, prm = burnman.equilibrate(composition, assemblage,
                                           equality_constraints,
                                           initial_state_from_assemblage=True,
                                           store_iterates=False)
            if sol.success:
                print('yo', assemblage.pressure/1.e9)
                x_m1s.append(x_m1)
                x_m2s.append(fper.molar_fractions[1])
                pressures.append(assemblage.pressure)

        plt.plot(x_m1s, np.array(pressures)/1.e9, linewidth=3., color=color)
        plt.plot(x_m2s, np.array(pressures)/1.e9, linewidth=3., color=color)
        """

        # bdg + fper
        x_m1s = []
        pressures = []
        x_m2s = []
        x_m1_array = np.linspace(0.01, 0.3, 21)

        child_solutions['ring'].guess = np.array([1. - x_m1_array[0],
                                                  x_m1_array[0]])
        child_solutions['mg_fe_bdg'].guess = np.array([1. - x_m1_array[0],
                                                       x_m1_array[0]])
        solutions['mw'].guess = np.array([1. - x_m1_array[0], x_m1_array[0]])

        for x_m1 in x_m1_array:
            composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1. - x_m1), 'Si': 1., 'O': 4.}

            child_solutions['ring'].set_composition([1. - x_m1, x_m1])
            child_solutions['mg_fe_bdg'].set_composition(child_solutions['mg_fe_bdg'].guess)
            solutions['mw'].set_composition(solutions['mw'].guess)

            assemblage = burnman.Composite([child_solutions['ring'],
                                            child_solutions['mg_fe_bdg'],
                                            solutions['mw']], [1., 0., 0.])
            assemblage.set_state(25.e9, T0)
            equality_constraints = [('T', T0),
                                    ('phase_proportion',
                                     (child_solutions['ring'], 1.0))]
            sol, prm = burnman.equilibrate(composition, assemblage,
                                           equality_constraints,
                                           initial_composition_from_assemblage=True,
                                           initial_state_from_assemblage=True,
                                           store_iterates=False)
            if sol.success:
                print(assemblage.pressure/1.e9)
                x_m1s.append(x_m1)
                x_m2s.append((solutions['mw'].molar_fractions[1]
                              + child_solutions['mg_fe_bdg'].molar_fractions[1])
                             / 2.)

                child_solutions['mg_fe_bdg'].guess = child_solutions['mg_fe_bdg'].molar_fractions
                solutions['mw'].guess = solutions['mw'].molar_fractions

                pressures.append(assemblage.pressure)

        plt.plot(x_m1s, np.array(pressures)/1.e9, linewidth=3., color=color)
        plt.plot(x_m2s, np.array(pressures)/1.e9, linewidth=3., color=color)

        # rw -> fper + stv
        x_m1s = np.linspace(0.2, 0.99, 21)
        pressures = np.empty_like(x_m1s)
        x_m2s = np.empty_like(x_m1s)

        child_solutions['ring'].guess = np.array([1. - x_m1s[0], x_m1s[0]])
        Pi = 22.e9
        for i, x_m1 in enumerate(x_m1s):
            composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1. - x_m1), 'Si': 1., 'O': 4.}
            assemblage = burnman.Composite([child_solutions['ring'],
                                            solutions['mw'],
                                            endmembers['stv']],
                                           [0., 2./3., 1./3.])
            assemblage.set_state(Pi, T0)

            solutions['mw'].set_composition([1. - x_m1, x_m1])
            child_solutions['ring'].set_composition(child_solutions['ring'].guess)

            equality_constraints = [('T', T0),
                                    ('phase_proportion',
                                     (child_solutions['ring'], 0.0))]
            sol, prm = burnman.equilibrate(composition, assemblage,
                                           equality_constraints,
                                           initial_composition_from_assemblage=True,
                                           initial_state_from_assemblage=True,
                                           store_iterates=False)
            x_m2s[i] = child_solutions['ring'].molar_fractions[1]
            child_solutions['ring'].guess = child_solutions['ring'].molar_fractions
            pressures[i] = assemblage.pressure
            Pi = assemblage.pressure

        plt.plot(x_m1s, pressures/1.e9, linewidth=3., color=color)
        plt.plot(x_m2s, pressures/1.e9, linewidth=3., color=color)


    Matsuzaka_2000_assemblages = Matsuzaka_et_al_2000_rw_wus_stv.get_assemblages(dataset)

    P_rw_fper = []
    for assemblage in Matsuzaka_2000_assemblages:
        P_rw_fper.append([assemblage.nominal_state[0], assemblage.nominal_state[1],
                          assemblage.stored_compositions[0][0][1],
                          assemblage.stored_compositions[1][0][1]])

    P, Ts, x_rw, x_fper = np.array(P_rw_fper).T
    mask = [i for i, T in enumerate(Ts) if np.abs(T - 1673.15) < 0.1]
    plt.scatter(x_rw[mask], P[mask]/1.e9, color='orange', label='1673.15 K')
    plt.scatter(x_fper[mask], P[mask]/1.e9, color='orange')
    mask = [i for i, T in enumerate(Ts) if np.abs(T - 1873.15) < 0.1]
    plt.scatter(x_rw[mask], P[mask]/1.e9, color='red', label='1873.15 K')
    plt.scatter(x_fper[mask], P[mask]/1.e9, color='red')

    P_Xfe_phase = {'ol': [], 'wad': [], 'sp': []}
    for assemblage in Frost_2003_assemblages:
        if len(assemblage.phases) > 2:
            for i, phase in enumerate(assemblage.phases):
                for m in ['ol', 'wad', 'sp']:
                    if phase == solutions[m]:
                        P_shift = storage['dict_experiment_uncertainties'][assemblage.experiment_id]['P']

                        P_Xfe_phase[m].append([assemblage.nominal_state[0],
                                               P_shift,
                                               assemblage.stored_compositions[i][0][-1]])

    arrow_params = {'shape': 'full',
                    'width': 0.001,
                    'length_includes_head': True,
                    'head_starts_at_zero': False}

    for m in ['ol', 'wad', 'sp']:
        pressures, pressure_shift, xs = np.array(P_Xfe_phase[m]).T
        for i in range(len(xs)):
            plt.arrow(xs[i], pressures[i]/1.e9, 0.,
                      pressure_shift[i]/1.e9,
                      **arrow_params)
        plt.scatter(xs, pressures/1.e9, s=80., label='data')

    plt.legend()
    plt.show()
    return None
