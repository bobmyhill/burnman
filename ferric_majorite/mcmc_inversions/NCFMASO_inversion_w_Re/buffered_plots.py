import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from scipy.optimize import fsolve


# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))

import burnman


def buffered_KLB_plot(dataset, storage, buffer, n_log_units):
    n_ln_units = np.log(np.power(10., n_log_units))

    endmembers = dataset['endmembers']
    solutions = dataset['solutions']
    child_solutions = dataset['child_solutions']

    # Alias solutions:
    ol = solutions['ol']
    wad = solutions['wad']
    rw = child_solutions['ring']
    gt = solutions['gt']
    cpx_od = solutions['cpx']


    # KLB-1 (Takahashi, 1986; Walter, 1998; Holland et al., 2013)
    KLB_1_composition_plus_oxygen = {'Si': 39.4,
                         'Al': 2.*2.,
                         'Ca': 3.3,
                         'Mg': 49.5,
                         'Fe': 5.2,
                         'Na': 0.26*2.,
                         'O': 39.4*2. + 2.*3. + 3.3 + 49.5 + 5.2 + 0.26 + 2.} # reduced starting mix + O2

    # KLB-1 first
    ol.guess = np.array([0.9, 0.1])
    wad.guess = np.array([0.87, 0.13])
    rw.guess = np.array([0.84, 0.16])
    gt.guess = np.array([0.6, 0.2, 0.15, 0.03, 0.01, 0.01])
    cpx_od.guess = np.array([0.7, 0.1, 0.05, 0.02, 0.05, 0.03, 0.05])


    if buffer == 'EMOD':
        #0.5*(Mg2Si2O6) + MgCO3 - Mg2SiO4 - C  = O2
        HP_hen = burnman.minerals.HGP_2018_ds633.hen()
        HP_mag = burnman.minerals.HGP_2018_ds633.mag()
        HP_fo = burnman.minerals.HGP_2018_ds633.fo()
        HP_mwd = burnman.minerals.HGP_2018_ds633.mwd()
        HP_mrw = burnman.minerals.HGP_2018_ds633.mrw()
        HP_diam = burnman.minerals.HGP_2018_ds633.diam()

        EMOD_O2 = burnman.CombinedMineral([HP_hen, HP_mag, HP_fo, HP_diam],
                                          [0.5, 1., -1., -1.])
        EMWD_O2 = burnman.CombinedMineral([HP_hen, HP_mag, HP_mwd, HP_diam],
                                          [0.5, 1., -1., -1.])
        EMRD_O2 = burnman.CombinedMineral([HP_hen, HP_mag, HP_mrw, HP_diam],
                                          [0.5, 1., -1., -1.])
        combined_buffer = burnman.CombinedMineral([HP_hen, HP_mag, HP_fo, HP_diam],
                                                  [0.5, 1., -1., -1.],
                                                  [0., -n_ln_units*burnman.constants.gas_constant,
                                                   0.])
    elif buffer == 'Re-ReO2':
        combined_buffer = burnman.CombinedMineral([endmembers['ReO2'],
                                                   endmembers['Re']],
                                                  [1., -1.],
                                                  [0., -n_ln_units*burnman.constants.gas_constant,
                                                   0.])
    else:
        raise Exception('buffer not recognised')




    P_min = 6.e9

    ol.guess = np.array([0.9, 0.1])
    wad.guess = np.array([0.8, 0.2])
    rw.guess = np.array([0.84, 0.16])
    gt.guess = np.array([0.35, 0.2, 0.1, 0.05, 0.25, 0.05])
    cpx_od.guess = np.array([0.7, 0.1, 0.05, 0.02, 0.05, 0.03, 0.05])


    P0 = 13.e9
    T0 = 1750.
    composition = KLB_1_composition_plus_oxygen
    assemblage = burnman.Composite([ol, gt, cpx_od, combined_buffer])
    equality_constraints = [('P', P0), ('T', T0)]

    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   store_iterates=False)

    print(gt.molar_fractions)

    fs = assemblage.molar_fractions
    fs.append(0.)
    n = assemblage.n_moles
    wad.set_composition(wad.guess)
    assemblage = burnman.Composite([ol, gt, cpx_od, combined_buffer, wad], fs)
    assemblage.n_moles = n
    assemblage.set_state(P0, T0)
    equality_constraints = [('T', T0), ('phase_proportion', (wad, 0.0))]
    sol_wad_in, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                          store_assemblage=True,
                                          store_iterates=False)

    sol_wad_in = [sol_wad_in]
    print(gt.molar_fractions)

    P_wad_in = assemblage.pressure

    equality_constraints = [('T', T0), ('phase_proportion', (ol, 0.0))]
    sol_ol_out, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                          store_assemblage=True,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)

    sol_ol_out = [sol_ol_out]
    P_ol_out = assemblage.pressure

    assemblage = burnman.Composite([wad, cpx_od, combined_buffer, gt])
    equality_constraints = [('T', T0), ('phase_proportion', (cpx_od, 0.0))]
    sol_cpx_out, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                           store_assemblage=True,
                                           store_iterates=False)

    sol_cpx_out = [sol_cpx_out]
    P_cpx_out = assemblage.pressure

    P0 = 16.e9
    assemblage = burnman.Composite([wad, gt, combined_buffer])
    equality_constraints = [('P', P0), ('T', T0)]

    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   store_iterates=False)

    rw.set_composition(rw.guess)
    fs = assemblage.molar_fractions
    fs.append(0.)
    n = assemblage.n_moles
    assemblage = burnman.Composite([wad, gt, combined_buffer, rw], fs)
    assemblage.n_moles = n
    assemblage.set_state(sol.x[0], sol.x[1])

    equality_constraints = [('T', T0), ('phase_proportion', (rw, 0.0))]

    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)
    P_rw_in = assemblage.pressure

    equality_constraints = [('T', T0), ('phase_proportion', (wad, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)

    P_wad_out = assemblage.pressure
    print(np.array([P_wad_in, P_ol_out, P_cpx_out, P_rw_in, P_wad_out])/1.e9)

    # ol-cpx-gt-iron

    assemblage = burnman.Composite([ol, cpx_od, combined_buffer, gt])
    pressures = np.linspace(P_min, P_wad_in-0.2e9, 21)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_ol_cpx, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                           store_assemblage=True,
                                           initial_state_from_assemblage=False,
                                           initial_composition_from_assemblage=False,
                                           store_iterates=False)

    a = sol_ol_out[0].assemblage.copy()
    assemblage = burnman.Composite([ph for ph
                                    in a.phases if ph.name != 'olivine'])
    assemblage.set_state(a.pressure, a.temperature)
    assemblage.set_fractions([a.molar_fractions[i] for i
                              in range(len(a.phases))
                              if a.phases[i].name != 'olivine'])
    assemblage.n_moles = a.n_moles
    pressures = np.linspace(P_ol_out, P_cpx_out, 2)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_wad_cpx, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                            initial_state_from_assemblage=True,
                                            initial_composition_from_assemblage=True,
                                            store_assemblage=True,
                                            store_iterates=False)


    assemblage = burnman.Composite([wad, combined_buffer, gt])
    pressures = np.linspace(P_cpx_out, P_rw_in, 21)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_wad, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                        store_assemblage=True,
                                        store_iterates=False)



    assemblage = burnman.Composite([rw, combined_buffer, gt])
    pressures = np.linspace(P_wad_out, 20.e9, 21)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_rw, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                       store_assemblage=True,
                                       store_iterates=False)

    # Plotting
    pressures = []
    Fe3 = []
    FeT = []
    x_dmaj = []
    x_nmaj = []
    for sols in [sols_ol_cpx, sol_wad_in, sol_ol_out,
                 sols_wad, sols_rw]:
        pressures.extend([sol.assemblage.pressure
                          for sol in sols if sol.success])
        gt_idx = [phase.name for phase
                  in sols[0].assemblage.phases].index('garnet')
        c_gt = np.array([sol.assemblage.phases[gt_idx].molar_fractions
                         for sol in sols if sol.success])
        Fe3.extend([c[3]*2./(c[1]*3. + c[3]*2.) for c in c_gt])
        FeT.extend([(c[1]*3. + c[3]*2.) for c in c_gt])
        x_dmaj.extend([c[4] for c in c_gt])
        x_nmaj.extend([c[5] for c in c_gt])


    pressures = np.array(pressures)

    plt.style.use('ggplot')
    plt.plot(pressures/1.e9, x_dmaj, label='p(Mg$_3$(MgSi)Si$_3$O$_{{12}}$)')
    plt.plot(pressures/1.e9, x_nmaj, label='p(NaMg$_2$(AlSi)Si$_3$O$_{{12}}$)')
    plt.plot(pressures/1.e9, FeT, label='Fe atoms per 12 O')
    plt.plot(pressures/1.e9, Fe3, label='Fe$^{{3+}}$/(Fe$^{{2+}}$ + Fe$^{{3+}}$)')
    plt.legend()

    for P in [P_wad_in, P_ol_out, P_cpx_out, P_rw_in, P_wad_out]:
        plt.plot(np.array([P, P])/1.e9, [0., 0.6], color='k', linestyle=':')

    field_labels = [[P_min, P_wad_in, 'cpx+ol+gt'],
                    [P_ol_out, P_cpx_out, 'cpx+wad+gt'],
                    [P_cpx_out, P_rw_in, 'wad+gt'],
                    [P_wad_out, 20.e9, 'rw+gt']]

    for P0, P1, lbl in field_labels:
        plt.text((P0+P1)/2./1.e9, 0.3, lbl,
                 horizontalalignment='center',
                 verticalalignment='center')

    plt.title('Garnet in KLB-1 peridotite at {0} K'.format(T0))
    plt.xlabel('P (GPa)')
    plt.ylabel('composition')
    plt.xlim(P_min/1.e9,20.)
    plt.ylim(0.,1.0)
    plt.savefig('KLB-1_gt_{0}_plus_{1}_log_units.pdf'.format(buffer, n_log_units))
    plt.show()


def buffered_MORB_plot(dataset, storage, buffer, n_log_units):
    n_ln_units = np.log(np.power(10., n_log_units))

    endmembers = dataset['endmembers']
    solutions = dataset['solutions']
    child_solutions = dataset['child_solutions']
    
    # Alias solutions:
    gt = solutions['gt']
    cpx_od = solutions['cpx']
    stv = endmembers['stv']

    # MORB (Litasov et al., 2005)
    MORB_composition_plus_oxygen = {'Si': 53.9,
                        'Al': 9.76*2.,
                        'Ca': 13.0,
                        'Mg': 12.16,
                        'Fe': 8.64,
                        'Na': 2.54*2.,
                        'O': 53.9*2. + 9.76*3. + 13.0 + 12.16 + 8.64 + 2.54 + 2.} # reduced starting mix + 02

    if buffer == 'EMOD':
        #0.5*(Mg2Si2O6) + MgCO3 - Mg2SiO4 - C  = O2
        HP_hen = burnman.minerals.HGP_2018_ds633.hen()
        HP_mag = burnman.minerals.HGP_2018_ds633.mag()
        HP_fo = burnman.minerals.HGP_2018_ds633.fo()
        HP_mwd = burnman.minerals.HGP_2018_ds633.mwd()
        HP_mrw = burnman.minerals.HGP_2018_ds633.mrw()
        HP_diam = burnman.minerals.HGP_2018_ds633.diam()

        EMOD_O2 = burnman.CombinedMineral([HP_hen, HP_mag, HP_fo, HP_diam],
                                          [0.5, 1., -1., -1.])
        EMWD_O2 = burnman.CombinedMineral([HP_hen, HP_mag, HP_mwd, HP_diam],
                                          [0.5, 1., -1., -1.])
        EMRD_O2 = burnman.CombinedMineral([HP_hen, HP_mag, HP_mrw, HP_diam],
                                          [0.5, 1., -1., -1.])
        combined_buffer = burnman.CombinedMineral([HP_hen, HP_mag, HP_fo, HP_diam],
                                                  [0.5, 1., -1., -1.],
                                                  [0., -n_ln_units*burnman.constants.gas_constant,
                                                   0.])
    elif buffer == 'Re-ReO2':
        combined_buffer = burnman.CombinedMineral([endmembers['ReO2'],
                                                   endmembers['Re']],
                                                  [1., -1.],
                                                  [0., -n_ln_units*burnman.constants.gas_constant,
                                                   0.])
    else:
        raise Exception('buffer not recognised')




    P_min = 6.e9
    P0 = 13.e9
    T0 = 1750.

    gt.guess = np.array([0.6, 0.2, 0.15, 0.03, 0.01, 0.01])
    cpx = cpx_od
    cpx.guess = np.array([0.5, 0.1, 0.05, 0.05, 0.05, 0.2, 0.05])

    composition = MORB_composition_plus_oxygen
    assemblage = burnman.Composite([gt, cpx, stv, combined_buffer])
    equality_constraints = [('P', P0), ('T', T0)]

    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   store_iterates=False)

    equality_constraints = [('T', T0), ('phase_proportion', (cpx, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)
    P_cpx_out = sol.x[0]


    pressures = np.linspace(P_min, (P_cpx_out+P_min)/3., 21) # hard to find solution for whole range
    assemblage = burnman.Composite([cpx, stv, combined_buffer, gt])
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_gt_cpx, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                           store_assemblage=True,
                                           store_iterates=False)

    pressures = np.linspace(P_cpx_out, 20.e9, 21)
    assemblage = burnman.Composite([stv, combined_buffer, gt])
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_gt, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                       store_assemblage=True,
                                       store_iterates=False)


    # Plotting
    pressures = []
    Fe3 = []
    FeT = []
    x_dmaj = []
    x_nmaj = []
    for sols in [sols_gt_cpx, sols_gt]:
        pressures.extend([sol.assemblage.pressure
                          for sol in sols if sol.success])
        c_gt = np.array([sol.assemblage.phases[-1].molar_fractions
                         for sol in sols if sol.success])
        Fe3.extend([c[3]*2./(c[1]*3. + c[3]*2.) for c in c_gt])
        FeT.extend([(c[1]*3. + c[3]*2.) for c in c_gt])
        x_dmaj.extend([c[4] for c in c_gt])
        x_nmaj.extend([c[5] for c in c_gt])


    pressures = np.array(pressures)

    plt.style.use('ggplot')
    plt.plot(pressures/1.e9, x_dmaj, label='p(Mg$_3$(MgSi)Si$_3$O$_{{12}}$)')
    plt.plot(pressures/1.e9, x_nmaj, label='p(NaMg$_2$(AlSi)Si$_3$O$_{{12}}$)')
    plt.plot(pressures/1.e9, FeT, label='Fe atoms per 12 O')
    plt.plot(pressures/1.e9, Fe3, label='Fe$^{{3+}}$/(Fe$^{{2+}}$ + Fe$^{{3+}}$)')
    plt.legend()

    for P in [P_cpx_out]:
        plt.plot(np.array([P, P])/1.e9, [0., 1.], color='k', linestyle=':')

    field_labels = [[P_min, P_cpx_out, 'cpx+gt+stv'],
                    [P_cpx_out, 20.e9, 'gt+stv']]

    for P0, P1, lbl in field_labels:
        plt.text((P0+P1)/2./1.e9, 0.5, lbl,
                 horizontalalignment='center',
                 verticalalignment='center')

    plt.title('Garnet in MORB at {0} K'.format(T0))
    plt.xlabel('P (GPa)')
    plt.ylabel('composition')
    plt.xlim(P_min/1.e9,20.)
    plt.ylim(0.,1.)
    plt.savefig('MORB_gt_{0}_plus_{1}_log_units.pdf'.format(buffer, n_log_units))
    plt.show()
