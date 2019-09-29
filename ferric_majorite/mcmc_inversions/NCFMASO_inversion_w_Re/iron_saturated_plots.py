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


def iron_saturated_KLB_plot(dataset, storage):
    # KLB-1 (Takahashi, 1986; Walter, 1998; Holland et al., 2013)
    KLB_1_composition_Fe_saturated = {'Si': 39.4,
                                      'Al': 2.*2.,
                                      'Ca': 3.3,
                                      'Mg': 49.5,
                                      'Fe': 5.2 + 5.,
                                      'Na': 0.26*2.,
                                      'O': 39.4*2. + 2.*3. + 3.3 + 49.5 + 5.2 + 0.26} # reduced starting mix + Fe

    endmembers = dataset['endmembers']
    solutions = dataset['solutions']
    child_solutions = dataset['child_solutions']

    # Alias solutions:
    ol = solutions['ol']
    wad = solutions['wad']
    rw = child_solutions['ring']
    gt = solutions['gt']
    cpx_od = solutions['cpx']
    iron = endmembers['fcc_iron']

    ol.guess = np.array([0.9, 0.1])
    wad.guess = np.array([0.87, 0.13])
    rw.guess = np.array([0.84, 0.16])
    gt.guess = np.array([0.6, 0.2, 0.15, 0.03, 0.01, 0.01])
    cpx_od.guess = np.array([0.7, 0.1, 0.05, 0.02, 0.05, 0.03, 0.05])

    P0 = 13.e9
    T0 = 1750.

    composition = KLB_1_composition_Fe_saturated
    assemblage = burnman.Composite([ol, gt, cpx_od, iron])
    equality_constraints = [('P', P0), ('T', T0)]

    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   store_iterates=False)

    fs = assemblage.molar_fractions
    fs.append(0.)
    n = assemblage.n_moles
    wad.set_composition(wad.guess)
    assemblage = burnman.Composite([ol, gt, cpx_od, iron, wad], fs)
    assemblage.n_moles = n
    assemblage.set_state(sol.x[0], sol.x[1])
    equality_constraints = [('T', T0), ('phase_proportion', (wad, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)

    P_wad_in = assemblage.pressure


    equality_constraints = [('T', T0), ('phase_proportion', (ol, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)
    P_ol_out = assemblage.pressure

    assemblage = burnman.Composite([wad, cpx_od, iron, gt])
    equality_constraints = [('T', T0), ('phase_proportion', (cpx_od, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   store_assemblage=True,
                                   store_iterates=False)
    P_cpx_out = assemblage.pressure

    P0 = 16.e9
    composition = KLB_1_composition_Fe_saturated
    assemblage = burnman.Composite([wad, gt, iron])
    equality_constraints = [('P', P0), ('T', T0)]

    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   store_iterates=False)

    rw.set_composition(rw.guess)
    fs = assemblage.molar_fractions
    fs.append(0.)
    n = assemblage.n_moles
    assemblage = burnman.Composite([wad, gt, iron, rw], fs)
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


    # ol-cpx-gt-iron

    assemblage = burnman.Composite([ol, cpx_od, iron, gt])
    pressures = np.linspace(10.e9, P_wad_in, 21)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_ol_cpx, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                           store_assemblage=True,
                                           store_iterates=False)


    assemblage = burnman.Composite([wad, cpx_od, iron, gt])
    pressures = np.linspace(P_ol_out, P_ol_out+0.1e9, 2)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_wad_cpx, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                            store_assemblage=True,
                                            store_iterates=False)


    assemblage = burnman.Composite([wad, iron, gt])
    pressures = np.linspace(P_cpx_out, P_rw_in, 21)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_wad, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                        store_assemblage=True,
                                        store_iterates=False)



    assemblage = burnman.Composite([rw, iron, gt])
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
    for sols in [sols_ol_cpx, sols_wad_cpx, sols_wad, sols_rw]:
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

    for P in [P_wad_in, P_ol_out, P_cpx_out, P_rw_in, P_wad_out]:
        plt.plot(np.array([P, P])/1.e9, [0., 0.6], color='k', linestyle=':')

    field_labels = [[10.e9, P_wad_in, 'cpx+ol+gt'],
                    [P_ol_out, P_cpx_out, 'cpx+wad+gt'],
                    [P_cpx_out, P_rw_in, 'wad+gt'],
                    [P_wad_out, 20.e9, 'rw+gt']]

    for P0, P1, lbl in field_labels:
        plt.text((P0+P1)/2./1.e9, 0.3, lbl,
                 horizontalalignment='center',
                 verticalalignment='center')

    plt.title('Garnet in iron-saturated KLB-1 peridotite at {0} K'.format(T0))
    plt.xlabel('P (GPa)')
    plt.ylabel('composition')
    plt.xlim(10.,20.)
    plt.ylim(0.,0.6)
    plt.savefig('KLB-1_gt_Fe_saturated.pdf')
    plt.show()


def iron_saturated_MORB_plot(dataset, storage):

    # MORB (Litasov et al., 2005)
    MORB_composition_Fe_saturated = {'Si': 53.9,
                                     'Al': 9.76*2.,
                                     'Ca': 13.0,
                                     'Mg': 12.16,
                                     'Fe': 8.64 + 5.,
                                     'Na': 2.54*2.,
                                     'O': 53.9*2. + 9.76*3. + 13.0 + 12.16 + 8.64 + 2.54} # reduced starting mix + Fe

    endmembers = dataset['endmembers']
    solutions = dataset['solutions']
    child_solutions = dataset['child_solutions']

    # Alias solutions:
    ol = solutions['ol']
    wad = solutions['wad']
    rw = child_solutions['ring']
    gt = solutions['gt']
    cpx_od = solutions['cpx']
    fcc_iron = endmembers['fcc_iron']
    stv = endmembers['stv']

    gt.guess = np.array([0.6, 0.2, 0.15, 0.03, 0.01, 0.01])

    cpx = cpx_od # child_solutions['nocfs_cpx']

    cpx.guess = np.array([0.5, 0.1, 0.05, 0.05, 0.05, 0.2, 0.05])
    #cpx.guess = np.array([0.5, 0.1, 0.05, 0.05, 0.2, 0.05])

    P0 = 10.e9
    T0 = 1750.
    composition = MORB_composition_Fe_saturated
    assemblage = burnman.Composite([gt, cpx, stv, fcc_iron])
    equality_constraints = [('P', P0), ('T', T0)]

    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   store_iterates=False)

    equality_constraints = [('T', T0), ('phase_proportion', (cpx, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)
    P_cpx_out = sol.x[0]


    pressures = np.linspace(10.e9, (P_cpx_out+10.e9)/3., 21) # hard to find solution for whole range
    assemblage = burnman.Composite([cpx, stv, fcc_iron, gt])
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_gt_cpx, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                           store_assemblage=True,
                                           store_iterates=False)

    pressures = np.linspace(P_cpx_out, 20.e9, 21)
    assemblage = burnman.Composite([stv, fcc_iron, gt])
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

    field_labels = [[10.e9, P_cpx_out, 'cpx+gt+stv'],
                    [P_cpx_out, 20.e9, 'gt+stv']]

    for P0, P1, lbl in field_labels:
        plt.text((P0+P1)/2./1.e9, 0.5, lbl,
                 horizontalalignment='center',
                 verticalalignment='center')

    plt.title('Garnet in iron-saturated MORB at {0} K'.format(T0))
    plt.xlabel('P (GPa)')
    plt.ylabel('composition')
    plt.xlim(10.,20.)
    plt.ylim(0.,1.)
    plt.savefig('MORB_gt_Fe_saturated.pdf')
    plt.show()



def iron_saturated_mars_plot(dataset, storage):

    mars_DW1985 = burnman.Composition({'Na2O': 0.5,
                                       'CaO': 2.46,
                                       'FeO': 18.47,
                                       'MgO': 30.37,
                                       'Al2O3': 3.55,
                                       'SiO2': 44.65,
                                       'Fe': 10.}, 'weight')
    mars_DW1985.renormalize('atomic', 'total', 100.)
    mars_composition = dict(mars_DW1985.atomic_composition)

    endmembers = dataset['endmembers']
    solutions = dataset['solutions']
    child_solutions = dataset['child_solutions']

    # Alias solutions:
    ol = solutions['ol']
    wad = solutions['wad']
    rw = child_solutions['ring']
    gt = solutions['gt']
    cpx_od = solutions['cpx']
    fcc_iron = endmembers['fcc_iron']
    stv = endmembers['stv']

    # KLB-1 first
    ol.guess = np.array([0.9, 0.1])
    wad.guess = np.array([0.87, 0.13])
    rw.guess = np.array([0.84, 0.16])
    gt.guess = np.array([0.6, 0.2, 0.15, 0.03, 0.01, 0.01])
    cpx_od.guess = np.array([0.7, 0.1, 0.05, 0.02, 0.05, 0.03, 0.05])


    P0 = 13.e9
    composition = mars_composition
    assemblage = burnman.Composite([ol, gt, cpx_od, fcc_iron])
    equality_constraints = [('P', P0), ('T', T0)]

    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   store_iterates=False)

    fs = assemblage.molar_fractions
    fs.append(0.)
    n = assemblage.n_moles
    wad.set_composition(wad.guess)
    assemblage = burnman.Composite([ol, gt, cpx_od, fcc_iron, wad], fs)
    assemblage.n_moles = n
    assemblage.set_state(sol.x[0], sol.x[1])
    equality_constraints = [('T', T0), ('phase_proportion', (wad, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)

    P_wad_in = assemblage.pressure


    equality_constraints = [('T', T0), ('phase_proportion', (ol, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)
    P_ol_out = assemblage.pressure

    assemblage = burnman.Composite([wad, cpx_od, fcc_iron, gt])
    equality_constraints = [('T', T0), ('phase_proportion', (cpx_od, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   store_assemblage=True,
                                   store_iterates=False)
    P_cpx_out = assemblage.pressure

    P0 = 16.e9
    composition = KLB_1_composition_Fe_saturated
    assemblage = burnman.Composite([wad, gt, fcc_iron])
    equality_constraints = [('P', P0), ('T', T0)]

    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   store_iterates=False)

    rw.set_composition(rw.guess)
    fs = assemblage.molar_fractions
    fs.append(0.)
    n = assemblage.n_moles
    assemblage = burnman.Composite([wad, gt, fcc_iron, rw], fs)
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


    # ol-cpx-gt-iron

    assemblage = burnman.Composite([ol, cpx_od, fcc_iron, gt])
    pressures = np.linspace(10.e9, P_wad_in, 21)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_ol_cpx, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                           store_assemblage=True,
                                           store_iterates=False)


    assemblage = burnman.Composite([wad, cpx_od, fcc_iron, gt])
    pressures = np.linspace(P_ol_out, P_ol_out+0.1e9, 2)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_wad_cpx, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                            store_assemblage=True,
                                            store_iterates=False)


    assemblage = burnman.Composite([wad, fcc_iron, gt])
    pressures = np.linspace(P_cpx_out, P_rw_in, 21)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_wad, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                        store_assemblage=True,
                                        store_iterates=False)



    assemblage = burnman.Composite([rw, fcc_iron, gt])
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
    for sols in [sols_ol_cpx, sols_wad_cpx, sols_wad, sols_rw]:
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

    for P in [P_wad_in, P_ol_out, P_cpx_out, P_rw_in, P_wad_out]:
        plt.plot(np.array([P, P])/1.e9, [0., 0.6], color='k', linestyle=':')

    field_labels = [[10.e9, P_wad_in, 'cpx+ol+gt'],
                    [P_ol_out, P_cpx_out, 'cpx+wad+gt'],
                    [P_cpx_out, P_rw_in, 'wad+gt'],
                    [P_wad_out, 20.e9, 'rw+gt']]

    for P0, P1, lbl in field_labels:
        plt.text((P0+P1)/2./1.e9, 0.3, lbl,
                 horizontalalignment='center',
                 verticalalignment='center')

    plt.title('Garnet in iron-saturated Martian peridotite at {0} K'.format(T0))
    plt.xlabel('P (GPa)')
    plt.ylabel('composition')
    plt.xlim(10.,20.)
    plt.ylim(0.,0.6)
    plt.savefig('mars_gt_Fe_saturated.pdf')
    plt.show()
