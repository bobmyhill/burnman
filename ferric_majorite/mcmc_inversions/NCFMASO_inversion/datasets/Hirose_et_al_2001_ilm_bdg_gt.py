import numpy as np
import burnman

def get_assemblages(mineral_dataset):
    endmembers = mineral_dataset['endmembers']
    solutions = mineral_dataset['solutions']
    child_solutions = mineral_dataset['child_solutions']

    # garnet-bridgmanite-corundum data
    with open('data/Hirose_et_al_2001_MAS_ilm_gt_bdg.dat', 'r') as f:
        ilm_bdg_gt_data = [line.split() for line in f if line.split() != [] and line[0] != '#']

    delta_pressure = 2.e9 # Adjustment for Anderson -> Jamieson Au pressure scale

    Hirose_et_al_2001_MAS_assemblages = []

    for i, datum in enumerate(ilm_bdg_gt_data):
        phases = []

        PGPa, TC, phase1, phase2, ph1_Al2O3, ph2_Al2O3 = datum

        pressure = float(PGPa)*1.e9 + delta_pressure
        temperature = float(TC) + 273.15

        for (phase, Al) in [(phase1, ph1_Al2O3),
                            (phase2, ph1_Al2O3)]:
            Al = float(Al)
            c = np.array([(1. - Al)/2., Al, (1. - Al)/2.])
            sig_c = np.array([float(0.01), float(0.01), float(0.01)])

            if phase == 'gt':
                phases.append(child_solutions['py_dmaj_gt'])
            elif phase == 'bdg':
                phases.append(child_solutions['mg_al_bdg'])

            phases[-1].fitted_elements = ['Mg', 'Al', 'Si']
            phases[-1].composition = c
            phases[-1].compositional_uncertainties = sig_c

        assemblage = burnman.Composite(phases)
        assemblage.experiment_id = 'Hirose_2001_{0}'.format(i)
        assemblage.nominal_state = np.array([pressure, temperature]) # CONVERT P TO PA, T to K
        assemblage.state_covariances = np.array([[5.e8*5.e8, 0.], [0., 100.]])

        burnman.processanalyses.compute_and_set_phase_compositions(assemblage)

        assemblage.stored_compositions = ['composition not assigned']*len(assemblage.phases)
        for k in range(len(phases)):
            try:
                assemblage.stored_compositions[k] = (assemblage.phases[k].molar_fractions,
                                                     assemblage.phases[k].molar_fraction_covariances)
            except:
                pass

        Hirose_et_al_2001_MAS_assemblages.append(assemblage)

    return Hirose_et_al_2001_MAS_assemblages
