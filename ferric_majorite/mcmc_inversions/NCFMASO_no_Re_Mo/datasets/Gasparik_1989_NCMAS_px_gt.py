import numpy as np
import burnman

def get_assemblages(mineral_dataset):
    endmembers = mineral_dataset['endmembers']
    solutions = mineral_dataset['solutions']
    child_solutions = mineral_dataset['child_solutions']


    # NCMAS garnet-pyroxene partitioning data

    with open('data/Gasparik_1989_NCMAS_px_gt.dat', 'r') as f:
        px_gt_data = [line.split() for line in f if line.split() != [] and line[0] != '#']

    all_runs = [d[0] for d in px_gt_data]
    set_runs = list(set([d[0] for d in px_gt_data]))


    Gasparik_1989_NCMAS_assemblages = []

    for i, run in enumerate(set_runs):
        run_indices = [idx for idx, x in enumerate(px_gt_data) if x[0] == run]

        phases = []
        for run_idx in run_indices:
            run_id, mix, t, ramP, Pkbar, TC, phase, N, Na, Ca, Mg, Al, Si, cation_sum = px_gt_data[run_idx]

            # opx/hpx has quite a lot of sodium in it, and we haven't yet added ojd.
            # for now, just use gt-cpx equilibria (and there's one sample with fo)
            if phase == 'gt':
                if run_id == '515' or run_id == '264':
                    phases.append(child_solutions['NCMAS_gt'])
                else:
                    phases.append(child_solutions['py_gr_nmaj_gt'])

                phases[-1].fitted_elements = ['Na', 'Ca', 'Mg', 'Al', 'Si']
                phases[-1].composition = np.array([float(Na), float(Ca), float(Mg), float(Al), float(Si)])
                phases[-1].compositional_uncertainties = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
            elif phase == 'cpx':
                phases.append(child_solutions['di_jd']) # no cats component stable, v. little cen
                phases[-1].fitted_elements = ['Na', 'Ca', 'Mg', 'Al', 'Si']
                phases[-1].composition = np.array([float(Na), float(Ca), float(Mg), float(Al), float(Si)])
                phases[-1].compositional_uncertainties = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
            elif phase == 'cpv':
                phases.append(endmembers['cpv'])

        if len(phases) >= 2:
            assemblage = burnman.Composite(phases)
            assemblage.experiment_id = 'Gasparik_1989_NMAS_{0}'.format(run_id)
            assemblage.nominal_state = np.array([float(Pkbar)*1.e8,
                                                 float(TC)+273.15]) # CONVERT P TO PA, T to K
            assemblage.state_covariances = np.array([[5.e8*5.e8, 0.], [0., 100.]])

            burnman.processanalyses.compute_and_set_phase_compositions(assemblage)

            """
            try:
                print(phases[0].name, run_id, phases[0].molar_fractions)
            except:
                pass
            try:
                print(phases[1].name, run_id, phases[1].molar_fractions)
            except:
                pass
            """

            assemblage.stored_compositions = ['composition not assigned']*len(assemblage.phases)
            for k in range(len(phases)):
                try:
                    assemblage.stored_compositions[k] = (assemblage.phases[k].molar_fractions,
                                                         assemblage.phases[k].molar_fraction_covariances)
                except:
                    pass

            Gasparik_1989_NCMAS_assemblages.append(assemblage)

    return Gasparik_1989_NCMAS_assemblages
