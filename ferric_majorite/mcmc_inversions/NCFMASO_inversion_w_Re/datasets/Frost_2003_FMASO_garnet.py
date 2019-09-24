import numpy as np
import burnman

def get_assemblages(mineral_dataset):
    endmembers = mineral_dataset['endmembers']
    solutions = mineral_dataset['solutions']
    child_solutions = mineral_dataset['child_solutions']


    # Frost fper-ol-wad-rw partitioning data
    with open('data/Frost_2003_FMASO_garnet_analyses.dat', 'r') as f:
        ds = [line.split() for line in f if line.split() != [] and line[0] != '#']

    all_runs = [d[0] for d in ds]
    set_runs = list(set([d[0] for d in ds]))
    all_conditions = [(float(ds[all_runs.index(run)][2])*1.e9,
                   float(ds[all_runs.index(run)][3])) for run in set_runs]
    all_chambers = [list(set([d[1] for d in ds if d[0] == run])) for run in set_runs]

    Frost_2003_FMASO_gt_assemblages = []
    for i, run in enumerate(set_runs):
        run_indices = [idx for idx, x in enumerate(ds) if x[0] == run]
        pressure = float(ds[run_indices[0]][2])*1.e9
        temperature = float(ds[run_indices[0]][3])

        for j, chamber in enumerate(all_chambers[i]):
            chamber_indices = [run_idx for run_idx in run_indices
                               if (ds[run_idx][1] == chamber)]

            # gt, mw, ol, ring, wad

            phases = [endmembers['fcc_iron']] # all chambers are in equilibrium with fcc_iron
            for idx in chamber_indices:
                phase_name = ds[idx][4]
                if phase_name == 'gt':
                    if run == 'V170' and chamber == 'B':
                        phases.append(child_solutions['FMAS_gt']) # apparently very low Fe3+
                    elif float(ds[idx][9]) < 3.05:
                        phases.append(child_solutions['lp_FMASO_gt'])
                    else:
                        phases.append(child_solutions['FMASO_gt'])
                else:
                    try:
                        phases.append(solutions[phase_name])
                    except:
                        phases.append(child_solutions[phase_name])

            assemblage = burnman.Composite(phases)

            assemblage.experiment_id = 'Frost_2003_{0}'.format(run)
            assemblage.nominal_state = np.array([pressure, temperature])
            assemblage.state_covariances = np.array([[1.e7*1.e7, 0.], [0., 100.]]) # small uncertainty for P, hyperparams for true uncertainty

            n_phases = len(assemblage.phases)

            for k in range(1, n_phases): # fcc_iron is the only pure phase
                phase = assemblage.phases[k]
                d = ds[chamber_indices[k-1]]
                c = np.array(list(map(float, [d[5], d[7], d[9], d[11]])))
                sig_c = np.array([max(s, 0.01) for s in map(float, [d[6], d[8], d[10], d[12]])])

                sig_c = sig_c*4. # make uncertainties larger

                # Assign elements and uncertainties
                phase.fitted_elements = ['Fe', 'Al', 'Si', 'Mg']
                phase.composition = c
                phase.compositional_uncertainties = sig_c

            burnman.processanalyses.compute_and_set_phase_compositions(assemblage)
            """
            for phase in assemblage.phases:
                if phase == child_solutions['FMASO_gt'] or phase == child_solutions['lp_FMASO_gt']:
                    print(pressure, (phase.molar_fractions[2]*2.)/(phase.molar_fractions[2]*2. + phase.molar_fractions[1]*3.))
                    print(np.sqrt(phase.molar_fractions[1:3]))
                    print(np.sqrt(phase.molar_fraction_covariances[1][1]))
                    print(np.sqrt(phase.molar_fraction_covariances[2][2]))
                    #phase.molar_fraction_covariances[2][2]
            """

            assemblage.stored_compositions = ['composition not assigned']*n_phases
            assemblage.stored_compositions[1:] = [(assemblage.phases[k].molar_fractions,
                                                   assemblage.phases[k].molar_fraction_covariances)
                                                  for k in range(1, n_phases)]


            Frost_2003_FMASO_gt_assemblages.append(assemblage)

    return Frost_2003_FMASO_gt_assemblages
