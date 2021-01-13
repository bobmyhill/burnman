import numpy as np

from burnman.processanalyses import store_composition
from burnman.processanalyses import AnalysedComposite
from burnman.processanalyses import compute_and_store_phase_compositions
from global_constants import midpoint_proportion
from global_constants import constrain_endmembers
from global_constants import proportion_cutoff


def get_assemblages(mineral_dataset):
    endmembers = mineral_dataset['endmembers']
    solutions = mineral_dataset['solutions']

    # Frost fper-ol-wad-rw partitioning data
    with open('data/Frost_2003_CFMASO_garnet_analyses.dat', 'r') as f:
        ds = [line.split() for line in f
              if line.split() != [] and line[0] != '#']

    print('WARNING: Frost 2003 CFMASO garnets have shaky Fe3+ (mostly low, large uncertainties)')

    set_runs = list(set([d[0] for d in ds]))
    all_chambers = [list(set([d[1] for d in ds if d[0] == run]))
                    for run in set_runs]

    Frost_2003_CFMASO_gt_assemblages = []
    for i, run in enumerate(set_runs):
        run_indices = [idx for idx, x in enumerate(ds) if x[0] == run]
        pressure = float(ds[run_indices[0]][2])*1.e9
        temperature = float(ds[run_indices[0]][3])

        for j, chamber in enumerate(all_chambers[i]):
            chamber_indices = [run_idx for run_idx in run_indices
                               if (ds[run_idx][1] == chamber)]

            # gt, mw, ol, ring, wad
            # all chambers are in equilibrium with fcc_iron
            phases = [endmembers['fcc_iron']]
            for idx in chamber_indices:
                phase_name = ds[idx][4]

                if phase_name in ['gt', 'mw', 'ol', 'wad']:
                    phases.append(solutions[phase_name])
                elif phase_name == 'ring':
                    phases.append(solutions['sp'])
                else:
                    raise Exception('phase name not recognised')

            assemblage = AnalysedComposite(phases)

            assemblage.experiment_id = 'Frost_2003_{0}'.format(run)
            assemblage.nominal_state = np.array([pressure, temperature])
            assemblage.state_covariances = np.array([[1.e7*1.e7, 0.], [0., 100.]]) # small uncertainty for P, hyperparams for true uncertainty

            n_phases = len(assemblage.phases)
            fitted_elements = ['Fe', 'Al', 'Ca', 'Si', 'Mg', 'Na', 'Fef_B']
            for k in range(1, n_phases):  # fcc_iron is the only pure phase
                phase = assemblage.phases[k]
                d = ds[chamber_indices[k-1]]
                c = np.array(list(map(float,
                                      [d[5], d[7], d[9], d[11], d[13],
                                       0., 0.])))

                sig_c = np.array([max(s, 0.001)
                                  for s in map(float, [d[6], d[8], d[10],
                                                       d[12], d[14],
                                                       1.e-6, 1.e-6])])

                store_composition(phase, fitted_elements, c, sig_c)

            # Tweak compositions with 0.1% of a midpoint proportion
            # Do not consider (transformed) endmembers with < 5% abundance
            # in the solid solution. Copy the stored compositions from
            # each phase to the assemblage storage.
            assemblage.set_state(*assemblage.nominal_state)
            compute_and_store_phase_compositions(assemblage,
                                                 midpoint_proportion,
                                                 constrain_endmembers,
                                                 proportion_cutoff,
                                                 copy_storage=True)

            Frost_2003_CFMASO_gt_assemblages.append(assemblage)
    return Frost_2003_CFMASO_gt_assemblages
