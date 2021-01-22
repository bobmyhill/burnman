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

    # garnet-bridgmanite-corundum data
    with open('data/Liu_et_al_2016_gt_bdg_cor.dat', 'r') as f:
        gt_bdg_cor_data = [line.split() for line in f
                           if line.split() != [] and line[0] != '#']

    set_runs = list(set([d[0] for d in gt_bdg_cor_data]))

    Liu_et_al_2016_MAS_assemblages = []

    for i, run in enumerate(set_runs):
        run_indices = [idx for idx, x in enumerate(gt_bdg_cor_data)
                       if x[0] == run]

        phases = []
        for run_idx in run_indices:
            run_id, PGPa, TK, phase_name, N = gt_bdg_cor_data[run_idx][:5]
            c = gt_bdg_cor_data[run_idx][5:]
            Mg, Mgerr, Al, Alerr, Si, Sierr = list(map(float, c))

            pressure = float(PGPa)*1.e9
            temperature = float(TK)

            if phase_name == 'gt':
                phases.append(endmembers['py'])  # no excess Si for any run
            elif phase_name in ['bdg', 'cor']:
                phases.append(solutions[phase_name])
                store_composition(solutions[phase_name],
                                  ['Mg', 'Al', 'Si', 'Fe', 'Fef_B'],
                                  np.array([Mg, Al, Si, 0., 0.]),
                                  np.array([Mgerr, Alerr, Sierr,
                                            1.e-5, 1.e-5]))

        assemblage = AnalysedComposite(phases)
        assemblage.experiment_id = 'Liu_2016_{0}'.format(run_id)
        assemblage.nominal_state = np.array([pressure, temperature])
        assemblage.state_covariances = np.array([[5.e8*5.e8, 0.], [0., 100.]])

        # Tweak compositions with a proportion of a midpoint composition
        # Do not consider (transformed) endmembers with < proportion_cutoff
        # abundance in the solid solution. Copy the stored
        # compositions from each phase to the assemblage storage.
        assemblage.set_state(*assemblage.nominal_state)
        compute_and_store_phase_compositions(assemblage,
                                             midpoint_proportion,
                                             constrain_endmembers,
                                             proportion_cutoff,
                                             copy_storage=True)

        Liu_et_al_2016_MAS_assemblages.append(assemblage)

    return Liu_et_al_2016_MAS_assemblages
