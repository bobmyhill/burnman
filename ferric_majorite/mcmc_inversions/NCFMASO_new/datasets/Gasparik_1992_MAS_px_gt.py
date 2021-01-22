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

    # Garnet-pyroxene partitioning data
    with open('data/Gasparik_1992_MAS_px_gt.dat', 'r') as f:
        px_gt_data = [line.split() for line in f
                      if line.split() != [] and line[0] != '#']

    set_runs = list(set([d[0] for d in px_gt_data]))

    Gasparik_1992_MAS_assemblages = []

    for i, run in enumerate(set_runs):
        run_indices = [idx for idx, x in enumerate(px_gt_data) if x[0] == run]

        phases = []
        for run_idx in run_indices:
            run_id, mix, t = px_gt_data[run_idx][:3]
            ramP, Pkbar, TC, phase_name, N = px_gt_data[run_idx][3:8]
            Mg, Al, Si, cation_sum = px_gt_data[run_idx][8:]

            Mg = float(Mg)
            Al = float(Al)
            Si = float(Si)

            if phase_name in ['opx', 'gt']:
                phases.append(solutions[phase_name])

                store_composition(phases[-1],
                                  ['Mg', 'Al', 'Si', 'Na', 'Ca', 'Fe', 'Fe_B', 'O'],
                                  np.array([Mg, Al, Si, 0., 0., 0., 0., 6.]),
                                  np.array([0.02, 0.02, 0.02, 1.e-6, 1.e-6,
                                            1.e-6, 1.e-6, 0.02]))

            elif phase_name == 'hpx':
                phases.append(endmembers['hen'])  # within 0.5% of pure hen

        assemblage = AnalysedComposite(phases)
        assemblage.experiment_id = 'Gasparik_1992_MAS_{0}'.format(run_id)
        # Convert P from kbar to Pa, T from C to K
        assemblage.nominal_state = np.array([float(Pkbar)*1.e8,
                                             float(TC)+273.15])
        assemblage.state_covariances = np.array([[5.e8*5.e8, 0.],
                                                 [0., 100.]])

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

        Gasparik_1992_MAS_assemblages.append(assemblage)

    return Gasparik_1992_MAS_assemblages