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

    # NCMAS garnet-pyroxene partitioning data
    with open('data/Gasparik_1989_NCMAS_px_gt.dat', 'r') as f:
        px_gt_data = [line.split() for line in f
                      if line.split() != [] and line[0] != '#']

    set_runs = list(set([d[0] for d in px_gt_data]))

    Gasparik_1989_NCMAS_assemblages = []

    for i, run in enumerate(set_runs):
        run_indices = [idx for idx, x in enumerate(px_gt_data) if x[0] == run]

        phases = []
        for run_idx in run_indices:
            run_id, mix, t = px_gt_data[run_idx][:3]
            ramP, Pkbar, TC, phase_name, N = px_gt_data[run_idx][3:8]
            Na, Ca, Mg, Al, Si, cation_sum = px_gt_data[run_idx][8:]

            Na = float(Na)
            Ca = float(Ca)
            Mg = float(Mg)
            Al = float(Al)
            Si = float(Si)

            # opx/hpx has quite a lot of sodium in it, we haven't added ojd.
            # just use gt-cpx-cpv equilibria
            if phase_name in ['gt', 'cpx']:
                phases.append(solutions[phase_name])

                store_composition(phases[-1],
                                  ['Na', 'Ca', 'Mg', 'Al', 'Si', 'Fe', 'Fef_B'],
                                  np.array([Na, Ca, Mg, Al, Si, 0., 0.]),
                                  np.array([0.02, 0.02, 0.02, 0.02, 0.02,
                                            1.e-5, 1.e-5]))
            elif phase_name == 'cpv':
                phases.append(endmembers['cpv'])
            else:
                raise Exception(f'{phase_name} not recognised')

        if len(phases) >= 2:
            assemblage = AnalysedComposite(phases)
            assemblage.experiment_id = 'Gasparik_1989_NMAS_{0}'.format(run_id)
            # Convert P from kbar to Pa, T from C to K
            assemblage.nominal_state = np.array([float(Pkbar)*1.e8,
                                                 float(TC)+273.15])
            assemblage.state_covariances = np.array([[5.e8*5.e8, 0.],
                                                     [0., 100.]])

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

            Gasparik_1989_NCMAS_assemblages.append(assemblage)

    return Gasparik_1989_NCMAS_assemblages
