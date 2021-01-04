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

    # NMAS garnet-pyroxene partitioning data
    # Essentially looks at the reaction py + nagt <--> jd + cen
    # i.e. Mg3Al2Si3O12 + Na2MgSi5O12 <--> 2NaAlSi2O6 + 2Mg2Si2O6

    with open('data/Gasparik_1989_NMAS_px_gt.dat', 'r') as f:
        px_gt_data = [line.split() for line in f
                      if line.split() != [] and line[0] != '#']

    set_runs = list(set([d[0] for d in px_gt_data]))

    Gasparik_1989_NMAS_assemblages = []

    for i, run in enumerate(set_runs):
        run_indices = [idx for idx, x in enumerate(px_gt_data) if x[0] == run]

        phases = []
        for run_idx in run_indices:
            run_id, mix, t = px_gt_data[run_idx][:3]
            ramP, Pkbar, TC, phase_name, N = px_gt_data[run_idx][3:8]
            Na, Mg, Al, Si, cation_sum = px_gt_data[run_idx][8:]

            Na = float(Na)
            Mg = float(Mg)
            Al = float(Al)
            Si = float(Si)

            # opx/hpx has quite a lot of sodium in it, and we haven't yet added ojd.
            # for now, just use gt-cpx equilibria (and there's one sample with fo)
            if phase_name in ['gt', 'cpx']:
                phases.append(solutions[phase_name])

                store_composition(phases[-1],
                                  ['Na', 'Mg', 'Al', 'Si', 'Ca', 'Fe', 'Fe_B'],
                                  np.array([Na, Mg, Al, Si, 0., 0., 0.]),
                                  np.array([0.02, 0.02, 0.02, 0.02, 1.e-5,
                                            1.e-5, 1.e-5]))
            elif phase_name == 'fo':
                phases.append(endmembers['fo'])
            elif phase_name == 'hpx' and float(Na) < 0.02 and float(Al) < 0.02:
                phases.append(endmembers['hen'])  # within 0.5% of pure hen

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
            compute_and_store_phase_compositions(assemblage,
                                                 midpoint_proportion,
                                                 constrain_endmembers,
                                                 proportion_cutoff,
                                                 copy_storage=True)

            Gasparik_1989_NMAS_assemblages.append(assemblage)

    return Gasparik_1989_NMAS_assemblages
