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
    with open('data/Klemme_ONeill_2000_CMAS_opx_cpx_gt_ol_sp.dat', 'r') as f:
        expt_data = [line.split() for line in f if line.split() != []
                     and line[0] != '#']

    set_runs = list(set([d[1] for d in expt_data]))

    Klemme_ONeill_2000_CMAS_assemblages = []

    for i, run in enumerate(set_runs):

        run_indices = [idx for idx, d in enumerate(expt_data) if d[1] == run]
        n_phases = len(run_indices)
        phases = []
        for datum in [expt_data[idx] for idx in run_indices]:
            master_id, run_id = datum[0:2]
            TC, Pkbar, t = list(map(float, datum[2:5]))
            phase_name = datum[5]

            if phase_name in ['fo', 'sp']:
                phases.append(endmembers[phase_name])
            elif phase_name in ['opx', 'cpx', 'gt']:
                phases.append(solutions[phase_name])

                # composition = Si	Sierr	Al	Alerr	Mg	Mgerr	Ca	Caerr
                Si, Sierr, Al, Alerr = list(map(float, datum[6:10]))
                Mg, Mgerr, Ca, Caerr = list(map(float, datum[10:]))
                Caerr = max([0.01, Caerr, 0.02*Ca])
                Mgerr = max([0.01, Mgerr, 0.02*Mg])
                Alerr = max([0.01, Alerr, 0.02*Al])
                Sierr = max([0.01, Sierr, 0.02*Si])
                store_composition(phases[-1],
                                  ['Ca', 'Mg', 'Al', 'Si',
                                   'Na', 'Fe', 'Fe_B', 'Fef_B'],
                                  np.array([Ca, Mg, Al, Si,
                                            0., 0., 0., 0.]),
                                  np.array([Caerr, Mgerr, Alerr, Sierr,
                                            1.e-5, 1.e-5, 1.e-5, 1.e-5]))

            else:
                raise Exception('phase not recognised')

        assemblage = AnalysedComposite(phases)

        # Convert P from kbar to Pa, T from C to K
        assemblage.experiment_id = 'Klemme_ONeill_2000_CMAS_{0}'.format(run_id)
        assemblage.nominal_state = np.array([Pkbar*1.e8,
                                             TC+273.15])
        assemblage.state_covariances = np.array([[1.e8*1.e8, 0.], [0., 100.]])

        # Tweak compositions with 0.1% of a midpoint proportion
        # Do not consider (transformed) endmembers with < 5% abundance
        # in the solid solution. Copy the stored compositions from
        # each phase to the assemblage storage.
        compute_and_store_phase_compositions(assemblage,
                                             midpoint_proportion,
                                             constrain_endmembers,
                                             proportion_cutoff,
                                             copy_storage=True)

        Klemme_ONeill_2000_CMAS_assemblages.append(assemblage)

    return Klemme_ONeill_2000_CMAS_assemblages
