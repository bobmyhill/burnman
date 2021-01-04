import numpy as np

from burnman.processanalyses import store_composition
from burnman.processanalyses import AnalysedComposite
from burnman.processanalyses import compute_and_store_phase_compositions
from global_constants import midpoint_proportion
from global_constants import constrain_endmembers
from global_constants import proportion_cutoff


def get_assemblages(mineral_dataset):
    solutions = mineral_dataset['solutions']

    # Garnet-olivine partitioning data
    ol_gt_data = np.loadtxt('data/ONeill_Wood_1979_CFMAS_ol_gt.dat')

    ONeill_Wood_1979_CFMAS_assemblages = []
    run_id = 0
    for datum in ol_gt_data:
        PGPa, TK = datum[:2]
        xMgol, dxMgol, xFeol, dxFeol = datum[2:6]
        xMggt, dxMggt, xFegt, dxFegt, xCagt, dxCagt = datum[6:]

        run_id += 1

        assemblage = AnalysedComposite([solutions['ol'],
                                        solutions['gt']])

        # Convert pressure to Pa
        # Assign 0.5 kbar pressure error
        assemblage.experiment_id = 'ONeill_Wood_1979_CFMAS_{0}'.format(run_id)
        assemblage.nominal_state = np.array([PGPa*1.e9, TK])
        assemblage.state_covariances = np.array([[5.e7*5.e7, 0.], [0., 100.]])

        store_composition(solutions['ol'],
                          ['Mg', 'Fe'],
                          np.array([xMgol, xFeol]),
                          np.array([dxMgol, dxFeol]))

        store_composition(solutions['gt'],
                          ['Mg', 'Fe', 'Ca',
                           'Al', 'Si', 'Na', 'Fef_B', 'Mg_B'],
                          np.array([xMggt, xFegt, xCagt,
                                    2./3., 1., 0., 0., 0.]),
                          np.array([dxMggt, dxFegt, dxCagt,
                                    1.e-5, 1.e-5, 1.e-5, 1.e-5, 1.e-5]))

        # Tweak compositions with 0.1% of a midpoint proportion
        # Do not consider (transformed) endmembers with < 5% abundance
        # in the solid solution. Copy the stored compositions from
        # each phase to the assemblage storage.
        compute_and_store_phase_compositions(assemblage,
                                             midpoint_proportion,
                                             constrain_endmembers,
                                             proportion_cutoff,
                                             copy_storage=True)

        ONeill_Wood_1979_CFMAS_assemblages.append(assemblage)

    return ONeill_Wood_1979_CFMAS_assemblages
