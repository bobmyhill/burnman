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

    # bdg-fper-stv partitioning data
    with open('data/Tange_TNFS_2009_bdg_fper_stv.dat', 'r') as f:
        bdg_fper_stv_data = [line.split() for line in f
                             if line.split() != [] and line[0] != '#']

    Tange_TNFS_2009_FMS_assemblages = []

    for i, datum in enumerate(bdg_fper_stv_data):

        run_id = datum[0]
        PGPa, TC, ffpv, sig_fpv, ffper, sig_fper = list(map(float, datum[1:]))

        pressure = float(PGPa)*1.e9
        temperature = float(TC) + 273.15

        assemblage = AnalysedComposite([solutions['bdg'],
                                        solutions['mw'],
                                        endmembers['stv']])

        assemblage.experiment_id = 'Tange_2009_FMS_{0}'.format(run_id)
        assemblage.nominal_state = np.array([pressure, temperature])
        assemblage.state_covariances = np.array([[1.e9*1.e9, 0.], [0., 100.]])

        store_composition(solutions['bdg'],
                          ['Mg', 'Fe', 'Si',
                           'Al', 'Ca', 'Fef_A', 'Fef_B'],
                          np.array([1. - ffpv, ffpv, 1.,
                                    0., 0., 0., 0.]),
                          np.array([sig_fpv, sig_fpv, 1.e-5,
                                    1.e-5, 1.e-5, 1.e-5, 1.e-5]))

        store_composition(solutions['mw'],
                          ['Mg', 'Fe'],
                          np.array([1. - ffper, ffper]),
                          np.array([sig_fper, sig_fper]))

        # Tweak compositions with 0.1% of a midpoint proportion
        # Do not consider (transformed) endmembers with < 5% abundance
        # in the solid solution. Copy the stored compositions from
        # each phase to the assemblage storage.
        compute_and_store_phase_compositions(assemblage,
                                             midpoint_proportion,
                                             constrain_endmembers,
                                             proportion_cutoff,
                                             copy_storage=True)

        Tange_TNFS_2009_FMS_assemblages.append(assemblage)

    return Tange_TNFS_2009_FMS_assemblages
