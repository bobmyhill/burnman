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
    with open('data/Nakajima_FR_2012_FM_bdg_fper.dat', 'r') as f:
        bdg_fper_data = [line.split() for line in f
                         if line.split() != [] and line[0] != '#']

    Nakajima_FR_2012_assemblages = []

    # N.B. P in GPa
    for datum in bdg_fper_data:
        run_id, mix, PGPa, TK, t = datum[:5]
        pressure = float(PGPa)*1.e9
        temperature = float(TK)

        Febdg, Febdg_unc, Fefper, Fefper_unc = list(map(float, datum[5:9]))

        Mgbdg = 1. - Febdg
        Mgfper = 1. - Fefper

        # samples in eqm with metallic Fe, no Al, so low Fe3+ in bdg
        assemblage = AnalysedComposite([solutions['bdg'],
                                        solutions['mw']])

        assemblage.experiment_id = 'Nakajima_FR_2012_{0}'.format(run_id)
        assemblage.nominal_state = np.array([pressure, temperature])
        # 1 GPa uncertainty in pressure
        assemblage.state_covariances = np.array([[1.e9*1.e9, 0.], [0., 100.]])

        store_composition(solutions['bdg'],
                          ['Fe', 'Mg', 'Si', 'Al', 'Ca', 'Na', 'Fef_B', 'Fef_A'],
                          np.array([Febdg, Mgbdg, 1.,
                                    0., 0., 0., 0., 0.]),
                          np.array([Febdg_unc, Febdg_unc, 1.e-6,
                                    1.e-6, 1.e-6, 1.e-6, 1.e-6, 1.e-6]))

        store_composition(solutions['mw'],
                          ['Fe', 'Mg'],
                          np.array([Fefper, Mgfper]),
                          np.array([Fefper_unc, Fefper_unc]))

        # Tweak compositions with 0.1% of a midpoint proportion
        # Do not consider (transformed) endmembers with < 5% abundance
        # in the solid solution. Copy the stored compositions from
        # each phase to the assemblage storage.
        compute_and_store_phase_compositions(assemblage,
                                             midpoint_proportion,
                                             constrain_endmembers,
                                             proportion_cutoff,
                                             copy_storage=True)

        Nakajima_FR_2012_assemblages.append(assemblage)

    return Nakajima_FR_2012_assemblages
