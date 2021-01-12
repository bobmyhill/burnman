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
    with open('data/Seckendorff_ONeill_1992_ol_opx.dat', 'r') as f:
        ol_opx_data = [line.split() for line in f
                       if line.split() != [] and line[0] != '#']

    Seckendorff_ONeill_1992_assemblages = []

    # N.B. P in GPa
    for run_id, P, T, phase1, Fe1, Feerr1, phase2, Fe2, Feerr2 in ol_opx_data:

        pressure = float(P)*1.e9
        temperature = float(T)

        Feol = float(Fe1)
        Feopx = float(Fe2)
        Mgol = 1. - Feol
        Mgopx = 1. - Feopx

        Feol_unc = float(Feerr1)
        Feopx_unc = float(Feerr2)

        assemblage = AnalysedComposite([solutions['ol'],
                                        solutions['opx']])

        store_composition(solutions['ol'],
                          ['Mg', 'Fe', 'Si'],
                          np.array([Mgol, Feol, 0.5]),
                          np.array([Feol_unc, Feol_unc, 1.e-5]))

        store_composition(solutions['opx'],
                          ['Mg', 'Fe', 'Si', 'Al', 'Ca', 'Fef_B'],
                          np.array([Mgopx, Feopx, 1.,
                                    0., 0., 0.]),
                          np.array([Feopx_unc, Feopx_unc, 1.e-5,
                                    1.e-5, 1.e-5, 1.e-5]))

        assemblage.experiment_id = 'Seckendorff_ONeill_1992_{0}'.format(run_id)
        assemblage.nominal_state = np.array([pressure, temperature])
        assemblage.state_covariances = np.array([[0.1e9*0.1e9, 0.],
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

        Seckendorff_ONeill_1992_assemblages.append(assemblage)

    return Seckendorff_ONeill_1992_assemblages
