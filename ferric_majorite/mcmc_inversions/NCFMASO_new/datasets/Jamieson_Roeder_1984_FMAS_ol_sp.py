import numpy as np

from burnman.processanalyses import store_composition
from burnman.processanalyses import AnalysedComposite
from burnman.processanalyses import compute_and_store_phase_compositions
from global_constants import midpoint_proportion
from global_constants import constrain_endmembers
from global_constants import proportion_cutoff


def get_assemblages(mineral_dataset):
    solutions = mineral_dataset['solutions']

    # olivine-spinel partitioning data
    with open('data/Jamieson_Roeder_1984_FMAS_ol_sp.dat', 'r') as f:
        ol_sp_data = [line.split() for line in f if line.split() != []
                      and line[0] != '#']

    Jamieson_Roeder_1984_FMAS_ol_sp_assemblages = []

    # N.B. P in GPa, T in K
    run_id = 0
    for P, T, Mg1, Mgerr1, Mg2, Mgerr2 in ol_sp_data:
        run_id += 1

        Mg_ol = float(Mg1) / 100.
        Fe_ol = 1. - Mg_ol

        Mg_sp = float(Mg2) / 100.
        Fe_sp = 1. - Mg_sp

        # Here we make the uncertainties 4 times bigger
        # to more reasonably match the extent of disequilibrium
        Mg_ol_unc = float(Mgerr1) / 100. * 4.
        Mg_sp_unc = float(Mgerr2) / 100. * 4.

        store_composition(solutions['ol'],
                          ['Mg', 'Fe'],
                          np.array([Mg_ol, Fe_ol]),
                          np.array([Mg_ol_unc, Mg_ol_unc]))

        store_composition(solutions['sp'],
                          ['Mg', 'Fe', 'Al', 'Fef_A', 'Fef_B', 'Si'],
                          np.array([Mg_sp, Fe_sp, 2., 0., 0., 0.]),
                          np.array([Mg_sp_unc, Mg_sp_unc, 1.e-5,
                                    1.e-5, 1.e-5, 1.e-5]))

        assemblage = AnalysedComposite([solutions['ol'],
                                        solutions['sp']])

        # Convert pressure from GPa to Pa
        assemblage.experiment_id = 'Jamieson_Roeder_1984_FMAS_{0}'.format(run_id)
        assemblage.nominal_state = np.array([float(P)*1.e9, float(T)])
        assemblage.state_covariances = np.array([[1.e4*1.e4, 0.], [0., 100.]])

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

        Jamieson_Roeder_1984_FMAS_ol_sp_assemblages.append(assemblage)

    return Jamieson_Roeder_1984_FMAS_ol_sp_assemblages
