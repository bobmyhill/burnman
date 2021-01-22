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
    with open('data/Katsura_et_al_2004_FMS_ol_wad.dat', 'r') as f:
        ol_wad_data = [line.split() for line in f if line.split() != []
                       and line[0] != '#']

    Katsura_et_al_2004_FMS_ol_wad_assemblages = []

    # N.B. P in GPa, T in K
    for datum in ol_wad_data:
        run_id, TK, MgOV, MgOVe = datum[:4]
        PMatsui, PMatsuierr, tmin = datum[4:7]
        Mg1, Mgerr1, Mg2, Mgerr2, KD, KDe = datum[7:]

        Mg_ol = float(Mg1) / 100.
        Fe_ol = 1. - Mg_ol

        Mg_wad = float(Mg2) / 100.
        Fe_wad = 1. - Mg_wad

        Mg_ol_unc = float(Mgerr1) / 100.
        Mg_wad_unc = float(Mgerr2) / 100.

        assemblage = AnalysedComposite([solutions['ol'], solutions['wad']])

        Pvar = np.power(float(PMatsuierr)*1.e9, 2.)
        Tvar = 50.*50.

        store_composition(solutions['ol'],
                          ['Mg', 'Fe'],
                          np.array([Mg_ol, Fe_ol]),
                          np.array([Mg_ol_unc, Mg_ol_unc]))

        store_composition(solutions['wad'],
                          ['Mg', 'Fe'],
                          np.array([Mg_wad, Fe_wad]),
                          np.array([Mg_wad_unc, Mg_wad_unc]))

        # Convert pressure from GPa to Pa
        assemblage.experiment_id = 'Katsura_2004_FMS_{0}'.format(run_id)
        assemblage.nominal_state = np.array([float(PMatsui)*1.e9,
                                             float(TK)])
        assemblage.state_covariances = np.array([[Pvar, 0.],
                                                 [0., Tvar]])

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

        Katsura_et_al_2004_FMS_ol_wad_assemblages.append(assemblage)
    return Katsura_et_al_2004_FMS_ol_wad_assemblages
