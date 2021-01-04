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
    with open('data/Tsujino_et_al_2019_FMS_wad_ring.dat', 'r') as f:
        wad_rw_data = [line.split() for line in f if line.split() != []
                       and line[0] != '#']

    Tsujino_et_al_2019_FMS_wad_ring_assemblages = []

    # N.B. P in GPa, T in K
    for datum in wad_rw_data:
        run_id, load, tel, TK, t = datum[:5]
        Mg1, Mgerr1, Mg2, Mgerr2 = datum[5:9]
        KD, KDe, MgOV, MgOVe = datum[9:13]
        PTange, PTangeerr, PSpez, PSpezerr = datum[13:17]

        p_mwd = float(Mg1) / 100.
        p_mrw = float(Mg2) / 100.

        unc_mwd = float(Mgerr1) / 100.
        unc_mrw = float(Mgerr2) / 100.

        assemblage = AnalysedComposite([solutions['wad'], solutions['sp']])

        store_composition(solutions['wad'],
                          ['Mg', 'Fe'],
                          np.array([p_mwd, 1. - p_mwd]),
                          np.array([unc_mwd, unc_mwd]))

        store_composition(solutions['sp'],
                          ['Mg', 'Fe', 'Al', 'Fe_B', 'Mg_B', 'Si'],
                          np.array([p_mrw, 1. - p_mrw, 0., 0., 0., 0.5]),
                          np.array([unc_mrw, unc_mrw, 1.e-6, 1.e-6,
                                    1.e-6, 1.e-6]))

        # Tweak compositions with 0.1% of a midpoint proportion
        # Do not consider (transformed) endmembers with < 5% abundance
        # in the solid solution. Copy the stored compositions from
        # each phase to the assemblage storage.
        compute_and_store_phase_compositions(assemblage,
                                             midpoint_proportion,
                                             constrain_endmembers,
                                             proportion_cutoff,
                                             copy_storage=True)

        Pvar = np.power(float(PSpezerr)*1.e9, 2.)
        Tvar = 25.*25.

        assemblage.experiment_id = 'Tsujino_2019_FMS_{0}'.format(run_id)

        # Convert pressure to Pa from GPa
        assemblage.nominal_state = np.array([float(PSpez)*1.e9,
                                             float(TK)])
        assemblage.state_covariances = np.array([[Pvar, 0.],
                                                 [0., Tvar]])

        Tsujino_et_al_2019_FMS_wad_ring_assemblages.append(assemblage)
    return Tsujino_et_al_2019_FMS_wad_ring_assemblages
