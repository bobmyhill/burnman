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

    # Matsuzaka fper-rw-stv equilibria
    with open('data/Matsuzaka_et_al_2000_rw_wus_stv.dat', 'r') as f:
        ds = [line.split() for line in f
              if line.split() != [] and line[0] != '#']

    Matsuzaka_2000_assemblages = []
    for datum in ds:
        # all experimental rows have rw and mw as the first and second phases
        run_id, P, T, t = datum[:4]
        ph1, Mg_ring, Mgerr_ring = datum[4:7]
        ph2, Mg_mw, Mgerr_mw = datum[7:]

        Mg_ring = float(Mg_ring)
        Mgerr_ring = float(Mgerr_ring)
        Mg_mw = float(Mg_mw)
        Mgerr_mw = float(Mgerr_mw)

        Fe_mw = 1. - Mg_mw
        Fe_ring = 1. - Mg_ring

        # Time to equilibration was about 180 minutes
        # (don't include shorter runs)
        # All of the run products of the above experiments contained stishovite
        if float(t) > 179.:
            assemblage = AnalysedComposite([solutions['sp'],
                                            solutions['mw'],
                                            endmembers['stv']])

            assemblage.experiment_id = run_id
            assemblage.nominal_state = np.array([float(P)*1.e9, float(T)])
            assemblage.state_covariances = np.array([[0.5e9*0.5e9, 0.],
                                                     [0., 10.*10]])

            store_composition(solutions['mw'],
                              ['Fe', 'Mg'],
                              np.array([Fe_mw, Mg_mw]),
                              np.array([Mgerr_mw, Mgerr_mw]))

            store_composition(solutions['sp'],
                              ['Fe', 'Mg', 'Si',
                               'Al', 'Ca', 'Na', 'Fe_B', 'Fef_B'],
                              np.array([Fe_ring, Mg_ring, 0.5,
                                        0., 0., 0., 0., 0.]),
                              np.array([Mgerr_ring, Mgerr_ring, 1.e-5,
                                        1.e-5, 1.e-5, 1.e-5, 1.e-5, 1.e-5]))

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

            Matsuzaka_2000_assemblages.append(assemblage)

    return Matsuzaka_2000_assemblages
