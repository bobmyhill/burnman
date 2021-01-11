import numpy as np

from burnman.processanalyses import store_composition
from burnman.processanalyses import AnalysedComposite
from burnman.processanalyses import compute_and_store_phase_compositions
from global_constants import midpoint_proportion
from global_constants import constrain_endmembers
from global_constants import proportion_cutoff


def get_assemblages(mineral_dataset):
    solutions = mineral_dataset['solutions']

    # garnet-bridgmanite-corundum data
    with open('data/Hirose_et_al_2001_MAS_ilm_gt_bdg.dat', 'r') as f:
        ilm_bdg_gt_data = [line.split() for line in f
                           if line.split() != [] and line[0] != '#']

    # Adjustment for Anderson -> Jamieson Au pressure scale
    delta_pressure = 2.e9

    Hirose_et_al_2001_MAS_assemblages = []

    for i, datum in enumerate(ilm_bdg_gt_data):
        phases = []

        PGPa, TC, phase1, phase2, ph1_Al2O3, ph2_Al2O3 = datum

        pressure = float(PGPa)*1.e9 + delta_pressure
        temperature = float(TC) + 273.15

        for (phase_name, Al) in [(phase1, ph1_Al2O3),
                                 (phase2, ph2_Al2O3)]:
            Al = float(Al)
            Mg = (1. - Al)/2.
            Si = (1. - Al)/2.

            phases.append(solutions[phase_name])

            store_composition(solutions[phase_name],
                              ['Mg', 'Al', 'Si', 'Na', 'Ca', 'Fe', 'Fef_B', 'O'],
                              np.array([Mg, Al, Si, 0., 0., 0., 0., 1.5]),
                              np.array([0.01, 0.01, 0.01, 1.e-5, 1.e-5,
                                        1.e-5, 1.e-5, 1.e-5]))

        assemblage = AnalysedComposite(phases)
        assemblage.experiment_id = 'Hirose_2001_{0}'.format(i)
        assemblage.nominal_state = np.array([pressure, temperature])
        assemblage.state_covariances = np.array([[5.e8*5.e8, 0.], [0., 100.]])

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

        Hirose_et_al_2001_MAS_assemblages.append(assemblage)

    return Hirose_et_al_2001_MAS_assemblages
