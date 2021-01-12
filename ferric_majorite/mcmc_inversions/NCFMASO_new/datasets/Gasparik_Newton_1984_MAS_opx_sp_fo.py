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
    with open('data/Gasparik_Newton_1984_MAS_opx_sp_fo.dat', 'r') as f:
        opx_sp_fo_data = [line.split() for line in f
                          if line.split() != [] and line[0] != '#']

    Gasparik_Newton_1984_MAS_assemblages = []

    for run_id, mix, TC, Pkbar, t, N, Mg, Al, Si in opx_sp_fo_data:

        Mg = float(Mg)
        Al = float(Al)
        Si = float(Si)

        assemblage = AnalysedComposite([solutions['opx'],
                                        endmembers['sp'], endmembers['fo']])
        assemblage.experiment_id = 'Gasparik_Newton_1984_MAS_{0}'.format(run_id)
        # Convert P from kbar to Pa, T from C to K
        assemblage.nominal_state = np.array([float(Pkbar)*1.e8,
                                             float(TC)+273.15])
        assemblage.state_covariances = np.array([[0.1e9*0.1e9, 0.],
                                                 [0., 100.]])

        store_composition(solutions['opx'],
                          ['Mg', 'Al', 'Si', 'Na', 'Ca', 'Fe', 'Fe_B'],
                          np.array([Mg, Al, Si, 0., 0., 0., 0.]),
                          np.array([0.02, 0.02, 0.02, 1.e-5, 1.e-5,
                                    1.e-5, 1.e-5]))

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

        Gasparik_Newton_1984_MAS_assemblages.append(assemblage)

    return Gasparik_Newton_1984_MAS_assemblages
