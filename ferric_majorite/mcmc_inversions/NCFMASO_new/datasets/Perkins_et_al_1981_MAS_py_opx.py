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
    with open('data/Perkins_et_al_1981_MAS_py_opx.dat', 'r') as f:
        py_opx_data = [line.split() for line in f
                       if line.split() != [] and line[0] != '#']

    Perkins_et_al_1981_MAS_assemblages = []

    for run_id, PGPa, Perr, TK, Terr, Al_fraction in py_opx_data:

        assemblage = AnalysedComposite([solutions['opx'],
                                        endmembers['py']])
        assemblage.experiment_id = 'Perkins_et_al_1981_MAS_{0}'.format(run_id)

        # Convert P to Pa from GPa
        Perr = float(Perr)
        Terr = float(Terr)
        assemblage.nominal_state = np.array([float(PGPa)*1.e9, float(TK)])
        assemblage.state_covariances = np.array([[Perr*Perr*1.e18, 0.],
                                                 [0., Terr*Terr]])

        xAl = float(Al_fraction)

        store_composition(solutions['opx'],
                          ['Mg', 'Al', 'Si', 'Fe', 'Ca', 'Na', 'Fe_B'],
                          np.array([(1. - xAl)/2., xAl, (1. - xAl)/2.,
                                    0., 0., 0., 0.]),
                          np.array([0.025, 0.005, 0.005,
                                    1.e-6, 1.e-6, 1.e-6, 1.e-6]))

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

        Perkins_et_al_1981_MAS_assemblages.append(assemblage)

    return Perkins_et_al_1981_MAS_assemblages
