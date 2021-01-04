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
    with open('data/Gasparik_Newton_1984_MAS_py_opx_sp_fo.dat', 'r') as f:
        py_opx_sp_fo_data = [line.split() for line in f
                             if line.split() != [] and line[0] != '#']

    Gasparik_Newton_1984_MAS_univariant_assemblages = []

    for run_id, TC, Pkbar, Perr, xMgts in py_opx_sp_fo_data:
        Perr = float(Perr)

        assemblage = AnalysedComposite([solutions['opx'],
                                        endmembers['py'],
                                        endmembers['sp'],
                                        endmembers['fo']])
        assemblage.experiment_id = 'Gasparik_Newton_1984_MAS_{0}'.format(run_id)

        # Convert P from kbar to Pa, T from C to K
        assemblage.nominal_state = np.array([float(Pkbar)*1.e8,
                                             float(TC)+273.15])
        assemblage.state_covariances = np.array([[Perr*Perr*1.e16, 0.],
                                                 [0., 100.]])

        xMgts = float(xMgts)
        Mg = 2. - xMgts
        Si = Mg
        Al = 2.*xMgts
        store_composition(solutions['opx'],
                          ['Mg', 'Al', 'Si', 'Na', 'Ca', 'Fe', 'Fe_B', 'O'],
                          np.array([Mg, Al, Si, 0., 0., 0., 0., 6.]),
                          np.array([0.02, 0.02, 0.02, 1.e-5, 1.e-5,
                                    1.e-5, 1.e-5, 0.02]))

        # Tweak compositions with 0.1% of a midpoint proportion
        # Do not consider (transformed) endmembers with < 5% abundance
        # in the solid solution. Copy the stored compositions from
        # each phase to the assemblage storage.
        compute_and_store_phase_compositions(assemblage,
                                             midpoint_proportion,
                                             constrain_endmembers,
                                             proportion_cutoff,
                                             copy_storage=True)

        Gasparik_Newton_1984_MAS_univariant_assemblages.append(assemblage)

    return Gasparik_Newton_1984_MAS_univariant_assemblages
