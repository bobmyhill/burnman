import numpy as np

from burnman.processanalyses import store_composition
from burnman.processanalyses import AnalysedComposite
from burnman.processanalyses import compute_and_store_phase_compositions
from global_constants import midpoint_proportion
from global_constants import constrain_endmembers
from global_constants import proportion_cutoff


def get_assemblages(mineral_dataset):
    solutions = mineral_dataset['solutions']

    print('Warning: Perkins and Vielzeuf cpx could have Fe3+. '
          'Uncertainties guessed')
    print('N.B. Only using ol-cpx equilibria with Ca in ol < 0.02 ')

    # Garnet-pyroxene partitioning data
    cpx_ol_data = np.loadtxt('data/Perkins_Vielzeuf_1992_CFMS_ol_cpx.dat')

    Perkins_Vielzeuf_1992_CFMS_assemblages = []

    for run_id, (PGPa, TK, Cacpx, Mgcpx, Fecpx, Caol, Mgol, Feol) in enumerate(cpx_ol_data):

        if Caol < 0.02:
            assemblage = AnalysedComposite([solutions['ol'],
                                            solutions['cpx']])

            store_composition(solutions['ol'],
                              ['Mg', 'Fe'],
                              np.array([Mgol, Feol]),
                              np.array([0.01, 0.01]))

            # Ca+Mg+Fe sum to one
            store_composition(solutions['cpx'],
                              ['Ca', 'Mg', 'Fe', 'Si', 'Al', 'Na', 'Fef_B'],
                              np.array([Cacpx, Mgcpx, Fecpx, 1., 0., 0., 0.]),
                              np.array([0.01, 0.01, 0.01, 1.e-5,
                                        1.e-5, 1.e-5, 1.e-5]))

            assemblage.experiment_id = 'Perkins_Vielzeuf_1992_CFMS_{0}'.format(run_id)
            assemblage.nominal_state = np.array([PGPa*1.e9, TK])  # GPa to Pa
            assemblage.state_covariances = np.array([[5.e8*5.e8, 0.],
                                                     [0., 100.]])

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

            Perkins_Vielzeuf_1992_CFMS_assemblages.append(assemblage)

    return Perkins_Vielzeuf_1992_CFMS_assemblages
