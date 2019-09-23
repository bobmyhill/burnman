import numpy as np
import burnman
from burnman.processanalyses import compute_and_set_phase_compositions


def get_assemblages(mineral_dataset):
    solutions = mineral_dataset['solutions']
    child_solutions = mineral_dataset['child_solutions']

    print('Warning: Perkins and Vielzeuf cpx could have Fe3+. '
          'Uncertainties guessed')
    print('N.B. Only using ol-cpx equilibria with Ca in ol < 0.02 '
          'and Ca in cpx > 0.475 (near di-hed join)')

    # Garnet-pyroxene partitioning data
    cpx_ol_data = np.loadtxt('data/Perkins_Vielzeuf_1992_CFMS_ol_cpx.dat')

    Perkins_Vielzeuf_1992_CFMS_assemblages = []

    for run_id, (PGPa, TK, Cacpx, Mgcpx, Fecpx, Caol, Mgol, Feol) in enumerate(cpx_ol_data):

        if Cacpx > 0.475 and Caol < 0.02:
            assemblage = burnman.Composite([child_solutions['di_hed'],
                                            solutions['ol']])
            assemblage.experiment_id = 'Perkins_Vielzeuf_1992_CFMS_{0}'.format(run_id)
            assemblage.nominal_state = np.array([PGPa*1.e9, TK])  # GPa to Pa
            assemblage.state_covariances = np.array([[5.e8*5.e8, 0.],
                                                     [0., 100.]])

            for (phase, comp) in [[solutions['ol'], np.array([Mgol, Feol])],
                                  [child_solutions['di_hed'],
                                   np.array([Mgcpx, Fecpx])]]:
                phase.fitted_elements = ['Mg', 'Fe']
                phase.composition = comp
                phase.compositional_uncertainties = np.array([0.01, 0.01])

            compute_and_set_phase_compositions(assemblage)

            assemblage.stored_compositions = [(assemblage.phases[k].molar_fractions,
                                               assemblage.phases[k].molar_fraction_covariances)
                                              for k in range(2)]

            Perkins_Vielzeuf_1992_CFMS_assemblages.append(assemblage)

    return Perkins_Vielzeuf_1992_CFMS_assemblages