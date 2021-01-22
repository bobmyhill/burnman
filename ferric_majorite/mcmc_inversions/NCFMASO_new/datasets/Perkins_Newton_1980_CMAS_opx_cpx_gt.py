import numpy as np

from burnman.processanalyses import store_composition
from burnman.processanalyses import AnalysedComposite
from burnman.processanalyses import compute_and_store_phase_compositions
from global_constants import midpoint_proportion
from global_constants import constrain_endmembers
from global_constants import proportion_cutoff


def get_assemblages(mineral_dataset):
    solutions = mineral_dataset['solutions']

    # Garnet-pyroxene partitioning data
    with open('data/Perkins_Newton_1980_CMAS_opx_cpx_gt.dat', 'r') as f:
        opx_cpx_gt_data = [line.split() for line in f
                           if line.split() != [] and line[0] != '#']

    Perkins_Newton_1980_CMAS_assemblages = []

    for datum in opx_cpx_gt_data:
        if len(datum) == 19:  # just opx and cpx
            assemblage = AnalysedComposite([solutions['opx'],
                                            solutions['cpx']])
        elif len(datum) == 21:  # opx, cpx, gt
            assemblage = AnalysedComposite([solutions['opx'],
                                            solutions['cpx'],
                                            solutions['gt']])
            Mggt = (float(datum[19]) + float(datum[20]))/200.
            Mggterr = (float(datum[20]) - float(datum[19]))/200.
            store_composition(solutions['gt'],
                              ['Mg', 'Ca', 'Al', 'Si',
                               'Fe', 'Na', 'Fef_B'],
                              np.array([Mggt, 1. - Mggt, 2./3., 1.,
                                        0., 0., 0.]),
                              np.array([Mggterr, Mggterr, 1.e-5, 1.e-5,
                                        1.e-5, 1.e-5, 1.e-5]))
        else:
            raise Exception('Wrong number of columns')

        # Convert pressure from Pkbar to Pa and temperature from C to K
        pressure = float(datum[1])*1.e8
        temperature = float(datum[2]) + 273.15
        sig_p = 0.1e9 + pressure/20.

        assemblage.experiment_id = 'Perkins_et_al_1981_MAS_{0}'.format(datum[0])
        assemblage.nominal_state = np.array([pressure, temperature])
        assemblage.state_covariances = np.array([[sig_p*sig_p, 0.],
                                                 [0., 100.]])

        c_vector = np.array(list(map(float, datum[3:19])))
        cav = (c_vector[:8] + c_vector[8:])/2.
        cerr = np.abs((c_vector[8:] - c_vector[:8])/2.) + 0.001  # additional uncertainty

        Caopx, Mgopx, Alopx, Siopx = cav[:4]
        Cacpx, Mgcpx, Alcpx, Sicpx = cav[4:]
        Caopx_unc, Mgopx_unc, Alopx_unc, Siopx_unc = cerr[:4]
        Cacpx_unc, Mgcpx_unc, Alcpx_unc, Sicpx_unc = cerr[4:]

        store_composition(solutions['opx'],
                          ['Ca', 'Mg', 'Al', 'Si',
                           'Fe', 'Na', 'Fef_A', 'Fef_B'],
                          np.array([Caopx, Mgopx, Alopx, Siopx,
                                    0., 0., 0., 0.]),
                          np.array([Caopx_unc, Mgopx_unc, Alopx_unc, Siopx_unc,
                                    1.e-5, 1.e-5, 1.e-5, 1.e-5]))

        store_composition(solutions['cpx'],
                          ['Ca', 'Mg', 'Al', 'Si',
                           'Fe', 'Na', 'Fef_A', 'Fef_B'],
                          np.array([Cacpx, Mgcpx, Alcpx, Sicpx,
                                    0., 0., 0., 0.]),
                          np.array([Cacpx_unc, Mgcpx_unc, Alcpx_unc, Sicpx_unc,
                                    1.e-5, 1.e-5, 1.e-5, 1.e-5]))

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

        Perkins_Newton_1980_CMAS_assemblages.append(assemblage)

    return Perkins_Newton_1980_CMAS_assemblages
