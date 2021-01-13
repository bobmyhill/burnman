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

    # Garnet-olivine partitioning data
    ol_gt_data = np.loadtxt('data/ONeill_Wood_1979_ol_gt_KD.dat')

    ONeill_Wood_1979_assemblages = []
    run_id = 0
    for P, T, XMgOl, lnKD, lnKDerr in ol_gt_data:
        run_id += 1

        # KD is (XGtFe*XMgOl)/(XGtMg*XOlFe)
        # KD*(XMgGt*(1 - XMgOl)) = (XMgOl*(1 - XMgGt))
        # XMgGt*(KD*(1 - XMgOl)) = XMgOl - XMgOl*XMgGt
        # XMgGt*(KD*(1 - XMgOl) + XMgOl) = XMgOl
        # XMgGt = XMgOl/(KD*(1 - XMgOl) + XMgOl)
        # XMgGt = 1./(KD*(1./XMgOl - 1.) + 1.)
        # XMgGt = 1./(1. + KD*(1. - XMgOl)/XMgOl)

        KD = np.exp(lnKD)
        XMgGt = 1./(1. + ((1. - XMgOl)/XMgOl)*KD)
        dXMgGtdlnKD = -(1. - XMgOl)*KD/(XMgOl * np.power((1. - XMgOl)
                                                         * KD/XMgOl + 1., 2.))
        XMgGterr = np.abs(dXMgGtdlnKD*lnKDerr)  # typically ~0.01

        # Assume error is equal for both ol and gt
        XMgOlerr = XMgGterr

        assemblage = AnalysedComposite([solutions['ol'],
                                        solutions['gt']])

        assemblage.experiment_id = 'ONeill_Wood_1979_{0}'.format(run_id)
        assemblage.nominal_state = np.array([P*1.e9, T])  # P from GPa to Pa
        assemblage.state_covariances = np.array([[5.e7*5.e7, 0.],
                                                 [0., 100.]])  # 0.5 kbar P unc

        store_composition(solutions['ol'],
                          ['Mg', 'Fe'],
                          np.array([XMgOl, 1. - XMgOl]),
                          np.array([XMgOlerr, XMgOlerr]))

        store_composition(solutions['gt'],
                          ['Mg', 'Fe', 'Ca',
                           'Al', 'Si', 'Na', 'Fef_B', 'Mg_B'],
                          np.array([XMgGt, 1. - XMgGt, 0.,
                                    2./3., 1., 0., 0., 0.]),
                          np.array([XMgGterr, XMgGterr, 1.e-5,
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

        ONeill_Wood_1979_assemblages.append(assemblage)

    return ONeill_Wood_1979_assemblages
