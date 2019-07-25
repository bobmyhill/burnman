import numpy as np
import burnman

def get_assemblages(mineral_dataset):
    endmembers = mineral_dataset['endmembers']
    solutions = mineral_dataset['solutions']
    child_solutions = mineral_dataset['child_solutions']

    # Garnet-olivine partitioning data
    ol_gt_data = np.loadtxt('data/ONeill_Wood_1979_ol_gt_KD.dat')

    ONeill_Wood_1979_assemblages = []
    run_id=0
    for P, T, XMgOl, lnKD, lnKDerr in ol_gt_data:
        run_id+=1

        # KD is (XGtFe*XMgOl)/(XGtMg*XOlFe)
        # KD*(XMgGt*(1 - XMgOl)) = (XMgOl*(1 - XMgGt))
        # XMgGt*(KD*(1 - XMgOl)) = XMgOl - XMgOl*XMgGt
        # XMgGt*(KD*(1 - XMgOl) + XMgOl) = XMgOl
        # XMgGt = XMgOl/(KD*(1 - XMgOl) + XMgOl)
        # XMgGt = 1./(KD*(1./XMgOl - 1.) + 1.)
        # XMgGt = 1./(1. + KD*(1. - XMgOl)/XMgOl)

        KD = np.exp(lnKD)
        XMgGt = 1./( 1. + ((1. - XMgOl)/XMgOl)*KD)
        dXMgGtdlnKD = -(1. - XMgOl)*KD/(XMgOl * np.power((1. - XMgOl)
                                                         * KD/XMgOl + 1., 2.))
        XMgGterr = np.abs(dXMgGtdlnKD*lnKDerr)  # typically ~0.01

        assemblage = burnman.Composite([solutions['ol'],
                                        child_solutions['py_alm_gt']])

        assemblage.experiment_id = 'ONeill_Wood_1979_{0}'.format(run_id)
        assemblage.nominal_state = np.array([P*1.e9, T])  # P from GPa to Pa
        assemblage.state_covariances = np.array([[5.e7*5.e7, 0.],
                                                 [0., 100.]])  # 0.5 kbar P unc

        solutions['ol'].fitted_elements = ['Mg', 'Fe']
        solutions['ol'].composition = np.array([XMgOl, 1. - XMgOl])
        solutions['ol'].compositional_uncertainties = np.array([XMgGterr,
                                                                XMgGterr])

        child_solutions['py_alm_gt'].fitted_elements = ['Mg', 'Fe']
        child_solutions['py_alm_gt'].composition = np.array([XMgGt,
                                                             1. - XMgGt])
        child_solutions['py_alm_gt'].compositional_uncertainties = np.array([XMgGterr, XMgGterr])

        burnman.processanalyses.compute_and_set_phase_compositions(assemblage)

        assemblage.stored_compositions = [(assemblage.phases[k].molar_fractions,
                                           assemblage.phases[k].molar_fraction_covariances)
                                          for k in range(2)]

        ONeill_Wood_1979_assemblages.append(assemblage)

    return ONeill_Wood_1979_assemblages
