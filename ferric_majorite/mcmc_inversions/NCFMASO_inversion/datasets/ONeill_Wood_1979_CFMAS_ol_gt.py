import numpy as np
import burnman

def get_assemblages(mineral_dataset):
    endmembers = mineral_dataset['endmembers']
    solutions = mineral_dataset['solutions']
    child_solutions = mineral_dataset['child_solutions']


    # Garnet-olivine partitioning data
    ol_gt_data = np.loadtxt('data/ONeill_Wood_1979_CFMAS_ol_gt.dat')

    ONeill_Wood_1979_CFMAS_assemblages = []
    run_id=0
    for PGPa, TK, xMgol, dxMgol, xFeol, dxFeol, xMggt, dxMggt, xFegt, dxFegt, xCagt, dxCagt in ol_gt_data:
        run_id+=1

        assemblage = burnman.Composite([solutions['ol'],
                                        child_solutions['py_alm_gr_gt']])

        assemblage.experiment_id = 'ONeill_Wood_1979_CFMAS_{0}'.format(run_id)
        assemblage.nominal_state = np.array([PGPa*1.e9, TK]) # CONVERT PRESSURE TO Pa
        assemblage.state_covariances = np.array([[5.e7*5.e7, 0.], [0., 100.]]) # 0.5 kbar pressure error

        solutions['ol'].fitted_elements = ['Mg', 'Fe']
        solutions['ol'].composition = np.array([xMgol, xFeol])
        solutions['ol'].compositional_uncertainties = np.array([dxMgol, dxFeol])

        child_solutions['py_alm_gr_gt'].fitted_elements = ['Mg', 'Fe', 'Ca']
        child_solutions['py_alm_gr_gt'].composition = np.array([xMggt, xFegt, xCagt])
        child_solutions['py_alm_gr_gt'].compositional_uncertainties = np.array([dxMggt, dxFegt, dxCagt])

        burnman.processanalyses.compute_and_set_phase_compositions(assemblage)

        assemblage.stored_compositions = [(assemblage.phases[k].molar_fractions,
                                           assemblage.phases[k].molar_fraction_covariances)
                                          for k in range(2)]


        ONeill_Wood_1979_CFMAS_assemblages.append(assemblage)

    return ONeill_Wood_1979_CFMAS_assemblages
