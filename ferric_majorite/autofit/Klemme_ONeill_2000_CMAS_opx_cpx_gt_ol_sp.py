import numpy as np
from burnman.processanalyses import compute_and_set_phase_compositions, assemblage_affinity_misfit

from input_dataset import *


# Garnet-pyroxene partitioning data
with open('data/Klemme_ONeill_2000_CMAS_opx_cpx_gt_ol_sp.dat', 'r') as f:
    expt_data = [line.split() for line in f if line.split() != [] and line[0] != '#']

set_runs = list(set([d[1] for d in expt_data]))

Klemme_ONeill_2000_CMAS_assemblages = []

for i, run in enumerate(set_runs):
    
    run_indices = [idx for idx, d in enumerate(expt_data) if d[1] == run]
    n_phases = len(run_indices)
    phases = []
    for datum in [expt_data[idx] for idx in run_indices]:
        master_id, run_id = datum[0:2]
        TC, Pkbar, t = map(float, datum[2:5])
        phase = datum[5]
        c = map(float, datum[6:])

        if phase == 'fo':
            phases.append(fo)
        elif phase == 'sp':
            phases.append(sp)
        elif phase == 'opx':
            phases.append(child_solutions['oen_mgts_odi'])
            
            child_solutions['oen_mgts_odi'].fitted_elements = ['Si', 'Al', 'Mg', 'Ca']
            child_solutions['oen_mgts_odi'].composition = np.array([c[0], c[2], c[4], c[6]])
            child_solutions['oen_mgts_odi'].compositional_uncertainties = np.array([c[1], c[3], c[5], c[7]])
        
        elif phase == 'cpx':
            phases.append(child_solutions['di_cen_cats'])

            child_solutions['di_cen_cats'].fitted_elements = ['Si', 'Al', 'Mg', 'Ca']
            child_solutions['di_cen_cats'].composition = np.array([c[0], c[2], c[4], c[6]])
            child_solutions['di_cen_cats'].compositional_uncertainties = np.array([c[1], c[3], c[5], c[7]])
            
        elif phase == 'gt':
            phases.append(child_solutions['py_gr_gt'])

            child_solutions['py_gr_gt'].fitted_elements = ['Si', 'Al', 'Mg', 'Ca']
            child_solutions['py_gr_gt'].composition = np.array([c[0], c[2], c[4], c[6]])
            child_solutions['py_gr_gt'].compositional_uncertainties = np.array([c[1], c[3], c[5], c[7]])
        else:
            raise Exception('phase not recognised')

    

    assemblage = burnman.Composite(phases)
    assemblage.experiment_id = 'Klemme_ONeill_2000_CMAS_{0}'.format(run_id)
    assemblage.nominal_state = np.array([Pkbar*1.e8,
                                         TC+273.15]) # CONVERT P TO PA, T to K
    assemblage.state_covariances = np.array([[5.e7*5.e7, 0.], [0., 100.]])
    

    burnman.processanalyses.compute_and_set_phase_compositions(assemblage)

    
    assemblage.stored_compositions = ['composition not assigned']*n_phases
    for k in range(n_phases):
        try:
            assemblage.stored_compositions[k] = (assemblage.phases[k].molar_fractions,
                                                 assemblage.phases[k].molar_fraction_covariances)
        except:
            pass # fo and sp are endmembers
    
    Klemme_ONeill_2000_CMAS_assemblages.append(assemblage)
