import numpy as np

from input_dataset import *


# garnet-bridgmanite-corundum data
with open('data/Liu_et_al_2016_gt_bdg_cor.dat', 'r') as f:
    gt_bdg_cor_data = [line.split() for line in f if line.split() != [] and line[0] != '#']

all_runs = [d[0] for d in gt_bdg_cor_data]
set_runs = list(set([d[0] for d in gt_bdg_cor_data]))


Liu_et_al_2016_MAS_assemblages = []

for i, run in enumerate(set_runs):
    run_indices = [idx for idx, x in enumerate(gt_bdg_cor_data) if x[0] == run]

    phases = []
    for run_idx in run_indices:
        run_id, PGPa, TK, phase, N, Mg, Mgerr, Al, Alerr, Si, Sierr = gt_bdg_cor_data[run_idx]

        pressure = float(PGPa)*1.e9
        temperature = float(TK)
        
        c = np.array([float(Mg), float(Al), float(Si)])
        sig_c = np.array([float(Mgerr), float(Alerr), float(Sierr)])
        
        if phase == 'gt':
            phases.append(py) # no excess Si for any run
        elif phase == 'bdg':
            phases.append(child_solutions['mg_al_bdg'])
            phases[-1].fitted_elements = ['Mg', 'Al', 'Si']
            phases[-1].composition = c
            phases[-1].compositional_uncertainties = sig_c
        elif phase == 'cor':
            phases.append(child_solutions['al_mg_cor'])
            phases[-1].fitted_elements = ['Mg', 'Al', 'Si']
            phases[-1].composition = c
            phases[-1].compositional_uncertainties = sig_c

            
    assemblage = burnman.Composite(phases)
    assemblage.experiment_id = 'Liu_2016_{0}'.format(run_id)
    assemblage.nominal_state = np.array([pressure, temperature]) # CONVERT P TO PA, T to K
    assemblage.state_covariances = np.array([[5.e8*5.e8, 0.], [0., 100.]])
    
    burnman.processanalyses.compute_and_set_phase_compositions(assemblage)
    
    assemblage.stored_compositions = ['composition not assigned']*len(assemblage.phases)
    for k in range(len(phases)):
        try:
            assemblage.stored_compositions[k] = (assemblage.phases[k].molar_fractions,
                                                 assemblage.phases[k].molar_fraction_covariances)
        except:
            pass

    Liu_et_al_2016_MAS_assemblages.append(assemblage)
