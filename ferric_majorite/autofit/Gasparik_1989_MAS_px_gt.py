import numpy as np

from input_dataset import *


# Garnet-pyroxene partitioning data
with open('data/Gasparik_1989_MAS_px_gt.dat', 'r') as f:
    px_gt_data = [line.split() for line in f if line.split() != [] and line[0] != '#']

all_runs = [d[0] for d in px_gt_data]
set_runs = list(set([d[0] for d in px_gt_data]))


Gasparik_1989_MAS_assemblages = []

for i, run in enumerate(set_runs):
    run_indices = [idx for idx, x in enumerate(px_gt_data) if x[0] == run]

    phases = []
    for run_idx in run_indices:
        run_id, mix, t, ramP, Pkbar, TC, phase, N, Mg, Al, Si, cation_sum = px_gt_data[run_idx]

        if phase == 'gt':
            phases.append(child_solutions['py_dmaj_gt'])
            phases[-1].fitted_elements = ['Mg', 'Al', 'Si']
            phases[-1].composition = np.array([float(Mg), float(Al), float(Si)])
            phases[-1].compositional_uncertainties = np.array([0.01, 0.01, 0.01])
        elif phase == 'hpx':
            phases.append(hen) # within 0.5% of pure hen

            
    assemblage = burnman.Composite(phases)
    assemblage.experiment_id = 'Gasparik_1989_MAS_{0}'.format(run_id)
    assemblage.nominal_state = np.array([float(Pkbar)*1.e8,
                                         float(TC)+273.15]) # CONVERT P TO PA, T to K
    assemblage.state_covariances = np.array([[5.e8*5.e8, 0.], [0., 100.]])
    
    burnman.processanalyses.compute_and_set_phase_compositions(assemblage)

    

    assemblage.stored_compositions = ['composition not assigned']*len(assemblage.phases)
    for k in range(len(phases)):
        try:
            assemblage.stored_compositions[k] = (assemblage.phases[k].molar_fractions,
                                                 assemblage.phases[k].molar_fraction_covariances)
        except:
            pass

    Gasparik_1989_MAS_assemblages.append(assemblage)
