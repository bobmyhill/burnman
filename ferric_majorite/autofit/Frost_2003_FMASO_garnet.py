import numpy as np

from input_dataset import *

# Frost fper-ol-wad-rw partitioning data
with open('data/Frost_2003_FMASO_garnet_analyses.dat', 'r') as f:
    ds = [line.split() for line in f if line.split() != [] and line[0] != '#']

all_runs = [d[0] for d in ds]
set_runs = list(set([d[0] for d in ds]))
all_conditions = [(float(ds[all_runs.index(run)][2])*1.e9,
               float(ds[all_runs.index(run)][3])) for run in set_runs]
all_chambers = [list(set([d[1] for d in ds if d[0] == run])) for run in set_runs]

Frost_2003_gt_assemblages = []
for i, run in enumerate(set_runs):
    run_indices = [idx for idx, x in enumerate(ds) if x[0] == run]
    pressure = float(ds[run_indices[0]][2])*1.e9
    temperature = float(ds[run_indices[0]][3])

    for j, chamber in enumerate(all_chambers[i]):
        chamber_indices = [run_idx for run_idx in run_indices
                           if (ds[run_idx][1] == chamber and
                               ds[run_idx][4] != 'cen' and
                               ds[run_idx][4] != 'anB' and
                               ds[run_idx][4] != 'mag')]

        
        phases = []
        for idx in chamber_indices:
            try:
                phases.append(solutions[ds[idx][4]])
            except:
                phases.append(child_solutions[ds[idx][4]])

        assemblage = burnman.Composite(phases)
        
        assemblage.experiment_id = run
        assemblage.nominal_state = np.array([pressure, temperature])
        assemblage.state_covariances = np.array([[1.e7*1.e7, 0.], [0., 100.]])
        
        for k, idx in enumerate(chamber_indices):
            assemblage.phases[k].fitted_elements = ['Fe', 'Al', 'Si', 'Mg']
            
            assemblage.phases[k].composition = np.array([float(ds[idx][5]),
                                                         float(ds[idx][7]),
                                                         float(ds[idx][9]),
                                                         float(ds[idx][11])])
            assemblage.phases[k].compositional_uncertainties = np.array([max(float(ds[idx][6]), 0.001),
                                                                         max(float(ds[idx][8]), 0.001),
                                                                         max(float(ds[idx][10]), 0.001),
                                                                         max(float(ds[idx][12]), 0.001)])
            
        burnman.processanalyses.compute_and_set_phase_compositions(assemblage)
        
        assemblage.stored_compositions = [(assemblage.phases[k].molar_fractions,
                                           assemblage.phases[k].molar_fraction_covariances)
                                          for k in range(len(chamber_indices))]
        
        
        Frost_2003_gt_assemblages.append(assemblage)

