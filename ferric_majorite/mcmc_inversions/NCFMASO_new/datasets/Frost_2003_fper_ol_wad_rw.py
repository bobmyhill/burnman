import numpy as np

from burnman.processanalyses import store_composition
from burnman.processanalyses import AnalysedComposite
from burnman.processanalyses import compute_and_store_phase_compositions
from global_constants import midpoint_proportion
from global_constants import constrain_endmembers
from global_constants import proportion_cutoff


def get_assemblages(mineral_dataset):
    solutions = mineral_dataset['solutions']

    # Frost fper-ol-wad-rw partitioning data
    with open('data/Frost_2003_chemical_analyses.dat', 'r') as f:
        ds = [line.split() for line in f
              if line.split() != [] and line[0] != '#']

    set_runs = list(set([d[0] for d in ds]))
    all_chambers = [list(set([d[1] for d in ds if d[0] == run]))
                    for run in set_runs]

    Frost_2003_assemblages = []
    for i, run in enumerate(set_runs):
        run_indices = [idx for idx, x in enumerate(ds) if x[0] == run]
        pressure = float(ds[run_indices[0]][2])*1.e9
        temperature = float(ds[run_indices[0]][3])

        # Use only the data at > 2.2 GPa (i.e. not the PC experiment)
        if pressure > 2.2e9:
            for j, chamber in enumerate(all_chambers[i]):
                chamber_indices = [run_idx for run_idx in run_indices
                                   if (ds[run_idx][1] == chamber
                                       and ds[run_idx][4] != 'cen'
                                       and ds[run_idx][4] != 'anB'
                                       and ds[run_idx][4] != 'mag')]

                if len(chamber_indices) > 1:
                    phases = []
                    for idx in chamber_indices:
                        phase_name = ds[idx][4]
                        if phase_name in solutions:
                            phases.append(solutions[phase_name])
                        elif phase_name == 'ring':
                            phases.append(solutions['sp'])
                        else:
                            raise Exception(f"Phase {phase_name} not recognised")

                    assemblage = AnalysedComposite(phases)

                    assemblage.experiment_id = run
                    assemblage.nominal_state = np.array([pressure,
                                                         temperature])
                    assemblage.state_covariances = np.array([[1.e7*1.e7, 0.],
                                                             [0., 10.*10]])

                    fitted_elements = ['Fe', 'Al', 'Si', 'Mg',
                                       'Ca', 'Na', 'Fe_B', 'Mg_B']
                    for k, idx in enumerate(chamber_indices):
                        phase = assemblage.phases[k]
                        d = ds[chamber_indices[k]]
                        c = np.array(list(map(float,
                                              [d[5], d[7], d[9], d[11],
                                               0., 0., 0., 0.])))

                        sig_c = np.array([max(s, 0.001)
                                          for s in map(float, [d[6], d[8],
                                                               d[10], d[12],
                                                               1.e-6, 1.e-6,
                                                               1.e-6, 1.e-6])])

                        store_composition(phase, fitted_elements, c, sig_c)

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

                    Frost_2003_assemblages.append(assemblage)

    return Frost_2003_assemblages
