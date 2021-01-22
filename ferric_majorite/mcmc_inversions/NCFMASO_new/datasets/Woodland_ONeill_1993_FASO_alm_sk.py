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

    # Garnet-pyroxene partitioning data
    with open('data/Woodland_ONeill_1993_FASO_alm_sk_analyses.dat', 'r') as f:
        expt_data = [line.split() for line in f
                     if line.split() != [] and line[0] != '#']

    set_runs = list(set([d[0] for d in expt_data]))

    Woodland_ONeill_1993_FASO_assemblages = []
    for i, run in enumerate(set_runs):

        run_indices = [idx for idx, d in enumerate(expt_data) if d[0] == run]
        phase_names = [expt_data[idx][3] for idx in run_indices]
        phases = []
        for datum in [expt_data[idx] for idx in run_indices]:
            run_id = datum[0]
            PGPa, TK = list(map(float, datum[1:3]))
            phase = datum[3]
            c = list(map(float, datum[4:]))

            if phase in ['qtz', 'coe', 'stv', 'hem', 'fa', 'fcc_iron']:
                phases.append(endmembers[phase])
            elif phase in ['opx', 'sp', 'gt']:
                phases.append(solutions[phase])
                if phase == 'opx':  # we add oxygen as our constraint on Fe3+
                    nO = 6.
                elif phase == 'sp':
                    nO = 4.
                elif phase == 'gt':
                    nO = 12.
                store_composition(phases[-1],
                                  ['Al', 'Fe', 'Si', 'Mg', 'Ca', 'Na', 'O'],
                                  np.array([c[0], c[1]+c[2], c[3],
                                            0., 0., 0., nO]),
                                  np.array([0.01]*7))
            else:
                raise Exception('{0} not recognised'.format(phase))

        # Convert P to Pa
        pressure = PGPa * 1.e9
        temperature = TK

        sig_p = pressure/20. + 0.1e9

        # pick only experiments where one or more reactions are constrained
        if (((('sp' in phase_names) or ('hem' in phase_names)) and not
             (('hem' in phase_names) and ('coe' in phase_names)) and not
             run_id == 'u524')):
            assemblage = AnalysedComposite(phases)

            assemblage.experiment_id = 'Woodland_ONeill_1993_FASO_{0}'.format(run_id)

            assemblage.nominal_state = np.array([pressure, temperature])
            assemblage.state_covariances = np.array([[sig_p*sig_p, 0.],
                                                     [0., 100.]])

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

            Woodland_ONeill_1993_FASO_assemblages.append(assemblage)

    return Woodland_ONeill_1993_FASO_assemblages
