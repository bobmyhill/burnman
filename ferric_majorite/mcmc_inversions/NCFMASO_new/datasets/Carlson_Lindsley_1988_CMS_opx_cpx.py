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
    with open('data/Carlson_Lindsley_1988_CMS_opx_cpx.dat', 'r') as f:
        opx_cpx_data = [line.split() for line in f if line.split() != []
                        and line[0] != '#']

    set_runs = list(set([d[0] for d in opx_cpx_data]))

    Carlson_Lindsley_1988_CMS_assemblages = []

    print('Carlson and Lindsley inversion does not yet include pen or pig,')
    print('so only inverting opx-cpx data')
    for i, run in enumerate(set_runs):
        # for now, limit inversion to opx or cpx
        run_indices = [idx for idx, x in enumerate(opx_cpx_data)
                       if (x[0] == run and (x[3] == 'opx' or x[3] == 'cpx'))]
        if len(run_indices) > 1.:

            assemblage = AnalysedComposite([solutions['opx'],
                                            solutions['cpx']])

            assemblage.experiment_id = 'Carlson_Lindsley_1988_CMS_{0}'.format(run)

            state = list(map(float, opx_cpx_data[run_indices[0]][1:3]))

            pressure = state[0]*1.e8  # CONVERT P TO PA
            temperature = state[1] + 273.15  # TC to TK
            sig_p = pressure / 20.

            assemblage.nominal_state = np.array([pressure, temperature])
            assemblage.state_covariances = np.array([[sig_p*sig_p, 0.],
                                                     [0., 100.]])

            for idx in run_indices:
                datum = opx_cpx_data[idx]
                phase = datum[3]
                diav = (float(datum[4]) + float(datum[5]))/2.
                # additional uncertainty
                dierr = (float(datum[5]) - float(datum[4]))/2. + 0.005

                # diopside is CaMgSi2O6, so proportion of diopside is
                # also amount of Ca (on a 4 cation basis)
                xCa = diav
                xMg = 2. - diav
                xO = 5.999 # slightly less than 6 to prevent other endmembers
                if phase in ['opx', 'cpx']:
                    store_composition(solutions[phase],
                                      ['Mg', 'Ca', 'Si',
                                       'Na', 'Fe', 'Al', 'Fe_A', 'Fe_B', 'O'],
                                      np.array([xMg, xCa, 2.,
                                                0., 0., 0., 0., 0., xO]),
                                      np.array([dierr, dierr, 1.e-5,
                                                1.e-5, 1.e-5, 1.e-5, 1.e-5, 1.e-5, 1.e-5]))

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

            Carlson_Lindsley_1988_CMS_assemblages.append(assemblage)
    return Carlson_Lindsley_1988_CMS_assemblages
