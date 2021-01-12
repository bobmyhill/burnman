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

    # Open file and read contents, stripping out comment lines
    with open('data/Beyer_et_al_2021_NCFMASO.dat', 'r') as f:
        ds = [line.split() for line in f
              if line.split() != [] and line[0] != '#']

    set_expts = sorted(list(set([d[0] for d in ds])))
    expt_capsules = [list(set([d[2] for d in ds if d[0] == expt]))
                     for expt in set_expts]

    Beyer_et_al_2019_NCFMASO_assemblages = []

    # Loop over experiments
    for i, expt in enumerate(set_expts):

        expt_indices = [idx for idx, x in enumerate(ds) if x[0] == expt]
        pressure = float(ds[expt_indices[0]][3])*1.e9
        temperature = float(ds[expt_indices[0]][4])+273.15

        # Loop over capsules in each experiment
        for j, capsule in enumerate(expt_capsules[i]):
            phase_indices = [expt_idx for expt_idx in expt_indices
                             if (ds[expt_idx][2] == capsule)]

            # WARNING include all experiments for the time being.
            if (len(phase_indices) > 1):
                # and (ds[expt_indices[0]][5] == 'Fe'
                #     or ds[expt_indices[0]][5] == 'Mo')):
                phases = []
                for idx in phase_indices:

                    phase_name = ds[idx][1]  # expt_id, phase_name, start_mat
                    # P(GPa), T(C)
                    # buffer = ds[idx][5]
                    # O = ds[idx][6]

                    if len(ds[idx]) >= 25:
                        c = list(map(float, ds[idx][7:]))
                        Si, Si_unc, Al, Al_unc = c[:4]
                        Fe2, Fe2_unc, Fe3, Fe3_unc = c[4:8]
                        Mo, Mo_unc, Mg, Mg_unc = c[8:12]
                        Ca, Ca_unc, Na, Na_unc = c[12:16]
                        Re, total_cations = c[16:18]

                        Si_var = Si_unc * Si_unc
                        Al_var = Al_unc * Al_unc
                        Mg_var = Mg_unc * Mg_unc
                        Ca_var = Ca_unc * Ca_unc
                        Na_var = Na_unc * Na_unc

                        if len(ds[idx]) > 25:
                            Fe_total, FeO_total_EMPA = c[18:20]
                            unc_FeO_total_EMPA = c[20]
                            # FeO_total_unc_prop = c[21]
                            # unc_Fe2O3 = c[22]
                            Fef_over_Fetotal = c[23]
                            unc_Fef_over_Fetotal = c[24]

                            Fe_unc = (unc_FeO_total_EMPA
                                      / FeO_total_EMPA * Fe_total)

                            Fef = Fef_over_Fetotal * Fe_total

                            J = np.array([[1., 0.],
                                          [Fef_over_Fetotal, Fe_total]])
                            covFe = np.array([[Fe_unc*Fe_unc, 0.],
                                              [0.,
                                               unc_Fef_over_Fetotal
                                               * unc_Fef_over_Fetotal]])
                            covFe = np.einsum('ij, jk, lk', J, covFe, J)
                        else:
                            Fe_total = Fe2
                            Fe_var = Fe2_unc * Fe2_unc
                            Fef = 0.
                            covFe = np.array([[np.max([1.e-10, Fe_var]), 0.],
                                              [0., 1.e-5]])

                        # Fef is only on site B for gt, cpx
                        # other phases assumed to be Fe3+-free
                        if phase_name == 'sp':
                            fitted_elements = ['Fe', 'Mg', 'Na', 'Ca',
                                               'Al', 'Si', 'Fe',
                                               'Fef_B', 'Fe_B', 'Mg_B']
                            composition = np.array([Fe_total, Mg, Na, Ca,
                                           Al, Si, Fe_total, Fef, 0., 0.])
                            cov = np.array([Fe2_unc, Mg_unc, 1.e-5, 1.e-5,
                                            1.e-5, 1.e-5, 1.e-5,
                                            1.e-5, 1.e-5, 1.e-5])
                        else:
                            fitted_elements = ['Na', 'Ca', 'Mg',
                                               'Al', 'Si', 'Fe', 'Fef_B']
                            composition = [Na, Ca, Mg,
                                           Al, Si, Fe_total, Fef]
                            cov = np.zeros((7, 7))
                            cov[:5, :5] = np.diag([np.max([1.e-10, u]) for u in
                                                   [Na_var, Ca_var, Mg_var,
                                                    Al_var, Si_var]])
                            cov[5:, 5:] = covFe

                    if phase_name in ['gt', 'ol', 'wad', 'sp',
                                      'opx', 'cpx', 'hpx']:
                        phases.append(solutions[phase_name])
                        store_composition(solutions[phase_name],
                                          fitted_elements,
                                          composition,
                                          cov)
                    elif phase_name in ['stv', 'Mo', 'MoO2', 'Re', 'ReO2',
                                        'fcc_iron']:
                        phases.append(endmembers[phase_name])
                    else:
                        raise Exception(f'{phase_name} not recognised')

                assemblage = AnalysedComposite(phases)

                # make pressure uncertainties v small.
                # We'll add dP hyperparameters to deal with
                # the multicapsule nature of the experiments.
                assemblage.experiment_id = 'Beyer2019_{0}'.format(expt)
                assemblage.nominal_state = np.array([pressure, temperature])
                assemblage.state_covariances = np.array([[1.e7*1.e7, 0.],
                                                         [0., 50.*50]])

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

                # assemblage.set_state(*assemblage.nominal_state)
                # print(assemblage.experiment_id,
                # [ph.name for ph in assemblage.phases],
                #      assemblage_affinity_misfit(assemblage))
                Beyer_et_al_2019_NCFMASO_assemblages.append(assemblage)

    return Beyer_et_al_2019_NCFMASO_assemblages
