import numpy as np
import burnman
from fitting_functions import equilibrium_order
from burnman.processanalyses import assemblage_affinity_misfit

def get_assemblages(mineral_dataset):
    endmembers = mineral_dataset['endmembers']
    solutions = mineral_dataset['solutions']
    child_solutions = mineral_dataset['child_solutions']

    # Garnet-clinopyroxene partitioning data
    with open('data/Beyer_et_al_2019_NCFMASO.dat', 'r') as f:
        ds = [line.split() for line in f
              if line.split() != [] and line[0] != '#']

    set_runs = sorted(list(set([d[0] for d in ds])))
    all_chambers = [list(set([d[2] for d in ds if d[0] == run]))
                    for run in set_runs]

    Beyer_et_al_2019_NCFMASO_assemblages = []

    for i, run in enumerate(set_runs):

        run_indices = [idx for idx, x in enumerate(ds) if x[0] == run]
        pressure = float(ds[run_indices[0]][3])*1.e9
        temperature = float(ds[run_indices[0]][4])+273.15

        for j, chamber in enumerate(all_chambers[i]):
            chamber_indices = [run_idx for run_idx in run_indices
                               if (ds[run_idx][2] == chamber
                                   and ds[run_idx][1] != 'unwanted phases')]  # include all phases for now.

            # WARNING No Re runs (maybe not at eqm?)
            if (len(chamber_indices) > 1
                and (ds[run_indices[0]][5] == 'Fe')):
                phases = []
                for idx in chamber_indices:

                    phase_name = ds[idx][1]

                    if phase_name == 'ring?':  # be confident!
                        phase_name = 'ring'

                    # Begin loop over phases
                    if phase_name == 'gt':
                        c = np.array([float(ds[idx][cidx])
                                      for cidx in [7, 9, 11, 15, 17, 19]])
                        if c[-1] < 0.05:  # Na < 0.05
                            phases.append(child_solutions['xna_gt'])
                        elif c[-1] > c[0] - 3. - 0.05:  # Na > (Si - 3) - 0.05
                            phases.append(child_solutions['xmj_gt'])
                        else:
                            phases.append(solutions['gt'])

                    # only include cpx for Fe-saturated runs
                    elif (phase_name == 'cpx'
                          and (ds[run_indices[0]][5] == 'Fe')):
                        phases.append(solutions['cpx'])
                    elif (phase_name == 'hpx' and chamber == 'BFI'):
                        phases.append(child_solutions['mg_fe_hpx'])
                    elif phase_name == 'hpx':
                        phases.append(child_solutions['cfm_hpx'])
                    else:
                        phase_found = False
                        for phase_dict in [endmembers,
                                           solutions,
                                           child_solutions]:
                            if phase_name in phase_dict:
                                phases.append(phase_dict[phase_name])
                                phase_found = True
                        if not phase_found:
                            raise Exception('phase {0} not recognised'.format(phase_name))

                assemblage = burnman.Composite(phases)

                # make pressure uncertainties v small.
                # We'll add dP hyperparameters to deal with
                # the multichamber nature of the experiments.
                assemblage.experiment_id = 'Beyer2019_{0}'.format(run)
                assemblage.nominal_state = np.array([pressure, temperature])
                assemblage.state_covariances = np.array([[1.e7*1.e7, 0.],
                                                         [0., 50.*50]])
                for k, idx in enumerate(chamber_indices):

                    # Si, Al, FeT, [Mo], Mg, Ca, Na, [Re, no unc]
                    c = np.array([float(ds[idx][cidx]) for cidx in [7, 9,
                                                                    11, 15,
                                                                    17, 19]])

                    sig_c = np.array([max(float(ds[idx][cidx]), 0.01)
                                      for cidx in [8, 10, 12, 16, 18, 20]])

                    # cation_total = float(ds[idx][22])
                    f = float(ds[idx][23])  # Fe3+/sumFe
                    sig_f = float(ds[idx][24])  # sigma Fe3+/sumFe

                    # Solution phases in this dataset are:
                    # fper, ol, wad, ring (Fe-Mg exchange only)
                    # opx, hpx (CFMASO)
                    # cpx, gt (NCFMASO)
                    if (assemblage.phases[k] is solutions['mw']
                        or assemblage.phases[k] is solutions['ol']
                        or assemblage.phases[k] is solutions['wad']
                        or assemblage.phases[k] is child_solutions['ring']):

                        assemblage.phases[k].fitted_elements = ['Fe', 'Mg']
                        assemblage.phases[k].composition = c[2:4]
                        assemblage.phases[k].compositional_uncertainties = sig_c[2:4]

                    elif (assemblage.phases[k] is solutions['opx']
                          or assemblage.phases[k] is solutions['hpx']  # CFMAS
                          or assemblage.phases[k]
                          is child_solutions['cfm_hpx']  # CFMS
                          or assemblage.phases[k]
                          is child_solutions['mg_fe_hpx']):  # FMS
                        assemblage.phases[k].fitted_elements = ['Si', 'Al',
                                                                'Fe', 'Mg',
                                                                'Ca', 'Fe_B']
                        assemblage.phases[k].composition = np.zeros(6)
                        assemblage.phases[k].composition[:5] = c[0:5]
                        assemblage.phases[k].composition[5] = c[2]*0.5  # assume disorder

                        assemblage.phases[k].compositional_uncertainties = np.zeros(6)
                        assemblage.phases[k].compositional_uncertainties[0:5] = sig_c[0:5]
                        assemblage.phases[k].compositional_uncertainties[5] = c[2]*0.5  # large uncertainty for Mg on A

                        # The following adjusts compositions to reach equilibrium
                        a = burnman.Composite([assemblage.phases[k]])
                        burnman.processanalyses.compute_and_set_phase_compositions(a)

                        a.set_state(pressure, temperature)
                        equilibrium_order(assemblage.phases[k])

                    elif (assemblage.phases[k] is solutions['cpx']): # NCFMASO

                        assemblage.phases[k].fitted_elements = ['Si', 'Al',
                                                                'Fe', 'Mg',
                                                                'Ca', 'Na',
                                                                'Fef_B',
                                                                'Fe_A']

                        # Fudge Fe3+ and sigma for now
                        print('WARNING! Fudging Fe3+ (10%) in cpx '
                              'in Fe-saturated runs')
                        f = 0.1
                        sig_f = 0.1

                        fea = 0.5 - f/2. # assume disorder of Fe2+
                        sig_fea = 0.5 - f/2.  # large uncertainty for Fe on A

                        solutions['cpx'].composition = np.zeros(8)
                        solutions['cpx'].composition[:6] = c
                        solutions['cpx'].composition[6] = c[2] * f  # Fe3+
                        solutions['cpx'].composition[7] = c[2] * fea

                        # Uncertainties
                        solutions['cpx'].compositional_uncertainties = np.zeros(8)
                        solutions['cpx'].compositional_uncertainties[:6] = sig_c
                        solutions['cpx'].compositional_uncertainties[6] = c[2] * sig_f  # Fe3+
                        solutions['cpx'].compositional_uncertainties[7] = c[2] * sig_fea

                        # The following adjusts compositions to reach equilibrium
                        a = burnman.Composite([solutions['cpx']])
                        burnman.processanalyses.compute_and_set_phase_compositions(a)
                        a.set_state(pressure, temperature)
                        equilibrium_order(solutions['cpx'])

                    elif (assemblage.phases[k] is solutions['gt']
                          or assemblage.phases[k] is child_solutions['xna_gt']
                          or assemblage.phases[k] is child_solutions['xmj_gt']):
                        assemblage.phases[k].fitted_elements = ['Si', 'Al',
                                                                'Fe', 'Mg',
                                                                'Ca', 'Na',
                                                                'Fef_B']

                        # Composition
                        assemblage.phases[k].composition = np.zeros(7)
                        assemblage.phases[k].composition[:6] = np.copy(c)
                        assemblage.phases[k].composition[6] = c[2] * f  # Fe3+

                        # Uncertainties
                        assemblage.phases[k].compositional_uncertainties = np.zeros(7)
                        assemblage.phases[k].compositional_uncertainties[:6] = np.copy(sig_c)
                        assemblage.phases[k].compositional_uncertainties[6] = sig_f*c[2]

                    elif isinstance(assemblage.phases[k], burnman.Mineral):
                        pass
                    else:
                        raise Exception('phase not recognised: {0}'.format(assemblage.phases[k].name))

                burnman.processanalyses.compute_and_set_phase_compositions(assemblage)

                assemblage.stored_compositions = ['composition not assigned']*len(chamber_indices)
                for k in range(len(chamber_indices)):
                    if (assemblage.phases[k] is solutions['gt']
                          or assemblage.phases[k] is child_solutions['xna_gt']
                          or assemblage.phases[k] is child_solutions['xmj_gt']):
                        Fe3 = assemblage.phases[k].molar_fractions[assemblage.phases[k].endmember_names.index('andr')]
                        print([ph.name for ph in assemblage.phases])
                        print(Fe3/assemblage.phases[k].formula['Fe'])
                    try:
                        assemblage.stored_compositions[k] = (assemblage.phases[k].molar_fractions,
                                                             assemblage.phases[k].molar_fraction_covariances)
                    except AttributeError:
                        pass

                #assemblage.set_state(*assemblage.nominal_state)
                #print(assemblage.experiment_id, [ph.name for ph in assemblage.phases],
                #      assemblage_affinity_misfit(assemblage))
                Beyer_et_al_2019_NCFMASO_assemblages.append(assemblage)
    #exit()
    return Beyer_et_al_2019_NCFMASO_assemblages
