import numpy as np
import burnman

def get_assemblages(mineral_dataset):
    endmembers = mineral_dataset['endmembers']
    solutions = mineral_dataset['solutions']
    child_solutions = mineral_dataset['child_solutions']


    # Garnet-olivine partitioning data
    with open('data/Nakajima_FR_2012_FM_bdg_fper.dat', 'r') as f:
        bdg_fper_data = [line.split() for line in f if line.split() != [] and line[0] != '#']


    Nakajima_FR_2012_assemblages = []

    # N.B. P in GPa
    for run_id, mix, PGPa, TK, t, Febdg, Feerrbdg, Fefper, Feerrfper, _, _, _, _ in bdg_fper_data:

        p_mbdg = 1. - float(Febdg)
        p_per = 1. - float(Fefper)
        bdg_cov = float(Feerrbdg)*float(Feerrbdg) # sqrt2
        fper_cov = float(Feerrfper)*float(Feerrfper) # sqrt2


        assemblage = burnman.Composite([child_solutions['mg_fe_bdg'],
                                        solutions['mw']]) # in metallic Fe, low Fe3+ in bdg

        assemblage.experiment_id = 'Nakajima_FR_2012_{0}'.format(run_id)
        assemblage.nominal_state = np.array([float(PGPa)*1.e9, float(TK)]) # CONVERT PRESSURE TO GPA
        assemblage.state_covariances = np.array([[1.e9*1.e9, 0.], [0., 100.]]) # 1 GPa uncertainty

        child_solutions['mg_fe_bdg'].set_composition(np.array([p_mbdg, 1. - p_mbdg]))
        solutions['mw'].set_composition(np.array([p_per, 1. - p_per]))

        child_solutions['mg_fe_bdg'].molar_fraction_covariances = np.array([[bdg_cov, -bdg_cov],
                                                                            [-bdg_cov, bdg_cov]])
        solutions['mw'].molar_fraction_covariances = np.array([[fper_cov, -fper_cov],
                                                    [-fper_cov, fper_cov]])

        assemblage.stored_compositions = [(assemblage.phases[k].molar_fractions,
                                           assemblage.phases[k].molar_fraction_covariances)
                                          for k in range(2)]

        Nakajima_FR_2012_assemblages.append(assemblage)

    return Nakajima_FR_2012_assemblages
