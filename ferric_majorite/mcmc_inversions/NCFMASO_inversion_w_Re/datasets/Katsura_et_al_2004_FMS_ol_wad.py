import numpy as np
import burnman


def get_assemblages(mineral_dataset):
    solutions = mineral_dataset['solutions']
    child_solutions = mineral_dataset['child_solutions']

    # olivine-spinel partitioning data
    with open('data/Katsura_et_al_2004_FMS_ol_wad.dat', 'r') as f:
        ol_wad_data = [line.split() for line in f if line.split() != []
                      and line[0] != '#']

    Katsura_et_al_2004_FMS_ol_wad_assemblages = []

    # N.B. P in GPa, T in K
    for datum in ol_wad_data:
        run_id, TK, MgOV, MgOVe, PMatsui, PMatsuierr, tmin, Mg1, Mgerr1, Mg2, Mgerr2, KD, KDe = datum
        p_fo = float(Mg1) / 100.
        p_mwd = float(Mg2) / 100.

        ol_cov = float(Mgerr1)*float(Mgerr1) / 10000.
        wd_cov = float(Mgerr2)*float(Mgerr2) / 10000.

        assemblage = burnman.Composite([solutions['ol'],
                                        solutions['wad']])

        Pvar = np.power(float(PMatsuierr)*1.e9, 2.)
        Tvar = 50.*50.

        assemblage.experiment_id = 'Katsura_2004_FMS_{0}'.format(run_id)
        assemblage.nominal_state = np.array([float(PMatsui)*1.e9,
                                             float(TK)])  # CONVERT PRESSURE TO GPA
        assemblage.state_covariances = np.array([[Pvar, 0.],
                                                 [0., Tvar]])

        solutions['ol'].set_composition(np.array([p_fo, 1. - p_fo]))
        solutions['wad'].set_composition(np.array([p_mwd, 1. - p_mwd]))

        solutions['ol'].molar_fraction_covariances = np.array([[ol_cov, -ol_cov],
                                                               [-ol_cov, ol_cov]])
        solutions['wad'].molar_fraction_covariances = np.array([[wd_cov, -wd_cov],
                                                                [-wd_cov, wd_cov]])

        assemblage.stored_compositions = [(assemblage.phases[k].molar_fractions,
                                           assemblage.phases[k].molar_fraction_covariances)
                                          for k in range(2)]

        Katsura_et_al_2004_FMS_ol_wad_assemblages.append(assemblage)
    return Katsura_et_al_2004_FMS_ol_wad_assemblages
