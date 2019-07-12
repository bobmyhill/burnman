import numpy as np
import burnman


def get_assemblages(mineral_dataset):
    solutions = mineral_dataset['solutions']
    child_solutions = mineral_dataset['child_solutions']

    # olivine-spinel partitioning data
    with open('data/Jamieson_Roeder_1984_FMAS_ol_sp.dat', 'r') as f:
        ol_sp_data = [line.split() for line in f if line.split() != []
                      and line[0] != '#']

    Jamieson_Roeder_1984_FMAS_ol_sp_assemblages = []

    # N.B. P in GPa, T in K
    run_id = 0
    for P, T, Mg1, Mgerr1, Mg2, Mgerr2 in ol_sp_data:
        run_id += 1

        p_fo = float(Mg1) / 100.
        p_sp = float(Mg2) / 100.
        ol_cov = float(Mgerr1)*float(Mgerr1)/10000.  # sqrt2
        sp_cov = float(Mgerr2)*float(Mgerr2)/10000.  # sqrt2

        assemblage = burnman.Composite([solutions['ol'],
                                        child_solutions['sp_herc']])

        assemblage.experiment_id = 'Jamieson_Roeder_1984_FMAS_{0}'.format(run_id)
        assemblage.nominal_state = np.array([float(P)*1.e9, float(T)])  # CONVERT PRESSURE TO GPA
        assemblage.state_covariances = np.array([[1.e4*1.e4, 0.], [0., 100.]])

        solutions['ol'].set_composition(np.array([p_fo, 1. - p_fo]))
        child_solutions['sp_herc'].set_composition(np.array([p_sp, 1. - p_sp]))

        solutions['ol'].molar_fraction_covariances = np.array([[ol_cov, -ol_cov],
                                                               [-ol_cov, ol_cov]])
        child_solutions['sp_herc'].molar_fraction_covariances = np.array([[sp_cov, -sp_cov],
                                                                          [-sp_cov, sp_cov]])

        assemblage.stored_compositions = [(assemblage.phases[k].molar_fractions,
                                           assemblage.phases[k].molar_fraction_covariances)
                                          for k in range(2)]

        Jamieson_Roeder_1984_FMAS_ol_sp_assemblages.append(assemblage)

    return Jamieson_Roeder_1984_FMAS_ol_sp_assemblages
