import numpy as np
import burnman


def get_assemblages(mineral_dataset):
    solutions = mineral_dataset['solutions']
    child_solutions = mineral_dataset['child_solutions']

    # olivine-spinel partitioning data
    with open('data/Tsujino_et_al_2019_FMS_wad_ring.dat', 'r') as f:
        wad_rw_data = [line.split() for line in f if line.split() != []
                      and line[0] != '#']

    Tsujino_et_al_2019_FMS_wad_ring_assemblages = []

    # N.B. P in GPa, T in K
    for datum in wad_rw_data:
        run_id, load, tel, TK, t, Mg1, Mgerr1, Mg2, Mgerr2, KD, KDe, MgOV, MgOVe, PTange, PTangeerr, PSpez, PSpezerr = datum

        p_mwd = float(Mg1) / 100.
        p_mrw = float(Mg2) / 100.

        ol_cov = float(Mgerr1)*float(Mgerr1) / 10000.
        sp_cov = float(Mgerr2)*float(Mgerr2) / 10000.

        assemblage = burnman.Composite([solutions['wad'],
                                        child_solutions['ring']])

        Pvar = np.power(float(PSpezerr)*1.e9, 2.)
        Tvar = 25.*25.

        assemblage.experiment_id = 'Tsujino_2019_FMS_{0}'.format(run_id)
        assemblage.nominal_state = np.array([float(PSpez)*1.e9,
                                             float(TK)])  # CONVERT PRESSURE TO GPA
        assemblage.state_covariances = np.array([[Pvar, 0.],
                                                 [0., Tvar]])

        solutions['wad'].set_composition(np.array([p_mwd, 1. - p_mwd]))
        child_solutions['ring'].set_composition(np.array([p_mrw, 1. - p_mrw]))

        solutions['wad'].molar_fraction_covariances = np.array([[ol_cov, -ol_cov],
                                                               [-ol_cov, ol_cov]])
        child_solutions['ring'].molar_fraction_covariances = np.array([[sp_cov, -sp_cov],
                                                                          [-sp_cov, sp_cov]])

        assemblage.stored_compositions = [(assemblage.phases[k].molar_fractions,
                                           assemblage.phases[k].molar_fraction_covariances)
                                          for k in range(2)]

        Tsujino_et_al_2019_FMS_wad_ring_assemblages.append(assemblage)
    return Tsujino_et_al_2019_FMS_wad_ring_assemblages
