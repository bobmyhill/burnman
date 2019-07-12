import numpy as np
import burnman

def get_assemblages(mineral_dataset):
    endmembers = mineral_dataset['endmembers']
    solutions = mineral_dataset['solutions']
    child_solutions = mineral_dataset['child_solutions']


    # bdg-fper-stv partitioning data
    with open('data/Tange_TNFS_2009_bdg_fper_stv.dat', 'r') as f:
        bdg_fper_stv_data = [line.split() for line in f if line.split() != [] and line[0] != '#']


    Tange_TNFS_2009_FMS_assemblages = []

    for i, datum in enumerate(bdg_fper_stv_data):

        run_id, PGPa, TC, ffpv, sig_fpv, ffper, sig_fper = datum

        pressure = float(PGPa)*1.e9
        temperature = float(TC) + 273.15
        pv_fractions = np.array([1. - float(ffpv), float(ffpv)])
        var_fpv = np.power(float(sig_fpv), 2.)
        pv_covariances = np.array([[var_fpv, -var_fpv],
                                   [-var_fpv, var_fpv]])

        fper_fractions = np.array([1. - float(ffper), float(ffper)])
        var_fper = np.power(float(sig_fper), 2.)
        fper_covariances = np.array([[var_fper, -var_fper],
                                     [-var_fper, var_fper]])

        assemblage = burnman.Composite([child_solutions['mg_fe_bdg'],
                                        solutions['mw'],
                                        endmembers['stv']])

        assemblage.experiment_id = 'Tange_2009_FMS_{0}'.format(run_id)
        assemblage.nominal_state = np.array([pressure, temperature])
        assemblage.state_covariances = np.array([[1.e9*1.e9, 0.], [0., 100.]])

        assemblage.stored_compositions = [(pv_fractions, pv_covariances),
                                          (fper_fractions, fper_covariances),
                                          ['composition not assigned']]

        Tange_TNFS_2009_FMS_assemblages.append(assemblage)

    return Tange_TNFS_2009_FMS_assemblages
