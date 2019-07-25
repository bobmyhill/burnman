import numpy as np
import burnman

def get_assemblages(mineral_dataset):
    endmembers = mineral_dataset['endmembers']
    solutions = mineral_dataset['solutions']
    child_solutions = mineral_dataset['child_solutions']


    # Garnet-pyroxene partitioning data
    with open('data/Perkins_Newton_1980_CMAS_opx_cpx_gt.dat', 'r') as f:
        opx_cpx_gt_data = [line.split() for line in f if line.split() != [] and line[0] != '#']


    Perkins_Newton_1980_CMAS_assemblages = []

    for datum in opx_cpx_gt_data:
        if len(datum) == 19: # just opx and cpx
            assemblage = burnman.Composite([child_solutions['oen_mgts_odi'],
                                            child_solutions['di_cen_cats']])
        elif len(datum) == 21: # opx, cpx, gt
            assemblage = burnman.Composite([child_solutions['oen_mgts_odi'],
                                            child_solutions['di_cen_cats'],
                                            child_solutions['py_gr_gt']])
            child_solutions['py_gr_gt'].fitted_elements = ['Mg', 'Ca']
            Mggt = (float(datum[19]) + float(datum[20]))/200.
            Mggterr = (float(datum[20]) - float(datum[19]))/200.
            child_solutions['py_gr_gt'].composition = np.array([Mggt, 1. - Mggt])
            child_solutions['py_gr_gt'].compositional_uncertainties = np.array([Mggterr, Mggterr])
        else:
            raise Exception('Wrong number of columns')


        pressure = float(datum[1])*1.e8 # CONVERT Pkbar TO Pa
        temperature = float(datum[2]) + 273.15 #  TC to TK
        sig_p = 0.1e9 + pressure/20.

        assemblage.experiment_id = 'Perkins_et_al_1981_MAS_{0}'.format(datum[0])
        assemblage.nominal_state = np.array([pressure, temperature])
        assemblage.state_covariances = np.array([[sig_p*sig_p, 0.],
                                                 [0., 100.]])


        c_vector = np.array(list(map(float, datum[3:19])))
        cav = (c_vector[:8] + c_vector[8:])/2.
        cerr = np.abs((c_vector[8:] - c_vector[:8])/2.) + 0.001 # additional uncertainty

        child_solutions['oen_mgts_odi'].fitted_elements = ['Ca', 'Mg', 'Al', 'Si']
        child_solutions['oen_mgts_odi'].composition = cav[:4]
        child_solutions['oen_mgts_odi'].compositional_uncertainties = cerr[:4]
        child_solutions['di_cen_cats'].fitted_elements = ['Ca', 'Mg', 'Al', 'Si']
        child_solutions['di_cen_cats'].composition = cav[4:]
        child_solutions['di_cen_cats'].compositional_uncertainties = cerr[4:]

        burnman.processanalyses.compute_and_set_phase_compositions(assemblage)


        assemblage.stored_compositions = [(assemblage.phases[k].molar_fractions,
                                           assemblage.phases[k].molar_fraction_covariances)
                                          for k in range(len(assemblage.phases))]

        Perkins_Newton_1980_CMAS_assemblages.append(assemblage)

    return Perkins_Newton_1980_CMAS_assemblages
