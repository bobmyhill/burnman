import numpy as np
from fitting_functions import equilibrium_order
import burnman

def get_assemblages(mineral_dataset):
    endmembers = mineral_dataset['endmembers']
    solutions = mineral_dataset['solutions']
    child_solutions = mineral_dataset['child_solutions']


    # Garnet-olivine partitioning data
    with open('data/Seckendorff_ONeill_1992_ol_opx.dat', 'r') as f:
        ol_opx_data = [line.split() for line in f if line.split() != [] and line[0] != '#']


    Seckendorff_ONeill_1992_assemblages = []

    # N.B. P in GPa
    for run_id, P, T, phase1, Fe1, Feerr1, phase2, Fe2, Feerr2 in ol_opx_data:

        pressure = float(P)*1.e9
        temperature = float(T)

        p_fo = 1. - float(Fe1)
        p_en = 1. - float(Fe2)
        ol_cov = float(Feerr1)*float(Feerr1) # sqrt2
        opx_cov = float(Feerr2)*float(Feerr2) # sqrt2


        assemblage = burnman.Composite([solutions['ol'],
                                        child_solutions['mg_fe_opx']])

        assemblage.experiment_id = 'Seckendorff_ONeill_1992_{0}'.format(run_id)
        assemblage.nominal_state = np.array([pressure, temperature]) # CONVERT PRESSURE TO GPA
        assemblage.state_covariances = np.array([[0.1e9*0.1e9, 0.], [0., 100.]])

        solutions['ol'].set_composition(np.array([p_fo, 1. - p_fo]))
        child_solutions['mg_fe_opx'].set_composition(np.array([p_en, 1. - p_en, 0.]))

        solutions['ol'].molar_fraction_covariances = np.array([[ol_cov, -ol_cov],
                                                               [-ol_cov, ol_cov]])


        # The following adjusts compositions to reach equilibrium
        child_solutions['mg_fe_opx'].set_state(pressure, temperature)
        equilibrium_order(child_solutions['mg_fe_opx'])

        child_solutions['mg_fe_opx'].molar_fraction_covariances = np.array([[opx_cov, -opx_cov, -opx_cov],
                                                                            [-opx_cov, opx_cov, -opx_cov],
                                                                            [-opx_cov, -opx_cov, opx_cov]])

        assemblage.stored_compositions = [(assemblage.phases[k].molar_fractions,
                                           assemblage.phases[k].molar_fraction_covariances)
                                          for k in range(2)]

        Seckendorff_ONeill_1992_assemblages.append(assemblage)

    return Seckendorff_ONeill_1992_assemblages
