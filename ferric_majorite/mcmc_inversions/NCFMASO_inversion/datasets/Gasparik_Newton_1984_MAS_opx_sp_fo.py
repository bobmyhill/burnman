import numpy as np

from input_dataset import *


# Garnet-pyroxene partitioning data
with open('data/Gasparik_Newton_1984_MAS_opx_sp_fo.dat', 'r') as f:
    opx_sp_fo_data = [line.split() for line in f if line.split() != [] and line[0] != '#']


Gasparik_Newton_1984_MAS_assemblages = []

for run_id, mix, TC, Pkbar, t, N, Mg, Al, Si in opx_sp_fo_data:

    assemblage = burnman.Composite([child_solutions['oen_mgts'],
                                    endmembers['sp'], endmembers['fo']])
    assemblage.experiment_id = 'Gasparik_Newton_1984_MAS_{0}'.format(run_id)
    assemblage.nominal_state = np.array([float(Pkbar)*1.e8,
                                         float(TC)+273.15]) # CONVERT P TO PA, T to K
    assemblage.state_covariances = np.array([[0.1e9*0.1e9, 0.], [0., 100.]])


    child_solutions['oen_mgts'].fitted_elements = ['Mg', 'Al', 'Si']
    child_solutions['oen_mgts'].composition = np.array([float(Mg), float(Al), float(Si)])
    child_solutions['oen_mgts'].compositional_uncertainties = np.array([0.02, 0.02, 0.02])

    burnman.processanalyses.compute_and_set_phase_compositions(assemblage)


    assemblage.stored_compositions = ['composition not assigned']*len(assemblage.phases)
    assemblage.stored_compositions[0] = (assemblage.phases[0].molar_fractions,
                                         assemblage.phases[0].molar_fraction_covariances)

    Gasparik_Newton_1984_MAS_assemblages.append(assemblage)
