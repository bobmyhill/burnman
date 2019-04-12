import numpy as np

from input_dataset import *


# Garnet-pyroxene partitioning data
with open('data/Perkins_et_al_1981_MAS_py_opx.dat', 'r') as f:
    py_opx_data = [line.split() for line in f if line.split() != [] and line[0] != '#']


Perkins_et_al_1981_MAS_assemblages = []

for run_id, PGPa, Perr, TK, Terr, Al_fraction in py_opx_data:

    assemblage = burnman.Composite([child_solutions['oen_mgts'], py])
    assemblage.experiment_id = 'Perkins_et_al_1981_MAS_{0}'.format(run_id)
    assemblage.nominal_state = np.array([float(PGPa)*1.e9, float(TK)]) # CONVERT P TO Pa
    assemblage.state_covariances = np.array([[float(Perr)*float(Perr)*1.e18, 0.],
                                             [0., float(Terr)*float(Terr)]])


    child_solutions['oen_mgts'].fitted_elements = ['Mg', 'Al']
    child_solutions['oen_mgts'].composition = np.array([(1. - float(Al_fraction))/2.,
                                                        float(Al_fraction)])
    child_solutions['oen_mgts'].compositional_uncertainties = np.array([0.025, 0.005])
    
    burnman.processanalyses.compute_and_set_phase_compositions(assemblage)

    
    assemblage.stored_compositions = ['composition not assigned']*len(assemblage.phases)
    assemblage.stored_compositions[0] = (assemblage.phases[0].molar_fractions,
                                         assemblage.phases[0].molar_fraction_covariances)

    Perkins_et_al_1981_MAS_assemblages.append(assemblage)
