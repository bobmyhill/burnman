import numpy as np

from input_dataset import *


# Garnet-pyroxene partitioning data
with open('data/Gasparik_Newton_1984_MAS_py_opx_sp_fo.dat', 'r') as f:
    py_opx_sp_fo_data = [line.split() for line in f if line.split() != [] and line[0] != '#']


Gasparik_Newton_1984_MAS_univariant_assemblages = []

for run_id, TC, Pkbar, Perr, xMgts in py_opx_sp_fo_data:

    assemblage = burnman.Composite([child_solutions['oen_mgts'], py, sp, fo])
    assemblage.experiment_id = 'Gasparik_Newton_1984_MAS_{0}'.format(run_id)
    assemblage.nominal_state = np.array([float(Pkbar)*1.e8,
                                         float(TC)+273.15]) # CONVERT P TO PA, T to K
    assemblage.state_covariances = np.array([[float(Perr)*float(Perr)*1.e16, 0.], [0., 100.]])
    
    
    assemblage.stored_compositions = ['composition not assigned']*len(assemblage.phases)
    assemblage.stored_compositions[0] = (np.array([1. - float(xMgts), float(xMgts)]),
                                         np.array([[1.e-4, -1.e-4],
                                                   [-1.e-4, 1.e-4]]))

    Gasparik_Newton_1984_MAS_univariant_assemblages.append(assemblage)
