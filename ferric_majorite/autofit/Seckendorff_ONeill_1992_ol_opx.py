import numpy as np

from input_dataset import *


# Garnet-olivine partitioning data
with open('data/Seckendorff_ONeill_1992_ol_opx.dat', 'r') as f:
    ol_opx_data = [line.split() for line in f if line.split() != [] and line[0] != '#']

    
Seckendorff_ONeill_1992_assemblages = []

# N.B. P in GPa
for run_id, P, T, phase1, Fe1, Feerr1, phase2, Fe2, Feerr2 in ol_opx_data:
    

    p_fo = 1. - float(Fe1)
    p_en = 1. - float(Fe2)
    ol_cov = float(Feerr1)*float(Feerr1) # sqrt2
    opx_cov = float(Feerr2)*float(Feerr2) # sqrt2
    
    
    assemblage = burnman.Composite([ol, child_solutions['oen_ofs']])
    
    assemblage.experiment_id = 'Seckendorff_ONeill_1992_{0}'.format(run_id)
    assemblage.nominal_state = np.array([float(P)*1.e9, float(T)]) # CONVERT PRESSURE TO GPA
    assemblage.state_covariances = np.array([[5.e7*5.e7, 0.], [0., 100.]])

    ol.set_composition(np.array([p_fo, 1. - p_fo]))
    child_solutions['oen_ofs'].set_composition(np.array([p_en, 1. - p_en]))

    ol.molar_fraction_covariances = np.array([[ol_cov, -ol_cov],
                                              [-ol_cov, ol_cov]])
    child_solutions['oen_ofs'].molar_fraction_covariances = np.array([[opx_cov, -opx_cov],
                                                                      [-opx_cov, opx_cov]])

    assemblage.stored_compositions = [(assemblage.phases[k].molar_fractions,
                                       assemblage.phases[k].molar_fraction_covariances)
                                      for k in range(2)]
    
    Seckendorff_ONeill_1992_assemblages.append(assemblage)
