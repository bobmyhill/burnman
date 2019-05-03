import numpy as np

from input_dataset import *


# Garnet-pyroxene partitioning data
with open('data/Carlson_Lindsley_1988_CMS_opx_cpx.dat', 'r') as f:
    opx_cpx_data = [line.split() for line in f if line.split() != [] and line[0] != '#']

set_runs = list(set([d[0] for d in opx_cpx_data]))


Carlson_Lindsley_1988_CMS_assemblages = []

print('WARNING: Carlson and Lindsley inversion does not yet include pen, pig')
for i, run in enumerate(set_runs):
    # for now, limit inversion to opx or cpx
    run_indices = [idx for idx, x in enumerate(opx_cpx_data)
                   if (x[0] == run and (x[3] == 'opx' or x[3] == 'cpx'))]
    if len(run_indices) > 1.:
        
        assemblage = burnman.Composite([child_solutions['oen_odi'],
                                        child_solutions['di_cen']]) # note order of endmembers!
        
        assemblage.experiment_id = 'Carlson_Lindsley_1988_CMS_{0}'.format(run)

        state = map(float, opx_cpx_data[run_indices[0]][1:3])
        assemblage.nominal_state = np.array([state[0]*1.e8,
                                             state[1]+273.15]) # CONVERT P TO PA, TC to TK
        assemblage.state_covariances = np.array([[5.e7*5.e7, 0.], [0., 100.]])
    
        assemblage.stored_compositions = ['composition not assigned']*2
    
        for idx in run_indices:
            datum = opx_cpx_data[idx]
            phase = datum[3]
            diav = (float(datum[4]) + float(datum[5]))/2.
            dierr = (float(datum[5]) - float(datum[4]))/2. + 0.005 # additional uncertainty
            divar = dierr*dierr
            if phase == 'opx':
                assemblage.stored_compositions[0] = (np.array([1. - diav, diav]), # oen_odi
                                                     np.array([[divar, -divar],
                                                               [-divar, divar]]))
                
            elif phase == 'cpx':
                assemblage.stored_compositions[1] = (np.array([diav, 1. - diav]), # di_cen
                                                     np.array([[divar, -divar],
                                                               [-divar, divar]]))
    
    Carlson_Lindsley_1988_CMS_assemblages.append(assemblage)
