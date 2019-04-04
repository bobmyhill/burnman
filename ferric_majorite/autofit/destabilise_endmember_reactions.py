import numpy as np

from input_dataset import *


# Endmember reaction conditions
with open('data/destabilise_endmember_transitions.dat', 'r') as f:
    ds = [line.split() for line in f if line.split() != [] and line[0] != '#']
 
i = 0
destabilised_endmember_reaction_assemblages = []   
for d in ds:
    i+=1
    assemblage = burnman.Composite([endmembers[d[3]], endmembers[d[4]]])
    assemblage.experiment_id = 'endmember_rxn_{0}'.format(i)
    
    assemblage.nominal_state = np.array([float(d[0])*1.e9, float(d[2])])
    assemblage.state_covariances = np.array([[np.power(float(d[1])*1.e9, 2.), 0.], [0., 100.]])
    
    destabilised_endmember_reaction_assemblages.append(assemblage)


