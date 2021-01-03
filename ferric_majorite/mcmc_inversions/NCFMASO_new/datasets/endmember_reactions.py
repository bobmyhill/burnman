import numpy as np
import burnman


def get_assemblages(mineral_dataset):
    endmembers = mineral_dataset['endmembers']

    # Endmember reaction conditions
    with open('data/endmember_transitions.dat', 'r') as f:
        ds = [line.split() for line in f
              if line.split() != [] and line[0] != '#']

    i = 0
    endmember_reaction_assemblages = []
    for d in ds:
        i += 1
        assemblage = burnman.Composite([endmembers[d[j]]
                                        for j in range(4, len(d))])
        assemblage.experiment_id = 'endmember_rxn_{0}'.format(i)

        assemblage.nominal_state = np.array([float(d[0])*1.e9, float(d[2])])
        assemblage.state_covariances = np.array([[np.power(float(d[1])*1.e9,
                                                           2.), 0],
                                                 [0., np.power(float(d[3]),
                                                               2.)]])

        endmember_reaction_assemblages.append(assemblage)

    return endmember_reaction_assemblages
