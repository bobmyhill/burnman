# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.

"""
example_perplex
---------------

This minimal example demonstrates how burnman can be used to read and interrogate
a PerpleX tab file (as produced by burnman/misc/create_burnman_readable_perplex_table.py 

*Uses:*

* :doc:`PerplexMaterial`



*Demonstrates:*

* Use of PerplexMaterial


"""
import sys
import os
import numpy as np

sys.path.insert(1, os.path.abspath('..'))

import burnman
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter

def pad_ndarray_inverse_mirror(array, padding):
    padded_shape = [n + 2*padding[i] for i, n in enumerate(array.shape)]
    padded_array = np.zeros(padded_shape)

    array_indices = []
    for idx, v in np.ndenumerate(array):
        idx = tuple([idx[i] + padding[i] for i in range(len(padding))])
        padded_array[idx] = v
        array_indices.append(idx)
    padded_indices = [idx for idx, v in np.ndenumerate(padded_array) if idx not in array_indices]
    
    edge_indices = tuple([tuple([np.min([np.max([axis_idx, padding[dimension]]), padded_array.shape[dimension] - padding[dimension] - 1])
                                 for dimension, axis_idx in enumerate(idx)]) for idx in padded_indices])
    mirror_indices = tuple([tuple([2*edge_indices[i][j] - padded_indices[i][j] for j in range(len(array.shape))]) for i in range(len(padded_indices))])

    for i, idx in enumerate(padded_indices):
        padded_array[idx] = 2.*padded_array[edge_indices[i]] - padded_array[mirror_indices[i]]

    return padded_array
    
                         
def smooth_gridded_property(gridded_property, pressures, temperatures,
                            pressure_stdev, temperature_stdev, truncate=4.0):
    
    pp, TT = np.meshgrid(pressures, temperatures)
    property_grid = rock.evaluate([gridded_property], pp, TT)[0]

    pressure_resolution = pressures[1] - pressures[0] 
    temperature_resolution = temperatures[1] - temperatures[0]

    sigma = (temperature_stdev/temperature_resolution,
             pressure_stdev/pressure_resolution)

    padding = (int(np.ceil(4.*sigma[0])), int(np.ceil(4.*sigma[1])))
    padded_property_grid = pad_ndarray_inverse_mirror(property_grid, padding)

    smoothed_padded_property_grid = gaussian_filter(padded_property_grid,
                                                    sigma=sigma)

    smoothed_property_grid = smoothed_padded_property_grid[padding[0]:padding[0] + property_grid.shape[0],
                                                           padding[1]:padding[1] + property_grid.shape[1]]
    
    return pp, TT,  smoothed_property_grid



rock=burnman.PerplexMaterial('in23_2.tab')
pressures = np.linspace(20.e9, 30.e9, 101)
temperatures = np.linspace(1600., 1700., 101)

pressure_stdev = 1.e8
temperature_stdev = 10.
pp, TT, smoothed_entropies = smooth_gridded_property('S', pressures, temperatures,
                                                     pressure_stdev, temperature_stdev)

entropies = rock.evaluate(['S'], pp, TT)[0]

plt.plot(pp[0]/1.e9, smoothed_entropies[0])
plt.plot(pp[0]/1.e9, entropies[0])
plt.xlabel('Pressure (GPa)')
plt.ylabel('Entropy (J/K/mol)')
plt.show()


