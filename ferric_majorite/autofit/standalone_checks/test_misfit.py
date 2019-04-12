from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from scipy.optimize import minimize, fsolve
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))
import burnman
from burnman.solidsolution import SolidSolution as Solution
from burnman.processanalyses import compute_and_set_phase_compositions, assemblage_affinity_misfit
from burnman.equilibrate import equilibrate


from input_dataset import * 

# Test

composition = {'Fe': 0.3, 'Mg': 1.7, 'Si': 0.8, 'O': 3.6}
assemblage = burnman.Composite([fper, ol, wad], [0.2, 0.4, 0.4])
fper.guess = np.array([0.85, 0.15])
ol.guess = np.array([0.87, 0.13])
wad.guess = np.array([0.83, 0.17])
for m in [fper, ol, wad]:
    m.set_composition(m.guess)
assemblage.set_state(13.0e9, 1673.15)

equality_constraints = [('P', 13.7e9), ('T', 1673.15)]
sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False,
                        initial_composition_from_assemblage=True)
print(assemblage)

assemblage.experiment_id = 'TEST'
assemblage.nominal_state = np.array([assemblage.pressure + 1.e8,
                                     assemblage.temperature])

assemblage.state_covariances = np.array([[1.e8*1.e8, 0.],[0., 10.*10]])
                
for k in range(3):
    assemblage.phases[k].fitted_elements = ['Mg', 'Fe']
    
    assemblage.phases[k].composition = assemblage.phases[k].molar_fractions
    assemblage.phases[k].compositional_uncertainties = np.array([0.01, 0.01])
                    
burnman.processanalyses.compute_and_set_phase_compositions(assemblage)
                
assemblage.stored_compositions = [(assemblage.phases[k].molar_fractions,
                                   assemblage.phases[k].molar_fraction_covariances)
                                  for k in range(3)]
            
assemblage.set_state(*assemblage.nominal_state)


c_diff = np.array([[0.005, -0.005],
                   [0.002, -0.002],
                   [-0.003, 0.003]])
for k in range(3):
    molar_fractions, assemblage.phases[k].molar_fraction_covariances = assemblage.stored_compositions[k]
    assemblage.phases[k].set_composition(molar_fractions + c_diff[k])


print(assemblage_affinity_misfit(assemblage))
