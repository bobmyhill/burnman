# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2019 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
example_analytical_processing
-----------------------------
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman.minerals import SLB_2011
from burnman.solutionbases import feasible_solution_in_component_space
from burnman.processanalyses import compute_and_set_phase_compositions, assemblage_affinity_misfit


# Let's first create instances of the minerals in our sample and provide compositional uncertainties

components = ['Na2O', 'CaO', 'FeO', 'MgO', 'Al2O3', 'SiO2', 'TiO2'] # No Na, Fe3+, Cr3+ ...
garnet = feasible_solution_in_component_space(SLB_2011.garnet(), components)
print(garnet.endmember_formulae)

garnet.composition = {'Mg': 1.64, 'Ca': 1.5, 'Al': 1.85, 'Si': 3.05, 'Fe': 0.01, 'Na': 0.01}
garnet.compositional_uncertainties = {'Mg': 0.1, 'Ca': 0.1, 'Al': 0.1, 'Si': 0.2, 'Fe': 0.01, 'Na': 0.01}

olivine = SLB_2011.mg_fe_olivine()
olivine.composition = {'Mg': 1.8, 'Fe': 0.2, 'Si': 1.0}
olivine.compositional_uncertainties = {'Mg': 0.1, 'Fe': 0.1, 'Si': 0.1}

quartz = SLB_2011.quartz()

assemblage = burnman.Composite([garnet, olivine, quartz])
assemblage.nominal_state = 10.e9, 500.
assemblage.state_covariances = np.diag(np.array([1.e9*1.e9, 10.*10.])) 

# Now let's prepare the assemblage by assigning covariances to each of the solid solutions
compute_and_set_phase_compositions(assemblage)

# Assign a state to the assemblage
assemblage.set_state(*assemblage.nominal_state)

# Calculate the misfit
print(assemblage_affinity_misfit(assemblage))


