# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit
# for the Earth and Planetary Sciences
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
from burnman.processanalyses import compute_and_set_phase_compositions
from burnman.processanalyses import assemblage_affinity_misfit


# Let's first create instances of the minerals in our sample
# and provide compositional uncertainties

# First, we create an olivine instance and
# assign atomic proportions
olivine = SLB_2011.mg_fe_olivine()
olivine.fitted_elements = ['Mg', 'Fe', 'Si']
olivine.composition = np.array([1.8, 0.2, 1.0])
olivine.compositional_uncertainties = np.array([0.1, 0.1, 0.1])



# Next we create a garnet instance for a sodium-free system
# based on SLB_2011 garnet (3-site NCFMAS-majorite garnet)
components = ['CaO', 'FeO', 'MgO', 'Al2O3', 'SiO2']
garnet = feasible_solution_in_component_space(SLB_2011.garnet(), components)

# Let's say that we have evidence that only 0.03 of the total
# C site is comprised of Si (out of a total possible 1.0)
# We can add site constraints like this using an underscore to
# specify the site for which we have constraints.
garnet.fitted_elements = ['Mg', 'Ca', 'Al', 'Si', 'Fe', 'Si_C']  # Si only on C
garnet.composition = np.array([1.64, 1.5, 1.85, 3.05, 0.01, 0.03])
garnet.compositional_uncertainties = np.array([0.1, 0.1, 0.1, 0.2, 0.01, 0.01])

# Finally, we create a quaartz instance
# (obviously out of equilibrium with olivine,
# so we should end up with a high misfit)
quartz = SLB_2011.quartz()

assemblage = burnman.Composite([garnet, olivine, quartz], [0.2, 0.3, 0.5])
assemblage.nominal_state = (10.e9, 500.)
assemblage.state_covariances = np.diag(np.array([1.e9*1.e9, 10.*10.]))

# Now let's prepare the assemblage by assigning covariances
# to each of the solid solutions
compute_and_set_phase_compositions(assemblage)

# Assign a state to the assemblage
assemblage.set_state(*assemblage.nominal_state)

# Calculate the misfit
print(assemblage_affinity_misfit(assemblage))
print(assemblage)

# We can do the same for an olivine-wadsleyite composition:
# This time, compositions are normalised to 1/2 a unit formula;
# the fitting routines automatically renormalise everything.
olivine = SLB_2011.mg_fe_olivine()
olivine.fitted_elements = ['Mg', 'Fe', 'Si']
olivine.composition = np.array([0.825, 0.18, 0.5])
olivine.compositional_uncertainties = np.array([0.05, 0.05, 0.05])

wadsleyite = SLB_2011.mg_fe_wadsleyite()
wadsleyite.fitted_elements = ['Mg', 'Fe', 'Si']
wadsleyite.composition = np.array([0.72, 0.28, 0.5])
wadsleyite.compositional_uncertainties = np.array([0.05, 0.05, 0.05])


assemblage = burnman.Composite([olivine, wadsleyite], [0.5, 0.5])
assemblage.nominal_state = (13.e9, 1673.)
assemblage.state_covariances = np.diag(np.array([1.e9*1.e9, 10.*10.]))

# Now let's prepare the assemblage by assigning covariances
# to each of the solid solutions
compute_and_set_phase_compositions(assemblage)

# Assign a state to the assemblage
assemblage.set_state(*assemblage.nominal_state)

# Calculate the misfit
print(assemblage_affinity_misfit(assemblage))
print(assemblage)
