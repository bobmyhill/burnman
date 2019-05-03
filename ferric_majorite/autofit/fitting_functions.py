from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))
import burnman


def equilibrium_order(solution):
    """
    Finds the equilibrium molar fractions of the solution 
    with the composition given by the input molar fractions and
    at the input pressure and temperature.
    """
    assemblage = burnman.Composite([solution], [1.])
    assemblage.set_state(solution.pressure, solution.temperature)
    equality_constraints = [('P', solution.pressure), ('T', solution.temperature)]
    sol, prm = burnman.equilibrate(solution.formula, assemblage, equality_constraints,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False, verbose=False)
    if not sol.success:
        raise Exception('equilibrium state not found')
    
