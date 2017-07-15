# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU GPL v2 or later.


"""

example_solid_solution
----------------------
    
This example shows how to create different solid solution models and output
thermodynamic and thermoelastic quantities.

There are four main types of solid solution currently implemented in 
BurnMan:

1. Ideal solid solutions
2. Symmmetric solid solutions
3. Asymmetric solid solutions
4. Subregular solid solutions

These solid solutions can potentially deal with:

* Disordered endmembers (more than one element on a crystallographic site)
* Site vacancies
* More than one valence/spin state of the same element on a site

*Uses:*

* :doc:`mineral_database`
* :class:`burnman.solidsolution.SolidSolution`
* :class:`burnman.solutionmodel.SolutionModel`


*Demonstrates:*

* Different ways to define a solid solution
* How to set composition and state
* How to output thermodynamic and thermoelastic properties

"""
from __future__ import absolute_import

import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals

if __name__ == "__main__":
    
    class mg_fe_ca_garnet(burnman.SolidSolution):
        def __init__(self, molar_fractions=None):
            self.name='Dummy garnet'
            self.type='symmetric'
            self.endmembers = [[minerals.SLB_2011.pyrope(), '[Mg]3Al2Si3O12'],
                               [minerals.SLB_2011.almandine(), '[Fe]3Al2Si3O12'],
                               [minerals.SLB_2011.grossular(), '[Ca]3Al2Si3O12']]
            self.enthalpy_interaction=[[0.0e3,50.0e3],[50.0e3]]
            
            burnman.SolidSolution.__init__(self, molar_fractions)

    g=mg_fe_ca_garnet()

    # Check that (with the exception of the excess configurational entropy), the system is like a (py+alm) + gr binary

    p_gr = 0.4

    p_pys = np.linspace(0.1, 0.5, 5)

    for p_py in p_pys:
        p_alm = 0.6 - p_py
        g.set_composition([p_py, p_alm, p_gr])
        g.set_state(1.e5, 0.1)

        print g.excess_gibbs
