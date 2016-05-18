from __future__ import absolute_import

# Here we import standard python modules that are required for
# usage of BurnMan.  In particular, numpy is used for handling
# numerical arrays and mathematical operations on them, and
# matplotlib is used for generating plots of results of calculations
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import minerals


from SLB_andradite import andradite
andr = andradite()
py = minerals.SLB_2011.pyrope()
alm = minerals.SLB_2011.almandine()
gr = minerals.SLB_2011.grossular()
rw = minerals.SLB_2011.mg_fe_ringwoodite()
    
# Governing equations
# Fe + Fe3Fe2Si3O12 -> 3.Fe2SiO4
# 3.Mg2SiO4 + 2.Fe3Al2Si3O12 -> 3.Fe2SiO4 + 2.Mg3Al2Si3O12 


# If we know the composition of FeS melt and ringwoodite, we
# can calculate the activity of skiagite.

# Additional constraint might be a certain bulk oxygen content.
# For this, we would need to know the amount of iron in the other phases...

'''
Endmembers for garnet (10 symmetric interaction parameters)
pyrope
almandine
grossular
andradite
mg-majorite

A ringwoodite-garnet majorite field is often present at ~18 GPa, with wuestite becoming stable at higher temperature and Ca-perovskite at higher pressure

Thus, it seems reasonable to take solid solutions for ringwoodite and garnet only.
'''


class garnet(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name = 'garnet'
        self.type = 'symmetric'
        self.endmembers = [[minerals.SLB_2011.pyrope(), '[Mg]3[Al][Al]Si3O12'],
                           [minerals.SLB_2011.almandine(), '[Fe]3[Al][Al]Si3O12'],
                           [minerals.SLB_2011.grossular(), '[Ca]3[Al][Al]Si3O12'],
                           [andradite(), '[Ca]3[Fe][Fe]Si3O12'],
                           [minerals.SLB_2011.mg_majorite(), '[Mg]3[Mg][Si]Si3O12'],
                           [minerals.SLB_2011.jd_majorite(), '[Na2/3Al1/3]3[Al][Si]Si3O12']]

        
        self.energy_interaction = [[0.0, 30.e3, 0.0, 21.20278e3, 0.0],
                                   [0.0, 0.0, 0.0, 0.0],
                                   [0.0, 57.77596e3, 0.0],
                                   [57.77596e3, 0.0],
                                   [0.0]]

        burnman.SolidSolution.__init__(self, molar_fractions)

gt = garnet()

gt.set_composition([1.0, 0., 0., 0., 0., 0.])
gt.set_state(1.e9, 300.)
print(gt.molar_mass)

composition = burnman.processchemistry.component_to_atom_fractions({'Na2O': 0.5, 'CaO': 2.5, 'FeO': 18.4, 'MgO': 32.6, 'Al2O3': 2.0, 'SiO2': 44.0}, 'weight')
print(composition)

assemblage = burnman.Composite([sulfide_melt, gt, rw])
