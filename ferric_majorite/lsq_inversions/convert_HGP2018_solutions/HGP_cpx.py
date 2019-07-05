# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""

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
from burnman.minerals import HGP_2018_ds633
from burnman.solidsolution import SolidSolution
from burnman.combinedmineral import CombinedMineral

from burnman.solutionbases import transform_solution_to_new_basis, feasible_solution_in_component_space, feasible_endmember_occupancies_from_charge_balance, independent_endmember_occupancies_from_charge_balance, dependent_endmember_site_occupancies, dependent_endmember_sums, site_occupancies_to_strings, generate_complete_basis
from burnman.processanalyses import compute_and_set_phase_compositions, assemblage_affinity_misfit




class NCFMASO_clinopyroxene(SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name = 'clinopyroxene (NCFMASO)'
        self.endmembers = [[HGP_2018_ds633.di(),   '[Mg][Ca][Si]2O6'],
                           [cfs(),  '[Fe][Fe][Si]2O6'],
                           [HGP_2018_ds633.cats(), '[Al][Ca][Si1/2Al1/2]2O6'],
                           [cess(), '[Fef][Ca][Si1/2Al1/2]2O6'],
                           [HGP_2018_ds633.jd(),   '[Al][Na][Si]2O6'],
                           [cen(),  '[Mg][Mg][Si]2O6'],
                           [cfm(),  '[Mg][Fe][Si]2O6']] # note cfm ordered endmember with Mg on the Al site
        self.solution_type = 'asymmetric'
        self.alphas = [1.2, 1.0, 1.9, 1.9, 1.2, 1.0, 1.0]
        self.energy_interaction = [[25.8e3, 13.0e3, 8.e3, 26.e3, 29.8e3, 20.6e3],
                                   [25.e3, 43.3e3, 24.e3, 2.3e3, 3.5e3],
                                   [2.e3, 6.e3, 45.2e3, 27.e3],
                                   [3.e3, 57.3e3, 45.3e3],
                                   [40.e3, 40.e3],
                                   [4.e3]]
        self.volume_interaction = [[-0.03e-5, -0.06e-5, 0., 0., -0.03e-5, -0.03e-5],
                                   [-0.1e-5, 0., 0., 0., 0.],
                                   [0., 0., -0.35e-5, -0.1e-5],
                                   [0., 0., 0.],
                                   [0., 0.],
                                   [0.]]
        SolidSolution.__init__(self, molar_fractions=molar_fractions)

class cfs(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'clinoferrosilite',
                                 mineral_list = [HGP_2018_ds633.fs()],
                                 molar_amounts = [1.],
                                 free_energy_adjustment=[2.1e3, 2., 0.045e-5])  
class cess(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'ferric diopside',
                                 mineral_list = [HGP_2018_ds633.cats(),
                                                 HGP_2018_ds633.acm(),
                                                 HGP_2018_ds633.jd()],
                                 molar_amounts = [1., 1., -1.],
                                 free_energy_adjustment=[-3.45e3, 0., 0.])  
class cen(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'clinoenstatite',
                                 mineral_list = [HGP_2018_ds633.en()],
                                 molar_amounts = [1.],
                                 free_energy_adjustment=[3.5e3, 2., 0.048e-5])

class cfm(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'ordered clinoferroenstatite',
                                 mineral_list = [HGP_2018_ds633.en(),
                                                 HGP_2018_ds633.fs()],
                                 molar_amounts = [0.5, 0.5],
                                 free_energy_adjustment=[-1.6e3, 2., 0.0465e-5])
      

cpx = NCFMASO_clinopyroxene()

        
# old_cpx
# di, cfs, cats, cess, jd, cen, cfm
#[[di(),   '[Mg][Ca][Si]2O6'],
# [cfs(),  '[Fe][Fe][Si]2O6'],
# [cats(), '[Al][Ca][Si1/2Al1/2]2O6'],
# [cess(), '[Fef][Ca][Si1/2Al1/2]2O6'],
# [jd(),   '[Al][Na][Si]2O6'],
# [cen(),  '[Mg][Mg][Si]2O6'],
# [cfm(),  '[Mg][Fe][Si]2O6']]


# new cpx
# di, hed, cen, cfs, cats, jd, aeg
# hed = di + cfs - cfm
# aeg/acm = jd - cats + cess
new_cpx = transform_solution_to_new_basis(cpx, [[1, 0, 0, 0, 0, 0, 0],
                                                [1, 1, 0, 0, 0, 0, -1],
                                                [0, 0, 0, 0, 0, 1, 0],
                                                [0, 1, 0, 0, 0, 0, 0],
                                                [0, 0, 1, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 1, 0, 0],
                                                [0, 0, -1, 1, 1, 0, 0]],
                                        solution_name='new cpx')


print(new_cpx.alphas)
print(new_cpx.energy_interaction)
print(new_cpx.volume_interaction)

print('hed properties')
print(new_cpx.endmembers[1][0].params,
      new_cpx.endmembers[1][0].mixture.endmembers,
      new_cpx.endmembers[1][0].property_modifiers)

print('aeg/acm properties')
print(new_cpx.endmembers[6][0].params,
      new_cpx.endmembers[6][0].mixture.endmembers,
      new_cpx.endmembers[6][0].property_modifiers)

