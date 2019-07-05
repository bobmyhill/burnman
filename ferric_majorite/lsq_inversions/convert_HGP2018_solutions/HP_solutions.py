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


# clinopyroxene in NCFMASO system: (Na,Ca,Fe,Mg)(Mg,Fe,Fe3+,Al3+)(Al3+,Si)2O6
cpx_array = independent_endmember_occupancies_from_charge_balance([[1,2,2,2], [2,2,3,3], [6,8]], 12, as_fractions=False)
print(cpx_array)

# orthopyroxene in CFMASO system: (Ca,Fe,Mg)(Mg,Fe,Fe3+,Al3+)(Al3+,Si)2O6
opx_array = independent_endmember_occupancies_from_charge_balance([[2,2,2], [2,2,3,3], [6,8]], 12, as_fractions=False)
print(opx_array)

# spinel in FMASO system: (Mg,Fe,Fe3+,Al3+,Si)(Mg,Fe,Fe3+,Al3+)2O4
spinel_array = independent_endmember_occupancies_from_charge_balance([[2,2,3,3,4], [4,4,6,6]], 8, as_fractions=False)
print(spinel_array)

class NCFMASO_clinopyroxene(SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name = 'clinopyroxene (NCFMASO)'
        self.endmembers = [[HGP_2018_ds633.di(),   '[Mg][Ca][Si]2O6'],
                           [cfs(),  '[Fe][Fe][Si]2O6'],
                           [HGP_2018_ds633.cats(), '[Al][Ca][Si1/2Al1/2]2O6'],
                           [cess(), '[Fef][Ca][Si1/2Al1/2]2O6'],
                           [HGP_2018_ds633.jd(),   '[Al][Na][Si]2O6'],
                           [cen(),  '[Mg][Mg][Si]2O6'],
                           [cfm(),  '[Mg][Fe][Si]2O6']] # note cfm ordered endmember 
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
      

class FMASO_spinel(SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name = 'order-disorder spinel (FMASO)' # fake multiplicity of 1 for oct sites
        self.endmembers = [[HGP_2018_ds633.sp(),   '[Mg][Al]AlO4'], # ordered in ds633
                           [isp(),   '[Al][Mg1/2Al1/2]Mg1/2Al1/2O4'],
                           [HGP_2018_ds633.herc(), '[Fe][Al]AlO4'], # ordered in ds633
                           [iherc(), '[Al][Fe1/2Al1/2]Fe1/2Al1/2O4'],
                           [nmt(),   '[Fe][Fef]FefO4'],
                           [imt(),   '[Fef][Fe1/2Fef1/2]Fe1/2Fef1/2O4'],
                           [HGP_2018_ds633.mrw(),  '[Si][Mg]MgO4']]
        self.solution_type = 'symmetric'
        self.energy_interaction = [[-8.2e3, 3.5e3, -13.e3, 43.2e3, 49.1e3, 0.],
                                   [4.4e3, -6.0e3, 36.8e3, 20.0e3, 0.],
                                   [-8.2e3, 18.1e3, 49.0e3, 0.],
                                   [-4.0e3, 7.6e3, 0.],
                                   [18.1e3, 0.],
                                   [0.]]
        SolidSolution.__init__(self, molar_fractions=molar_fractions)

class isp(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'inverse spinel',
                                 mineral_list = [HGP_2018_ds633.sp()],
                                 molar_amounts = [1.],
                                 free_energy_adjustment=[23.6e3, 5.76303, 0.])
    
class iherc(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'inverse hercynite',
                                 mineral_list = [HGP_2018_ds633.herc()],
                                 molar_amounts = [1.],
                                 free_energy_adjustment=[23.6e3, 5.76303, 0.])


# I DON'T QUITE UNDERSTAND THE NEXT TWO MAKE DEFINITIONS
class nmt(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'normal magnetite',
                                 mineral_list = [HGP_2018_ds633.mt()],
                                 molar_amounts = [1.],
                                 free_energy_adjustment=[0., -5.76303, 0.])
class imt(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'inverse magnetite',
                                 mineral_list = [HGP_2018_ds633.mt()],
                                 molar_amounts = [1.],
                                 free_energy_adjustment=[0.3e3, 0., 0.])

class NCFMASO_orthopyroxene(SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name = 'orthopyroxene (NCFMASO)' # fake T-site multiplicity
        self.endmembers = [[HGP_2018_ds633.en(),   '[Mg][Mg][Si]0.5Si1.5O6'],
                           [HGP_2018_ds633.fs(),   '[Fe][Fe][Si]0.5Si1.5O6'],
                           [fm(),   '[Fe][Mg][Si]0.5Si1.5O6'], # ordered phase
                           [odi(),  '[Mg][Ca][Si]0.5Si1.5O6'],
                           [HGP_2018_ds633.mgts(), '[Al][Mg][Si1/2Al1/2]0.5Si1.5O6'],
                           [mess(), '[Fef][Mg][Si1/2Al1/2]0.5Si1.5O6'],
                           [ojd(), '[Al][Na][Si]0.5Si1.5O6']] 
        self.solution_type = 'asymmetric'
        self.alphas = [1., 1., 1., 1.2, 1., 1., 1., 1.2]
        self.energy_interaction = [[7.0e3, 4.0e3, 32.2e3, 12.5e3, 8.0e3, 35.e3],
                                   [4.e3, 25.54e3, 11.e3, 10.e3, 35.e3],
                                   [25.54e3, 15.e3, 12.e3, 35.e3],
                                   [75.5e3, 20.e3, 35.e3],
                                   [2.e3, 7.e3],
                                   [-11.e3]]
        self.volume_interaction = [[0., 0., 0.12e-5, -0.04e-5, 0., 0.],
                                   [0., 0.084e-5, -0.15e-5, 0., 0.],
                                   [0.084e-5, -0.15e-5, 0., 0.],
                                   [-0.84e-5, 0., 0.],
                                   [0., 0.],
                                   [0.]]
        
        SolidSolution.__init__(self, molar_fractions=molar_fractions)
   


class fm(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'ordered ferroenstatite',
                                 mineral_list = [HGP_2018_ds633.en(),
                                                 HGP_2018_ds633.fs()],
                                 molar_amounts = [0.5, 0.5],
                                 free_energy_adjustment=[-6.6e3, 0., 0.])
class odi(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'orthodiopside',
                                 mineral_list = [HGP_2018_ds633.di()],
                                 molar_amounts = [1.],
                                 free_energy_adjustment=[-0.1e3, -0.211, 0.005e-5]) # note sign of *entropy* change.

class mess(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'ferrienstatite',
                                 mineral_list = [HGP_2018_ds633.mgts(),
                                                 HGP_2018_ds633.acm(),
                                                 HGP_2018_ds633.jd()],
                                 molar_amounts = [1., 1., -1.],
                                 free_energy_adjustment=[4.8e3, 0., -0.089e-5])

class ojd(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'orthojadeite',
                                 mineral_list = [HGP_2018_ds633.jd()],
                                 molar_amounts = [1.],
                                 free_energy_adjustment=[18.8e3, 0., 0.])



spinel = FMASO_spinel()
cpx = NCFMASO_clinopyroxene()
opx = NCFMASO_orthopyroxene()

        
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
# di, hed, cen, cats, jd, aeg, cfs
# hed = di + cfs - cfm
# aeg/acm = jd - cats + cess
new_cpx = transform_solution_to_new_basis(cpx, [[1, 0, 0, 0, 0, 0, 0],
                                                [1, 1, 0, 0, 0, 0, -1],
                                                [0, 0, 0, 0, 0, 1, 0],
                                                [0, 0, 1, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 1, 0, 0],
                                                [0, 0, -1, 1, 1, 0, 0],
                                                [0, 1, 0, 0, 0, 0, 0]],
                                        solution_name='new cpx')


print(new_cpx.energy_interaction)
print(new_cpx.endmembers[1][0].params,
      new_cpx.endmembers[1][0].mixture.endmembers,
      new_cpx.endmembers[1][0].property_modifiers)
hed = HGP_2018_ds633.hed()
aeg = HGP_2018_ds633.acm()

temperatures = np.linspace(300., 1800., 101)
pressures = temperatures*0. + 1.e5
plt.plot(temperatures, (hed.evaluate(['gibbs'], pressures, temperatures)[0] -
                        new_cpx.endmembers[1][0].evaluate(['gibbs'], pressures, temperatures)[0])/1.e3,
         label='hed')


#cpx.set_composition([1., 1., 0., 0., 0., 0., -1.])
#plt.plot(temperatures, (new_cpx.endmembers[1][0].evaluate(['gibbs'], pressures, temperatures)[0] -
#                        cpx.evaluate(['gibbs'], pressures, temperatures)[0]),
#         linestyle=':', linewidth=3)

plt.plot(temperatures, (aeg.evaluate(['gibbs'], pressures, temperatures)[0] -
                        new_cpx.endmembers[5][0].evaluate(['gibbs'], pressures, temperatures)[0])/1.e3,
         label='acm')


#cpx.set_composition([0., 0., -1., 1., 1., 0., 0.])
#plt.plot(temperatures,(new_cpx.endmembers[5][0].evaluate(['gibbs'], pressures, temperatures)[0] -
#                       cpx.evaluate(['gibbs'], pressures, temperatures)[0]),
#         linestyle=':', linewidth=3)

plt.ylabel('$\\Delta$G (endmember - solution; kJ/mol)')
plt.xlabel('Temperature (K)')
plt.savefig('cpx_dependent_endmember_comparison.pdf')
plt.legend()
plt.show()
exit()
