from __future__ import absolute_import
from __future__ import print_function

import platform
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from scipy.optimize import fsolve
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))



import burnman
from burnman.minerals import JH_2015
from burnman.solutionbases import transform_solution_to_new_basis


fo = JH_2015.fo()
sp = JH_2015.sp()

"""
sp2 = burnman.minerals.HGP_2018_ds633.sp()
print(sp2)


print(sp.property_modifiers)
print(sp2.property_modifiers)
temperatures = np.linspace(300., 1800., 101)
pressures = temperatures*0. + 1.e9

Qs = np.empty_like(temperatures)
Qs2 = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    sp.set_state(1.e9, T)
    sp2.set_state(1.e9, T)
    Qs[i] = sp.property_modifier_properties[0]['Q']
    Qs2[i] = sp2.property_modifier_properties[0]['Q']

fig = plt.figure(figsize=(12, 5))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
ax[0].plot(temperatures-273.15, Qs, label='JH_2015 (HP_2012)')
ax[0].plot(temperatures-273.15, Qs2, label='HGP_2018')
#plt.plot(temperatures, sp.evaluate(['molar_heat_capacity_p'], pressures, temperatures)[0])
#plt.plot(temperatures, sp2.evaluate(['molar_heat_capacity_p'], pressures, temperatures)[0])
ax[1].plot(temperatures-273.15, (sp.evaluate(['S'], pressures, temperatures)[0]
                          - sp2.evaluate(['S'], pressures, temperatures)[0]))
for i in range(2):
    ax[i].set_xlabel('T (C)')
    ax[i].legend()
ax[0].set_ylabel('$Q$')
ax[1].set_ylabel('$\Delta S$')
plt.show()

exit()
"""

gt = transform_solution_to_new_basis(JH_2015.garnet(),
                                     np.array([[1., 0., 0., 0., 0.],
                                               [0., 0., 1., 0., 0.]]),
                                     solution_name='garnet')

# di, fs, cats, crdi, cess, jd, cen, cfm
cpx = transform_solution_to_new_basis(JH_2015.clinopyroxene(),
                                      np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                                                [0., 0., 1., 0., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 0., 0., 1., 0.]]),
                                      solution_name='cpx')

# en, fs, fm, odi, mgts, cren, mess
opx = transform_solution_to_new_basis(JH_2015.orthopyroxene(),
                                      np.array([[1., 0., 0., 0., 0., 0., 0.],
                                                [0., 0., 0., 1., 0., 0., 0.],
                                                [0., 0., 0., 0., 1., 0., 0.]]),
                                      solution_name='opx')


#print([e[0].property_modifiers for e in gt.endmembers])
print([e[0].property_modifiers for e in cpx.endmembers])
#print([e[0].property_modifiers for e in opx.endmembers])

assemblage1 = burnman.Composite([fo, sp, cpx, opx],
                               [0.30, 0.10, 0.30, 0.30])


opx.set_composition([0.74, 0.20, 0.06])
cpx.set_composition([0.33, 0.46, 0.21])
gt.set_composition([0.87, 0.13])
opx.guess = np.array([0.74, 0.20, 0.06])
cpx.guess = np.array([0.33, 0.46, 0.21])
gt.guess = np.array([0.87, 0.13])

assemblage1.set_state(2.4e9, 1200)

equality_constraints = [('T', 1200.),
                        ('P', 2.e9)]

sols, prm = burnman.equilibrate(assemblage1.formula, assemblage1,
                                equality_constraints,
                                initial_state_from_assemblage=True,
                                initial_composition_from_assemblage=True,
                                store_iterates=False,
                                store_assemblage=True)


assemblage = burnman.Composite([fo, sp, gt, cpx, opx],
                               [0.40, 0.05, 0.05, 0.1, 0.4])


temperatures = np.linspace(1000.0, 1385, 15)


equality_constraints = [('T', temperatures),
                        ('phase_proportion', (sp, np.array([0.0])))]

assemblage.set_state(2.0e9, 1400)

sols, prm = burnman.equilibrate(assemblage.formula, assemblage,
                                equality_constraints,
                                initial_state_from_assemblage=True,
                                initial_composition_from_assemblage=True,
                                store_iterates=False,
                                store_assemblage=True)


for s in sols:
    print(s.assemblage)
pressures = np.array([sol.assemblage.pressure for sol in sols])
Al_opx = np.array([sol.assemblage.phases[-1].formula['Al'] for sol in sols])

fig = plt.figure()
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]


PT_img = mpimg.imread('../../figures/Klemme_ONeill_2000_CMAS_sp_gt_PT.png')
AlT_img = mpimg.imread('../../figures/Klemme_ONeill_2000_CMAS_sp_gt_AlT.png')
ax[0].imshow(PT_img, extent=[700., 1600., 0.5, 3.5], aspect='auto') # T(C)
ax[1].imshow(AlT_img, extent=[0.5, 0.9, -1.7, -0.8], aspect='auto') # 1e3/T (K)

ax[0].plot(temperatures-273.15, pressures/1.e9)
ax[0].set_xlabel('Temperatures (C)')
ax[0].set_ylabel('Pressure (GPa)')
ax[1].plot(1000./temperatures, np.log(Al_opx))
ax[1].set_xlabel('Temperatures (C)')
ax[1].set_ylabel('ln Al opx (pfu)')
plt.show()
