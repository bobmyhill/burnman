from __future__ import absolute_import
from __future__ import print_function

import platform
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from scipy.optimize import nnls
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))



import burnman
from burnman.minerals import JH_2015, HGP_2018_ds633
from burnman.solutionbases import transform_solution_to_new_basis
from burnman.composition import Composition

"""
cats = JH_2015.cats()
temperatures = np.linspace(300., 2500., 101)
Qs = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    cats.set_state(2.e9, T)
    Qs[i] = cats.property_modifier_properties[0]['Q']
plt.plot(temperatures, Qs)
plt.show()
exit()
"""

KO_SM6 = Composition({'MgO': 30.7,
                      'CaO': 6.9,
                      'SiO2': 39.7,
                      'Al2O3': 22.7}, unit_type='weight')

KO_SM25 = Composition({'MgO': 36.4,
                       'CaO': 5.5,
                       'SiO2': 36.5,
                       'Al2O3': 21.4}, unit_type='weight')

KO_SM28 = Composition({'MgO': 39.6,
                       'CaO': 6.2,
                       'SiO2': 38.1,
                       'Al2O3': 16.1}, unit_type='weight')

KO_SM6.renormalize('atomic', 'O', 5.19)

#print(KO_SM6.atomic_composition)

fo = JH_2015.fo()
sp = HGP_2018_ds633.sp()
herc = JH_2015.herc()
cats = JH_2015.cats()
di = JH_2015.di()

print('current model tc347')
for m, G in [(fo, -2469.37), (sp, -2594.32),
             (herc, -2308.32), (cats, -3734.72), (di, -3629.82)]:
    m.set_state(2.e9, 1673.)
    print('{0}: {1:.3f}'.format(m.name, m.gibbs/1000. - G))


gt = transform_solution_to_new_basis(JH_2015.garnet(),
                                     np.array([[1., 0., 0., 0., 0.],
                                               [0., 0., 1., 0., 0.]]),
                                     solution_name='garnet')

# di, fs, cats, crdi, cess, jd, cen, cfm
cpx = transform_solution_to_new_basis(JH_2015.clinopyroxene(),
                                      np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 0., 0., 1., 0.],
                                                [0., 0., 1., 0., 0., 0., 0., 0.]]),
                                      solution_name='cpx')

# en, fs, fm, odi, mgts, cren, mess
opx = transform_solution_to_new_basis(JH_2015.orthopyroxene(),
                                      np.array([[1., 0., 0., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 1., 0., 0.],
                                                [0., 0., 0., 1., 0., 0., 0.]]),
                                      solution_name='opx')


#print([e[0].property_modifiers for e in gt.endmembers])
#print([e[0].property_modifiers for e in cpx.endmembers])
#print([e[0].property_modifiers for e in opx.endmembers])

assemblage2 = burnman.Composite([fo, gt, cpx, opx],
                                [0.115, 0.290, 0.284, 0.311],
                                name='C191-S14')


opx.set_composition([0.77714171, 0.13795236, 0.08490593])
cpx.set_composition([0.47651093, 0.38200906, 0.14148001])
gt.set_composition([0.87010073, 0.12989927])
assemblage2.set_state(2.7e9, 1673.15)

elements = ['Ca', 'Mg', 'Al', 'Si', 'O']
C = []
for ph in assemblage2.phases:
    C.append([0, 0, 0, 0, 0])
    for i, e in enumerate(elements):
        try:
            C[-1][i] = ph.formula[e]
        except KeyError:
            pass

b = [0, 0, 0, 0, 0]
for i, e in enumerate(elements):
    try:
        b[i] = KO_SM6.atomic_composition[e]
    except KeyError:
        pass
C = np.array(C)
b = np.array(b)


sol = nnls(C.T, b)
print('{0} phase proportions: {1}, res: {2:.2f}'.format(assemblage2.name,
                                                        sol[0], sol[1]))


assemblage2 = burnman.Composite([fo, gt, cpx, opx],
                                [0.115, 0.290, 0.284, 0.311],
                                name='C191-S29')
opx.set_composition([0.78014434, 0.15395243, 0.06590323])
cpx.set_composition([0.66256277, 0.18513338, 0.15230386])
gt.set_composition([0.86362201, 0.13637799])
assemblage2.set_state(2.0e9, 1473.15)

elements = ['Ca', 'Mg', 'Al', 'Si', 'O']
C = []
for ph in assemblage2.phases:
    C.append([0, 0, 0, 0, 0])
    for i, e in enumerate(elements):
        try:
            C[-1][i] = ph.formula[e]
        except KeyError:
            pass

b = [0, 0, 0, 0, 0]
for i, e in enumerate(elements):
    try:
        b[i] = KO_SM6.atomic_composition[e]
    except KeyError:
        pass
C = np.array(C)
b = np.array(b)


sol = nnls(C.T, b)
print('{0} phase proportions: {1}, res: {2:.2f}'.format(assemblage2.name,
                                                        sol[0], sol[1]))

assemblage1 = burnman.Composite([fo, sp, cpx, opx],
                                [0.115, 0.290, 0.284, 0.311],
                                name='C163-S10')
cpx.set_composition([0.51974, 0.29790, 0.18236])
opx.set_composition([0.73089, 0.19488, 0.07423])
assemblage1.set_state(2.e9, 1673.15)


elements = ['Ca', 'Mg', 'Al', 'Si', 'O']
C = []
for ph in assemblage1.phases:
    C.append([0, 0, 0, 0, 0])
    for i, e in enumerate(elements):
        try:
            C[-1][i] = ph.formula[e]
        except KeyError:
            pass

b = [0, 0, 0, 0, 0]
for i, e in enumerate(elements):
    try:
        b[i] = KO_SM6.atomic_composition[e]
    except KeyError:
        pass
C = np.array(C)
b = np.array(b)

sol = nnls(C.T, b)
print('{0} phase proportions: {1}, res: {2:.2f}'.format(assemblage1.name,
                                                        sol[0], sol[1]))

assemblage1.set_fractions(sol[0])

print('cpx', cpx.gibbs, cpx.S, cpx.molar_heat_capacity_p)
print('opx', opx.gibbs, opx.S, opx.molar_heat_capacity_p)
print('fo', fo.gibbs, fo.S, fo.molar_heat_capacity_p)
print('sp', sp.gibbs, sp.S, sp.molar_heat_capacity_p)


# PERPLEX output
#                    N(g)          G(J)     S(J/K)     V(J/bar)      Cp(J/K)       Alpha(1/K)  Beta(1/bar)    Cp/Cv    Density(kg/m3)
# Cpx(JH)           212.14       -3404863   537.11       6.6232       265.64      0.37687E-04  0.10276E-05   1.0612       3202.9
# Opx(JH)           202.25       -3330435   528.42       6.3595       269.97      0.35783E-04  0.11386E-05   1.0464       3180.3
# fo                140.69       -2321943   367.14       4.5224       188.35      0.41965E-04  0.89296E-06   1.0860       3111.0
# sp                142.26       -2444487   #365.40#       4.0802       190.38      0.29140E-04  0.57720E-06   1.0557       3486.7
# System            100.00       -1649464   257.54       3.0802       130.84      0.35592E-04  0.95890E-06   1.0549       3246.5



opx.guess = np.array(opx.molar_fractions)
cpx.guess = np.array(cpx.molar_fractions)
gt.guess = np.array(gt.molar_fractions)


assemblage = burnman.Composite([fo, sp, gt, cpx, opx],
                               [0.40, 0.05, 0.05, 0.1, 0.4])


temperatures = np.linspace(1000.0, 1800., 25)


equality_constraints = [('T', temperatures),
                        ('phase_proportion', (sp, np.array([0.0])))]

assemblage.set_state(2.0e9, 1400)

sols, prm = burnman.equilibrate(KO_SM6.atomic_composition,
                                assemblage,
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
