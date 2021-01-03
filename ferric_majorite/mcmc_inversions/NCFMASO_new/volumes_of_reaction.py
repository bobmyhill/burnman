from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))

from create_dataset import create_dataset
from burnman.solutionbases import transform_solution_to_new_basis

dataset, storage, labels = create_dataset(import_assemblages=False)
endmembers = dataset['endmembers']
solutions = dataset['solutions']


print('Olivine-polymorph-skiagite (Fe buffered)')
print('3Fe2SiO4 <-> Fe3Fe2Si3O12 + Fe')

# py alm gr andr dmaj nagt
sk = transform_solution_to_new_basis(solutions['gt'],
                                     np.array([[0., 1., -1., 1., 0., 0.]]),
                                     solution_name='sk')

for (P, T, fe_ol_polymorph) in [(12.e9, 1673.15, endmembers['fa']),
                                (16.e9, 1673.15, endmembers['fwd']),
                                (20.e9, 1673.15, endmembers['frw'])]:

    minerals = [fe_ol_polymorph,  endmembers['fcc_iron'], sk]
    for m in minerals:
        m.set_state(P, 1673.15)

    delta_V = minerals[2].V + minerals[1].V - 3.*minerals[0].V
    print('{0:.1f} GPa, {1:.1f} K: {2:.1f} cm^3/mol-skiagite ({3})'.format(P/1.e9, T, delta_V*1.e6, fe_ol_polymorph.name))

print('')


fmaj = transform_solution_to_new_basis(solutions['gt'],
                                       np.array([[-1., 1., 0., 0., 1., 0.]]),
                                       solution_name='Fe-majorite (MgSi on oct)')
print('Clinopyroxene -> majorite')
print('3Fe2Si2O6 + Mg2Si2O6 -> 2Fe3(MgSi)Si3O12')
minerals = [endmembers['cfs'], endmembers['cen'], fmaj]
P = 14.e9
for m in minerals:
    m.set_state(P, 1673.15)

delta_V = 2.*minerals[2].V - 3.*minerals[0].V - minerals[1].V
print('{0:.1f} GPa, {1:.1f} K: {2:.1f} cm^3/2 moles-fmaj'.format(P/1.e9, T,
                                                                 delta_V*1.e6))

print('')

# MORB reaction (Fe with excess silica, majorite stable
# - otherwise need cpx for other Fe,Si bearing phase)
print('Clinoferrosilite-skiagite (stv + Fe buffered)')
print('3Fe2Si2O6 -> Fe3Fe2Si3O12 + 3 SiO2 + Fe')
print(sk.formula)
minerals = [endmembers['cfs'], sk, endmembers['stv'], endmembers['fcc_iron']]
P = 14.e9
for m in minerals:
    m.set_state(P, 1673.15)

delta_V = minerals[1].V + 3.*minerals[2].V + minerals[3].V - 3.*minerals[0].V
print('{0:.1f} GPa, {1:.1f} K: {2:.1f} cm^3/mol-skiagite'.format(P/1.e9, T,
                                                                 delta_V*1.e6))

print('')

print('Ferric-in-majorite (stv + Fe buffered)')
print('2Fe3(MgSi)Si3O12 <-> Fe3Fe2Si3O12 + 3 SiO2 + Fe + 1/2 Mg3(MgSi)Si3O12')
minerals = [fmaj, sk, endmembers['stv'],
            endmembers['fcc_iron'], endmembers['dmaj']]
P = 14.e9
for m in minerals:
    m.set_state(P, 1673.15)

delta_V = minerals[1].V + 3.*minerals[2].V + minerals[3].V + 0.5*minerals[4].V - 2.*minerals[0].V
print('{0:.1f} GPa, {1:.1f} K: {2:.1f} cm^3/mol-skiagite'.format(P/1.e9, T,
                                                                 delta_V*1.e6))

# There are similar reactions for O2 rather than Fe:

# Pyrolite reaction (O2, majorite stable)
# Fe3(MgSi)Si3O12 + 6Fe2SiO4 + 3/2 O2 <-> 3Fe3Fe2Si3O12 + 1/4 Mg3(MgSi)Si3O12

# MORB reaction (O2 with excess silica, majorite stable)
# 5Fe3(MgSi)Si3O12 + 3/2 O2 <-> 3Fe3Fe2Si3O12 + 6SiO2 + 5/4 Mg3(MgSi)Si3O12
