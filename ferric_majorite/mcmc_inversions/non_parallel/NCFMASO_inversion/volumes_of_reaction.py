from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from scipy.optimize import minimize, fsolve
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../burnman'):
    sys.path.insert(1, os.path.abspath('../..'))
import burnman
from burnman.solidsolution import SolidSolution as Solution
from burnman.processanalyses import compute_and_set_phase_compositions, assemblage_affinity_misfit
from burnman.equilibrate import equilibrate
from fitting_functions import equilibrium_order

from input_dataset import *
from preload_params import *
    
print('Olivine-polymorph-skiagite (Fe buffered)')
print('3Fe2SiO4 <-> Fe3Fe2Si3O12 + Fe')
for (P, T, fe_ol_polymorph) in [(12.e9, 1673.15, fa),
                                (16.e9, 1673.15, fwd),
                                (20.e9, 1673.15, frw)]:
                             
    minerals = [fe_ol_polymorph, fcc_iron, child_solutions['sk_gt']]
    for m in minerals:
        m.set_state(P, 1673.15)

    delta_V = minerals[2].V + minerals[1].V - 3.*minerals[0].V
    print('{0:.1f} GPa, {1:.1f} K: {2:.1f} cm^3/mol-skiagite ({3})'.format(P/1.e9, T, delta_V*1.e6, fe_ol_polymorph.name))
     
print('')


fmaj = transform_solution_to_new_basis(gt, np.array([[-1., 1., 0., 0., 1., 0.]]),
                                       solution_name='Fe-majorite (MgSi on oct)')
print('Clinopyroxene -> majorite')
print('3Fe2Si2O6 + Mg2Si2O6 -> 2Fe3(MgSi)Si3O12')
minerals = [cfs, cen, fmaj]
P = 14.e9
for m in minerals:
    m.set_state(P, 1673.15)

delta_V = 2.*minerals[2].V - 3.*minerals[0].V - minerals[1].V
print('{0:.1f} GPa, {1:.1f} K: {2:.1f} cm^3/2 moles-fmaj'.format(P/1.e9, T, delta_V*1.e6))

print('')

# MORB reaction (Fe with excess silica, majorite stable - otherwise need cpx for other Fe,Si bearing phase)
print('Clinoferrosilite-skiagite (stv + Fe buffered)')
print('3Fe2Si2O6 -> Fe3Fe2Si3O12 + 3 SiO2 + Fe')
print(child_solutions['sk_gt'].formula)
minerals = [cfs, child_solutions['sk_gt'], stv, fcc_iron]
P = 14.e9
for m in minerals:
    m.set_state(P, 1673.15)

delta_V = minerals[1].V + 3.*minerals[2].V + minerals[3].V - 3.*minerals[0].V
print('{0:.1f} GPa, {1:.1f} K: {2:.1f} cm^3/mol-skiagite'.format(P/1.e9, T, delta_V*1.e6))

print('')

print('Ferric-in-majorite (stv + Fe buffered)')
print('2Fe3(MgSi)Si3O12 <-> Fe3Fe2Si3O12 + 3 SiO2 + Fe + 1/2 Mg3(MgSi)Si3O12')
minerals = [fmaj, child_solutions['sk_gt'], stv, fcc_iron, dmaj]
P = 14.e9
for m in minerals:
    m.set_state(P, 1673.15)

delta_V = minerals[1].V + 3.*minerals[2].V + minerals[3].V + 0.5*minerals[4].V - 2.*minerals[0].V
print('{0:.1f} GPa, {1:.1f} K: {2:.1f} cm^3/mol-skiagite'.format(P/1.e9, T, delta_V*1.e6))




# There are similar reactions for O2 rather than Fe:

# Pyrolite reaction (O2, majorite stable)
# Fe3(MgSi)Si3O12 + 6Fe2SiO4 + 3/2 O2 <-> 3Fe3Fe2Si3O12 + 1/4 Mg3(MgSi)Si3O12

# MORB reaction (O2 with excess silica, majorite stable)
# 5Fe3(MgSi)Si3O12 + 3/2 O2 <-> 3Fe3Fe2Si3O12 + 6SiO2 + 5/4 Mg3(MgSi)Si3O12
