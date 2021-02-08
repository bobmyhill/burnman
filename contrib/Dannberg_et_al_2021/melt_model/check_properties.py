from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../../'))

import burnman
from burnman.processchemistry import dictionarize_formula, formula_mass


from model_parameters import *
from eos_and_averaging import thermodynamic_properties, average_solid_properties, average_melt_properties, average_composite_properties

mpv = burnman.Mineral(params=mpv_params)
fpv = burnman.Mineral(params=fpv_params)
per = burnman.Mineral(params=per_params)
wus = burnman.Mineral(params=wus_params)

mpv.params['P_0'] = 0.
fpv.params['P_0'] = 0.
per.params['P_0'] = 0.
wus.params['P_0'] = 0.


pressure = 20.e9
temperature = 4000.
mass_fraction_pv = 0.2
p_fpv = 0.22
p_wus = 0.21

pv = burnman.SolidSolution(solution_type='ideal',
                           endmembers = [[mpv, '[Mg]SiO3'],
                                         [fpv, '[Fe]SiO3']],
                           molar_fractions=[1. - p_fpv, p_fpv])

fper = burnman.SolidSolution(solution_type='ideal',
                             endmembers = [[per, '[Mg]O'],
                                           [wus, '[Fe]O']],
                             molar_fractions=[1. - p_wus, p_wus])

solid = burnman.Composite(phases=[pv, fper],
                          fractions=[mass_fraction_pv, 1. - mass_fraction_pv],
                          fraction_type='mass', name='Solid')

solid.set_state(pressure, temperature)
solid.set_averaging_scheme('Reuss')
#burnman.tools.check_eos_consistency(solid, P=80.e9, T=4000., verbose=True)


solid_pty = average_solid_properties(pressure, temperature,
                                     p_fpv, p_wus, mass_fraction_pv)

import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    print((solid.K_T-1./solid_pty['beta_T'])/solid.K_T)

print((solid.molar_mass - solid_pty['molar mass'])/solid.molar_mass)
print((solid.C_p - solid_pty['molar_C_p'])/solid.C_p)
print((solid.rho - solid_pty['rho'])/solid.rho)
print((solid.V - solid_pty['V'])/solid.V)
print((solid.alpha - solid_pty['alpha'])/solid.alpha)
