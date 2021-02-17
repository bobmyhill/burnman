from __future__ import absolute_import
from __future__ import print_function
import os.path
import sys
sys.path.insert(1, os.path.abspath('../..'))

import burnman
from burnman.minerals import SLB_2011, HP_2011_ds62, DKS_2013_liquids
from burnman.minerals.Pitzer_Sterner_1994 import H2O_Pitzer_Sterner

from burnman import constants
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import matplotlib.pyplot as plt

# Elemental entropies taken from PerpleX's hp633ver.dat
S_el_MgO = 135.2550
S_el_SiO2 = 223.9600
S_el_H2O = 233.2550


# Master endmembers
H2OL = H2O_Pitzer_Sterner()
hen = SLB_2011.hp_clinoenstatite()
wad = SLB_2011.mg_wadsleyite()

# Make other endmembers:
make_new = False
if make_new:
    fo = SLB_2011.forsterite()
    ring = SLB_2011.mg_ringwoodite()
    lm = burnman.Composite([SLB_2011.mg_perovskite(), SLB_2011.periclase()],
                           [0.5, 0.5], name='bdg+per')


    Mg2SiO4L = DKS_2013_liquids.Mg2SiO4_liquid()

    dS = 7.5 # de Koker uses 0 (obvs) to fit fo melting point at 1 bar. Something between 0 and 15 is ok.
    dV = 0 # must be > -2.e-7, because otherwise melt becomes denser than fo at the fo-wad-melt invariant
    Mg2SiO4L.property_modifiers = [['linear', {'delta_E': 0,
                                              'delta_S': dS, 'delta_V': dV}]]

    fo.set_state(16.7e9, 2315+273.15) # Presnall and Walter
    Mg2SiO4L.set_state(16.7e9, 2315+273.15)

    Mg2SiO4L.property_modifiers = [['linear', {'delta_E': fo.gibbs - Mg2SiO4L.gibbs,
                                               'delta_S': dS, 'delta_V': dV}]]

    P_ref = 16.7e9
    T_ref = 2315+273.15
    wad.set_state(P_ref, T_ref)
    for m in [fo, Mg2SiO4L]:
        m.set_state(P_ref, T_ref)
        print(f'{m.name}: {m.molar_internal_energy - wad.molar_internal_energy}, {m.S - wad.S}, {m.V - wad.V}')

    P_ref = 22.68e9
    T_ref = 2210. # near wad-ring-lm triple point
    wad.set_state(P_ref, T_ref)
    for m in [ring]:
        m.set_state(P_ref, T_ref)
        print(f'{m.name}: {m.molar_internal_energy - wad.molar_internal_energy}, {m.S - wad.S}, {m.V - wad.V}')


    for m in [lm]:
        m.set_state(P_ref, T_ref)
        print(f'{m.name}: {m.molar_internal_energy*2. - wad.molar_internal_energy}, {m.S*2. - wad.S}, {m.V*2. - wad.V}')

    exit()

fo = burnman.CombinedMineral([SLB_2011.mg_wadsleyite()], [1.], [-19410., 4.1, 1.86e-6], name='forsterite')
ring = burnman.CombinedMineral([SLB_2011.mg_wadsleyite()], [1.], [8208., -5.3, -8.72e-7], name='ring')
lm = burnman.CombinedMineral([SLB_2011.mg_wadsleyite()], [1.], [79081., -0.53, -3.54e-6], name='bdg+per')


Mg2SiO4L = burnman.CombinedMineral([SLB_2011.mg_wadsleyite()], [1.], [0., 0., 0.], name='liquid')

Mg2SiO4L.property_modifiers = [['linlog', {'delta_E': 175553., 'delta_S': 100.3, 'delta_V': -3.277e-6,
                                           'a': 2.60339693e-06, 'b': 2.64753089e-11, 'c': 1.18703511e+00}]]




V_ex = 2.8e-05

H2MgSiO4fo = burnman.CombinedMineral([H2O_Pitzer_Sterner(),
                                      SLB_2011.hp_clinoenstatite()],
                                      [1., 0.5], [-1173380. + 1273.15*208 - 13.e9*V_ex, 208, V_ex])

# No clear pressure trend in Demouchy wad data
H2MgSiO4wad = burnman.CombinedMineral([H2O_Pitzer_Sterner(),
                                      SLB_2011.hp_clinoenstatite()],
                                      [1., 0.5], [-1173380. - 28000. + 1273.15*203 - 13.e9*V_ex, 203, V_ex])

H2MgSiO4ring = burnman.CombinedMineral([H2O_Pitzer_Sterner(),
                                        SLB_2011.hp_clinoenstatite()],
                                       [1., 0.5], [-1173380. - 32000. + 1273.15*208 - 13.e9*V_ex, 208, V_ex])

"""
Mg2SiO4L = DKS_2013_liquids.Mg2SiO4_liquid()

dS = 7.5 # de Koker uses 0 (obvs) to fit fo melting point at 1 bar. Something between 0 and 15 is ok.
dV = 0 # must be > -2.e-7, because otherwise melt becomes denser than fo at the fo-wad-melt invariant
Mg2SiO4L.property_modifiers = [['linear', {'delta_E': 0,
                                          'delta_S': dS, 'delta_V': dV}]]

fo.set_state(16.7e9, 2315+273.15) # Presnall and Walter
Mg2SiO4L.set_state(16.7e9, 2315+273.15)

Mg2SiO4L.property_modifiers = [['linear', {'delta_E': fo.gibbs - Mg2SiO4L.gibbs,
                                           'delta_S': dS, 'delta_V': dV}]]
"""
