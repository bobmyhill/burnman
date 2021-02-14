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

# Solid endmembers
fo = SLB_2011.forsterite()
wad = SLB_2011.mg_wadsleyite()
ring = SLB_2011.mg_ringwoodite()
lm = burnman.Composite([SLB_2011.mg_perovskite(), SLB_2011.periclase()],
                       [0.5, 0.5], name='bdg+per')
hen = SLB_2011.hp_clinoenstatite()

# phA = (MgO)7(SiO2)2(H2O)3
E_phA_HSC_to_SUP = S_el_MgO*7. + S_el_SiO2*2. + S_el_H2O*3.

H2MgSiO4fo = burnman.CombinedMineral([HP_2011_ds62.phA(),
                                      SLB_2011.mg_wadsleyite(),
                                      SLB_2011.hp_clinoenstatite()],
                                      [1./3., -5./3., 1.], [E_phA_HSC_to_SUP/3. - 1100.e3, 0., 0.])

# Liquid endmembers
Mg2SiO4L = DKS_2013_liquids.Mg2SiO4_liquid()

dS = 7.5 # de Koker uses 0 (obvs) to fit fo melting point at 1 bar. Something between 0 and 15 is ok.
dV = 0 # must be > -2.e-7, because otherwise melt becomes denser than fo at the fo-wad-melt invariant
Mg2SiO4L.property_modifiers = [['linear', {'delta_E': 0,
                                           'delta_S': dS, 'delta_V': dV}]]

fo.set_state(16.7e9, 2315+273.15) # Presnall and Walter
Mg2SiO4L.set_state(16.7e9, 2315+273.15)

Mg2SiO4L.property_modifiers = [['linear', {'delta_E': fo.gibbs - Mg2SiO4L.gibbs,
                                           'delta_S': dS, 'delta_V': dV}]]
burnman.Mineral.__init__(Mg2SiO4L)

H2OL = H2O_Pitzer_Sterner()