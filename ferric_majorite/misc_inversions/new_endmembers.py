# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import Mineral, minerals
from burnman.processchemistry import dictionarize_formula, formula_mass
from burnman.equilibrate import equilibrate
import matplotlib.image as mpimg
from scipy.optimize import fsolve



# THIS FILE IS FOR WORKING ON INCORPORATING NEW ENDMEMBERS...
qtz = minerals.SLB_2011.quartz()
qtz.set_state(1.e5, 298.15)
print(qtz.gibbs, qtz.S)


iron = minerals.HP_2011_ds62.iron()
andr = minerals.HP_2011_ds62.andr()
iron.set_state(1.e5, 298.)
andr.set_state(1.e5, 298.)
print(iron.gibbs, andr.gibbs)

fcc = minerals.SE_2015.fcc_iron()
bcc = minerals.SE_2015.bcc_iron()
hcp = minerals.SE_2015.hcp_iron()
liq = minerals.SE_2015.liquid_iron()

bcc.set_state(1.e5, 298.)
print(bcc.gibbs)

O2 = minerals.HP_2011_fluids.O2()
O2.set_state(1.e5, 298.)
print(O2.gibbs)

