from __future__ import absolute_import
from __future__ import print_function
import os.path
import sys
sys.path.insert(1, os.path.abspath('../..'))

import burnman
from burnman.minerals import SLB_2011, HP_2011_ds62

"""
molar weight (g), elemental entropy (R)
Na2O     61.9790    205.1750
MgO      40.3040    135.2550
Al2O3   101.9610    364.4250
SiO2     60.0840    223.9600
K2O      94.1960    231.9350
CaO      56.0770    144.2050
TiO2     79.8660    235.8700
MnO      70.9370    134.7950
FeO      71.8440    129.8550
NiO      74.6930    132.3750
ZrO2    123.2200    244.3300
Cl2      70.9060    223.0800
O2       31.9990    205.1500
H2O      18.0150    233.2550
CO2      44.0100    210.8900
CuO      79.5450    135.7250
Cr2O3   151.9900    358.8110
"""

G_per_Barin = -609268.

per_SLB = SLB_2011.periclase()
per_HP = HP_2011_ds62.per()

per_SLB.set_state(1.e5, 298.15)
per_HP.set_state(1.e5, 298.15)


print(G_per_Barin + 135.2550*298.15, per_SLB.gibbs, per_HP.gibbs + 135.2550*298.15)
print(G_per_Barin, per_SLB.S, per_HP.S)
