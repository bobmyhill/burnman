import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import DKS_2013_liquids, DKS_2013_solids
from burnman import constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


per = DKS_2013_solids.periclase()
pv = DKS_2013_solids.perovskite()
stv = DKS_2013_solids.stishovite()

SiO2_liq=DKS_2013_liquids.SiO2_liquid()
MgO_liq=DKS_2013_liquids.MgO_liquid()


# In this model, we assume that MgO and FeO 
# behave in a similar way; i.e.:
# W_FeO_SiO2 = W_MgO_SiO2,
# W_FeO_MgO = 0

# Under these assumptions, we can model 
# mu_MgO, mu_FeO and mu_SiO2 as a function of 
# pressure, temperature and melt composition. 

class FeO_MgO_SiO2_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Subregular MgO-SiO2 liquid'
        self.type='subregular'

        self.endmembers = [[DKS_2013_liquids.MgO_liquid(), '[Fe]O'],
                           [DKS_2013_liquids.MgO_liquid(), '[Mg]O'], 
                           [DKS_2013_liquids.SiO2_liquid(), '[Si]O2']]

        self.enthalpy_interaction = [[[0., 0.], 
                                      [-108600., -182300.]],
                                     [[-108600., -182300.]]]
        self.entropy_interaction   = [[[0., 0.], 
                                       [61.2, 15.5]],
                                      [[61.2, 15.5]]]
        self.volume_interaction  = [[[0., 0.], 
                                     [4.32e-7, 1.35e-7]],
                                    [[4.32e-7, 1.35e-7]]]

        burnman.SolidSolution.__init__(self, molar_fractions)



liq = FeO_MgO_SiO2_liquid()

c = [0.2, 0.4, 0.4]
P = 25.e9
T = 3000.

liq.set_composition(c)
liq.set_state(P, T)

print liq.excess_partial_gibbs
