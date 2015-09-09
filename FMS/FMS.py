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

class MgO_SiO2_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Subregular MgO-SiO2 liquid'
        self.type='subregular'

        self.endmembers = [[DKS_2013_liquids.MgO_liquid(), '[Fe]O']
                           [DKS_2013_liquids.MgO_liquid(), '[Mg]O'], 
                           [DKS_2013_liquids.SiO2_liquid(), '[Si]O2']]

        self.enthalpy_interaction = [[[0., 0.]]]
        self.volume_interaction   = [[[0., 0.]]]
        self.entropy_interaction  = [[[0., 0.]]]
                        
        burnman.SolidSolution.__init__(self, molar_fractions)

SiO2_liq.set_state(1.e9, 2000.)
print SiO2_liq.gibbs

# In this model, we assume that MgO and FeO behave in a similar way; i.e.:
# W_FeO_SiO2 = W_MgO_SiO2,
# W_FeO_MgO = 0

# Under these assumptions, we can model mu_MgO, mu_FeO and mu_SiO2 as a function of 
# pressure, temperature and melt composition. 
