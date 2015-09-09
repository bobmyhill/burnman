import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
#from burnman.minerals import DKS_2013_liquids_tweaked
from burnman.minerals import Myhill_silicate_liquid
from burnman import constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#SiO2_liq=DKS_2013_liquids_tweaked.SiO2_liquid()
SiO2_liq=Myhill_silicate_liquid.SiO2_liquid()

pressures = np.linspace(11.e9, 15.e9, 5)
temperatures = np.linspace(1000., 3000., 5)
for P in pressures:
    for T in temperatures:
        SiO2_liq.set_state(P, T)
        print P/1.e9, T, SiO2_liq.gibbs, SiO2_liq.H, SiO2_liq.S, SiO2_liq.V

