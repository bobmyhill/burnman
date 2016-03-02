import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    Komabayashi_2014,\
    Myhill_calibration_iron,\
    Fe_Si_O
from Fe_Si_O_liquid_models import *
from fitting_functions import *
from burnman.solidsolution import SolidSolution
import numpy as np
from scipy.optimize import fsolve

FeSiO_melt = metallic_Fe_Si_O_liquid()
silicate_melt = MgO_FeO_SiO2_liquid()

# Equilibrium between metallic and silicate melt governed by:
# mu_FeO (silicate) = mu_FeO (metallic)
# mu_SiO2 (silicate) = mu_SiO2 = 2mu_FeO + mu_Si - 2mu_Fe (metallic)


# Input: silicate melt composition
# Output: metallic melt composition

def metallic_melt_composition(args, P, T, X_FeO_sil, X_SiO2_sil):
    X_FeO_metal, X_Si_metal = args
    X_Fe_metal = 1. - X_FeO_metal - X_Si_metal

    X_MgO_sil = 1.-X_FeO_silicate-X_SiO2_silicate
    silicate_melt.set_composition([X_MgO_sil, X_FeO_sil, X_SiO2_sil])
    metallic_melt.set_composition([X_Fe_metal, X_Si_metal, X_FeO_metal])

    silicate_melt.set_state(P, T)
    metallic_melt.set_state(P, T)

    mu_FeO_sil = silicate_melt.partial_gibbs[1] 
    mu_FeO_metal = metallic_melt.partial_gibbs[2]

    mu_SiO2_sil = silicate_melt.partial_gibbs[2] 
    mu_SiO2_metal = 2.*metallic_melt.partial_gibbs[2] \
        - 2.*metallic_melt.partial_gibbs[0] \
        + metallic_melt.partial_gibbs[1]

    return [mu_FeO_sil - mu_FeO_metal,
            mu_SiO2_sil - mu_SiO2_metal]

