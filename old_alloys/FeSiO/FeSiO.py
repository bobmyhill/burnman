# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.chemicalpotentials import *

# First, let's import the metallic melt model
from Fe_Si_O_liquid_models import *
# In this model, the mixing properties are based on intermediate compounds
# There are:
# Two FeO0.5 intermediates (to provide an asymmetric mixing model)
# One Fe0.5Si0.5 intermediate (to fit the melting curve of Lord et al., 2010)
# One Fe0.5Si0.5O0.5 intermediate 


# Now there are two melts which we want to equilibrate with each other. 
# The first is an MgO-FeO-SiO2 melt, which gives us three chemical potentials:
# mu_MgO, mu_FeO, mu_SiO2
# The second is the metallic melt, which also gives us three chemical potentials
# mu_Fe, mu_FeO, mu_Si
# To find the equilibrium between these two phases, we need to equate two chemical potentials:
# mu_FeO = mu_FeO
# mu_SiO2 = mu_Si + 2*(mu_FeO - mu_Fe)


#silicate_melt = FeO_MgO_SiO2_liquid.FeO_MgO_SiO2_liquid()
metallic_melt = metallic_Fe_Si_O_liquid()

P = 25.e9
T = 3000.

#FeO_MgO_SiO2 wt percents
oxides = ['FeO', 'MgO', 'SiO2']
molar_masses = np.array([71.844, 40.3044, 60.08])
wt_composition_silicate_melt = np.array([4.4, 40.0, 53.8])

molar_fractions = (wt_composition_silicate_melt/molar_masses) \
    / np.sum(wt_composition_silicate_melt/molar_masses)
print oxides
print molar_fractions

exit()


#component_formulae=['FeO', 'SiO2', 'O2']
#component_formulae_dict=[dictionarize_formula(f) for f in component_formulae]
#chem_potentials=chemical_potentials(FMQ, component_formulae_dict)
