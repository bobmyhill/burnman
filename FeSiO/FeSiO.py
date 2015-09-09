# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman

# In this model, the mixing properties are based on intermediate compounds
# There are:
# Two FeO0.5 intermediates (to provide an asymmetric mixing model)
# One Fe0.5Si0.5 intermediate (to fit the melting curve of Lord et al., 2010)
# One Fe0.5Si0.5O0.5 intermediate 
class metallic_Fe_Si_O_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='metallic liquid Fe-Si-O solution'
        self.type='full_subregular'
        self.endmembers = [[Fe_liq, '[Fe]'], [Si_liq, '[Si]'], [FeO_liq, 'Fe[O]']]
        self.intermediates = [[[FeSi_liq, FeSi_liq],
                               [FeFeO_liq_0,FeFeO_liq_1]],
                              [[FeSiO_liq,FeSiO_liq]]]

        burnman.SolidSolution.__init__(self, molar_fractions)
