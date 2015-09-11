import os, sys
sys.path.insert(1,os.path.abspath('..'))
import burnman

# This model is based on the de Koker, Karki and Stixrude (2013)
# FPMD simulations for MgO and SiO2.

# Their results are fit reasonably well across the binary by 
# a subregular solution model. However, it should be noted that 
# the MgO side of the binary could be fit better with a more ideal
# model involving mixing between MgO and Mg2SiO4. 
# Between Mg2SiO4 and MgSiO3, a rapid decrease in MgO activity 
# might mirror a change in silicate-oxide 
# speciation in the melt.

# For now, let's stick with the simpler binary model...

# To add FeO, we assume that MgO and FeO 
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

        self.endmembers = [[burnman.minerals.DKS_2013_liquids.MgO_liquid(), 
                            '[Fe]O'],
                           [burnman.minerals.DKS_2013_liquids.MgO_liquid(), 
                            '[Mg]O'], 
                           [burnman.minerals.DKS_2013_liquids.SiO2_liquid(), 
                            '[Si]O2']]

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



