import burnman
from burnman.minerals import DKS_2013_liquids_tweaked

class MgO_SiO2_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Subregular MgO-SiO2 liquid'
        self.type='subregular'

        self.endmembers = [[DKS_2013_liquids_tweaked.MgO_liquid(), '[Mg]O'],
                           [DKS_2013_liquids_tweaked.SiO2_liquid(), '[Si]O2']]

        # self.enthalpy_interaction = [[[-92201 + 14.e9*3.2e-6, -198032. + 14.e9*+3.2e-6]]] # 14 GPa, no T tweak
        # self.enthalpy_interaction = [[[-91297 + 14.e9*3.2e-6, -196645. + 14.e9*+3.2e-6]]] # 14 GPa, 10 K tweak
        self.enthalpy_interaction = [[[8740. + 14.e9*2.0e-6, -96127. + 14.e9*+2.0e-6]]] # 14 GPa, 20 K tweak

        self.volume_interaction   = [[[-2.0e-6, -2.0e-6]]]
        self.entropy_interaction  = [[[40., 40.]]]
                        
        burnman.SolidSolution.__init__(self, molar_fractions)


