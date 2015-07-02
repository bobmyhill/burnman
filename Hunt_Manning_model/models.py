import burnman
from burnman.minerals import DKS_2013_liquids_tweaked

class MgO_SiO2_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Subregular MgO-SiO2 liquid'
        self.type='subregular'

        self.endmembers = [[DKS_2013_liquids_tweaked.MgO_liquid(), '[Mg]O'],
                           [DKS_2013_liquids_tweaked.SiO2_liquid(), '[Si]O2']]

        self.enthalpy_interaction = [[[-87653.77 + 14.e9*3.2e-6, -191052. + 14.e9*+3.2e-6]]] # 14 GPa
        self.volume_interaction   = [[[-3.2e-6, -3.2e-6]]]
        self.entropy_interaction  = [[[0., 0.]]]
                        
        burnman.SolidSolution.__init__(self, molar_fractions)


