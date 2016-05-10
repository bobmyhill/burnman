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

        
        H0 = -90385.61948467 # 20
        H1 = -195253.7654867 # 20
        V0=-3.66714911e-06
        V1=-2.46583700e-07
        S0=82.
        S1=4.
        V0=-1.8e-06
        V1=-1.8e-06
        #S0=48.
        #S1=48.
        Pref=14.e9
        Tref=2185+273.15 + 20.
        self.energy_interaction = [[[H0-Pref*V0 + Tref*S0, H1-Pref*V1 + Tref*S1]]]
        self.volume_interaction   = [[[V0, V1]]]
        self.entropy_interaction  = [[[S0, S1]]]               
        burnman.SolidSolution.__init__(self, molar_fractions)
        
        '''
        # Direct from DKS2013
        H0 = 2.77424715e+04
        H1 = -2.04118121e+05 
        S0 = 8.13454888e+01 
        S1 = 4.09162664e+00
        V0 = -3.62424386e-06 
        V1 = -2.56157514e-07
        Pref=0.
        Tref=0.
        self.enthalpy_interaction = [[[H0-Pref*V0 + Tref*S0, H1-Pref*V1 + Tref*S1]]]
        self.volume_interaction   = [[[V0, V1]]]
        self.entropy_interaction  = [[[S0, S1]]]
        burnman.SolidSolution.__init__(self, molar_fractions)
        '''
