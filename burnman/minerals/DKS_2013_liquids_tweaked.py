# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

"""
DKS_2013
Liquids from de Koker and Stixrude (2013) FPMD simulations
"""

from burnman.mineral import Mineral
from burnman.solidsolution import SolidSolution
from burnman.solutionmodel import *

# Vector parsing for DKS liquid equation of state
def vector_to_array(a, Of, Otheta):
    array=np.empty([Of+1, Otheta+1])
    for i in range(Of+1):
        for j in range(Otheta+1):
            n=int((i+j)*((i+j)+1.)/2. + j)
            array[i][j]=a[n]
    return array

class SiO2_liquid(Mineral):
    def __init__(self):
        self.params = {
            'name': 'SiO2_liquid',
            'formula': {'Mg': 0 , 'Si': 1.0 , 'O': 2.0 },
            'equation_of_state': 'dks_l',
            'V_0': 2.78e-05 ,
            'T_0': 3000.0 ,
            'F_0': 1618718.19063, #-2360007.614 ,
            'S_0': 0., #-0.1380253514 ,
            'O_theta': 2 ,
            'O_f': 5 ,
            'm': 0.91 ,
            'a': [-1945.93156, -226.6835978, 455.0286309, 2015.65287, -200.585046, -216.6028187, 48369.72992, 441.5340414, 73.07765325, 0.0, -651587.652, 20701.69954, 892.12209, 0.0, 0.0, 4100181.286, -128258.7237, -1228.478753, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
            'zeta_0': 0.0004266056389 ,
            'xi': 0.8639433047 ,
            'Tel_0': 5651.204964 ,
            'eta': -0.2783503528 ,
            'el_V_0': 1e-06
            }
        self.params['a'] = vector_to_array(self.params['a'], self.params['O_f'], self.params['O_theta'])*1e3 # [J/mol]
        Mineral.__init__(self)


class MgO_liquid(Mineral):
    def __init__(self):
        self.params = {
            'name': 'MgO_liquid',
            'formula': {'Mg': 1.0 , 'Si': 0 , 'O': 1.0 },
            'equation_of_state': 'dks_l',
            'V_0': 1.646e-05 ,
            'T_0': 3000.0 ,
            'F_0': 689722.614099, #-1089585.069 ,
            'S_0': 0., #-0.05477244661 ,
            'O_theta': 2 ,
            'O_f': 3 ,
            'm': 0.63 ,
            'a': [-925.2677296, -155.3240992, 260.8211743, 5323.167667, 466.3722398, -88.30035696, 10473.87879, 1997.967054, 50.72520834, 0.0, 0.0, -9914.621337, 71.89989255, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
            'zeta_0': 0.002194565772 ,
            'xi': 0.411459446 ,
            'Tel_0': 1620.106387 ,
            'eta': -0.986457555 ,
            'el_V_0': 1.620953559e-05
            }
        self.params['a'] = vector_to_array(self.params['a'], self.params['O_f'], self.params['O_theta'])*1e3 # [J/mol]
        Mineral.__init__(self)


