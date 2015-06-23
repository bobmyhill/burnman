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

def adjust_vector_a(Fxs0, Sxs0, Pxs0, params):
    n = 2.
    m = params['m']
    T_0 = params['T_0']
    V_0 = params['V_0']
    a00 = Fxs0

    a10 = 3.*V_0*Pxs0
    a20 = -(1.*n + 3.)*a10
    a30 = -(2.*n + 3.)*a20
    a40 = -(3.*n + 3.)*a30

    a01 = (1.-0.*m) / m * (-T_0*Sxs0)
    a02 = (1.-1.*m) / m * a01
    a03 = (1.-2.*m) / m * a02
    a04 = (1.-3.*m) / m * a03 # sign error in thesis?

    params['a'][0] += a00

    params['a'][1] += a10
    params['a'][2] += a01

    params['a'][3] += a20
    params['a'][5] += a02

    params['a'][6] += a30
    params['a'][9] += a03

    params['a'][10] += a40
    params['a'][14] += a04

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
            'V_0': 2.78e-05 , # was 2.78e-05 
            'T_0': 3000.0 ,
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
        Fxs0= 3.6 + 1617.47564545 # kJ/mol, -5 for FPMD stv
        Sxs0=20.e-3 # kJ/mol, 20e-3 for FPMD stv
        Pxs0=00000.
        adjust_vector_a(Fxs0, Sxs0, Pxs0, self.params)
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
        Fxs0= 689.722614099 # kJ/mol
        Sxs0= 0.e-3 # kJ/mol
        Pxs0=00000.
        adjust_vector_a(Fxs0, Sxs0, Pxs0, self.params)
        self.params['a'] = vector_to_array(self.params['a'], self.params['O_f'], self.params['O_theta'])*1e3 # [J/mol]
        Mineral.__init__(self)


