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
            'name': 'SiO2 liquid',
            'formula': {'Si': 1., 'O': 2.},
            'equation_of_state': 'dks_l',
            'V_0': 0.2780000000E+02*1e-6, # F_cl (1,1) # should be m^3/mol
            'T_0': 0.3000000000E+04, # F_cl (1,2) # K
            'E_0': -.2360007614E+04 * 1e3, # F_cl (1,3) # J/mol
            'S_0': -.1380253514E+00, # F_cl (1,4) # J/K/mol
            'O_theta': 2,
            'O_f':5,
            'm': 0.91, # F_cl (5)
            'a': [-.1945931560E+04, -.2266835978E+03, 0.4550286309E+03, 0.2015652870E+04, \
                        -.2005850460E+03, -.2166028187E+03, 0.4836972992E+05, 0.4415340414E+03, \
                        0.7307765325E+02, 0.0000000000E+00, -.6515876520E+06, 0.2070169954E+05, \
                        0.8921220900E+03, 0.0000000000E+00, 0.0000000000E+00, 0.4100181286E+07, \
                        -.1282587237E+06, -.1228478753E+04, 0.0000000000E+00, 0.0000000000E+00, \
                        0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, \
                        0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, \
                        0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, \
                        0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00],
            'zeta_0':0.4266056389E-03, # [J K^-2] F_el (1)
            'xi':0.8639433047E+00, # [], F_el (2)
            'Tel_0':0.5651204964E+04, # [K], F_el (3)
            'eta':-.2783503528E+00, # []
            'el_V_0':0.1000000000E+01 * 1e-6 # [m^3/mol]
            } # F_el (4)
        self.params['a'] = vector_to_array(self.params['a'], self.params['O_f'], self.params['O_theta'])*1e3 # [J/mol]
        Mineral.__init__(self)

class MgO_liquid(Mineral):
    def __init__(self):
        self.params={
            'name': 'MgO liquid',
            'formula': {'Mg': 1., 'O': 1.},
            'equation_of_state': 'dks_l',
            'V_0': 0.1646000000E+02*1e-6, # F_cl (1,1) # should be m^3/mol
            'T_0': 0.3000000000E+04, # F_cl (1,2) # K
            'E_0': -.1089585069E+04 * 1e3, # F_cl (1,3) # J/mol
            'S_0': -.5477244661E-01, # F_cl (1,4) # J/K/mol
            'O_theta': 2, 
            'O_f':3, 
            'm': 0.63, # F_cl (5)
            'a': [-.9252677296E+03, -.1553240992E+03, 0.2608211743E+03, 0.5323167667E+04, \
                        0.4663722398E+03, -.8830035696E+02, 0.1047387879E+05, 0.1997967054E+04, \
                        0.5072520834E+02, 0.0000000000E+00, 0.0000000000E+00, -.9914621337E+04, \
                        0.7189989255E+02, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, \
                        0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, \
                        0.0000000000E+00],
            'zeta_0': 0.2194565772E-02, # [J K^-2] F_el (1)
            'xi': 0.4114594460E+00, # [], F_el (2)
            'Tel_0': 0.1620106387E+04, # [K], F_el (3)
            'eta': -.9864575550E+00, # []
            'el_V_0': 0.1620953559E+02 * 1e-6 # [m^3/mol]
            } # F_el (4)
        self.params['a'] = vector_to_array(self.params['a'], self.params['O_f'], self.params['O_theta'])*1e3 # [J/mol]
        Mineral.__init__(self)