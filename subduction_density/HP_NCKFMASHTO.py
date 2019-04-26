# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2017 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
HP_2011 (ds-62)
Minerals from Holland and Powell 2011 and references therein
Update to dataset version 6.2
The values in this document are all in S.I. units,
unlike those in the original tc-ds62.txt
File autogenerated using HPdata_to_burnman.py
"""

from ..mineral import Mineral
from ..solidsolution import SolidSolution
from ..combinedmineral import CombinedMineral
from ..solutionmodel import *
from ..processchemistry import dictionarize_formula, formula_mass
from .HP_2011_ds62 import *
"""
SOLID SOLUTIONS

N.B. VERY IMPORTANT: The excess entropy term in the regular solution model has the opposite sign to the values in Holland and Powell, 2011. This is consistent with its treatment as an excess entropy term (G=H-T*S+P*V), rather than a thermal correction to the interaction parameter (W=W+T*W_T+P*W_P).
"""

class garnet(SolidSolution):

    def __init__(self, molar_amounts=None):
        self.name = 'garnet'
        self.endmembers = [[py(), '[Mg]3[Al]2Si3O12'],
                           [alm(), '[Fe]3[Al]2Si3O12'],
                           [gr(), '[Ca]3[Al]2Si3O12'],
                           [kho(), '[Mg]3[Fe]2Si3O12']]
        self.solution_type = 'asymmetric'
        self.alphas = [1.0, 1.0, 2.7, 1.0]
        self.energy_interaction = [[2.5e3, 31.e3, 5.4e3],
                                   [5.e3, 22.6e3],
                                   [-15.3e3]]
        SolidSolution.__init__(self, molar_amounts=molar_amounts)

class kho(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'khoharite',
                                 mineral_list = [andr(), gr(), py()],
                                 molar_amounts = [1., -1., 1.],
                                 free_energy_adjustment=[27.e3, 0., 0.])
        
class plagioclase(SolidSolution):
    def __init__(self, molar_amounts=None):
        self.name = 'plagioclase'
        self.endmembers = [[san(), '[K]'],
                           [abh(), '[Na]'],
                           [anmod(), '[Ca]']]
        self.solution_type = 'asymmetric'
        self.alphas = [1.0, 0.643, 1.0]
        self.energy_interaction = [[25.1e3, 40.e3],
                                   [3.1e3]]
        self.entropy_interaction = [[10.8, 0.],
                                    [0.]] # -1 -kJ/K = 1e3 J/K
        self.volume_interaction = [[0.338e-5, 0.],
                                   [0.]] # 1 kJ/kbar = 1.e-5 J/Pa
        SolidSolution.__init__(self, molar_amounts=molar_amounts)

class anmod(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'anorthite',
                                 mineral_list = [an()],
                                 molar_amounts = [1.],
                                 free_energy_adjustment=[7.03e3, 4.66, 0.])        


class epidote(SolidSolution):
    def __init__(self, molar_amounts=None):
        self.name = 'epidote'
        self.endmembers = [[cz(), '[Al][Al]'],
                           [ep(), '[Al][Fef]'],
                           [fep(), '[Fef][Fef]']] # note order.
        self.solution_type = 'symmetric'
        self.energy_interaction = [[1.e3, 3.e3],
                                   [1.e3]]
        SolidSolution.__init__(self, molar_amounts=molar_amounts)


class margarite(SolidSolution):
    def __init__(self, molar_amounts=None):
        self.name = 'margarite'
        self.endmembers = [[mumod(),   '[K][Al][Al][Si1/2Al1/2]2'],
                           [celmod(),  '[K][Mg][Al][Si]2'],
                           [fcelmod(), '[K][Fe][Al][Si]2'],
                           [pamod(),   '[Na][Al][Al][Si1/2Al1/2]2'],
                           [ma(),   '[Ca][Al][Al][Al]2'],
                           [fmu(),  '[K][Al][Fef][Si1/2Al1/2]2']]
        self.solution_type = 'asymmetric'
        self.alphas = [0.63, 0.63, 0.63, 0.37, 0.63, 0.63]
        self.energy_interaction = [[0.e3, 0.e3, 10.12e3, 34.e3, 0.e3],
                                   [0.e3, 45.e3, 50.e3, 0.e3],
                                   [45.e3, 50.e3, 0.e3],
                                   [18.e3, 30.e3],
                                   [35.e3]]
        
        self.entropy_interaction = [[0., 0., 3.4, 0., 0.],
                                    [0., 0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 0.],
                                    [0.]]

        self.volume_interaction = [[0.2e-5, 0.2e-5, 0.353e-5, 0., 0.],
                                   [0., 0.25e-5, 0., 0.],
                                   [0.25e-5, 0., 0.],
                                   [0., 0.],
                                   [0.]]

        
        SolidSolution.__init__(self, molar_amounts=molar_amounts)


class muscovite(SolidSolution):
    def __init__(self, molar_amounts=None):
        self.name = 'muscovite'
        self.endmembers = [[mu(),   '[K][Al][Al][Si1/2Al1/2]2'],
                           [cel(),  '[K][Mg][Al][Si]2'],
                           [fcel(), '[K][Fe][Al][Si]2'],
                           [pa(),   '[Na][Al][Al][Si1/2Al1/2]2'],
                           [mamod(),   '[Ca][Al][Al][Al]2'],
                           [fmu(),  '[K][Al][Fef][Si1/2Al1/2]2']]
        self.solution_type = 'asymmetric'
        self.alphas = [0.63, 0.63, 0.63, 0.37, 0.63, 0.63]
        self.energy_interaction = [[0.e3, 0.e3, 10.12e3, 34.e3, 0.e3],
                                   [0.e3, 45.e3, 50.e3, 0.e3],
                                   [45.e3, 50.e3, 0.e3],
                                   [18.e3, 30.e3],
                                   [35.e3]]
        
        self.entropy_interaction = [[0., 0., 3.4, 0., 0.],
                                    [0., 0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 0.],
                                    [0.]]

        self.volume_interaction = [[0.2e-5, 0.2e-5, 0.353e-5, 0., 0.],
                                   [0., 0.25e-5, 0., 0.],
                                   [0.25e-5, 0., 0.],
                                   [0., 0.],
                                   [0.]]

        
        SolidSolution.__init__(self, molar_amounts=molar_amounts)


class mumod(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'muscovite',
                                 mineral_list = [mu()],
                                 molar_amounts = [1.],
                                 free_energy_adjustment=[1.e3, 0., 0.])  
class celmod(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'celadonite',
                                 mineral_list = [cel()],
                                 molar_amounts = [1.],
                                 free_energy_adjustment=[5.e3, 0., 0.])  
class fcelmod(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'ferroceladonite',
                                 mineral_list = [fcel()],
                                 molar_amounts = [1.],
                                 free_energy_adjustment=[5.e3, 0., 0.])
          
class pamod(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'paragonite',
                                 mineral_list = [pa()],
                                 molar_amounts = [1.],
                                 free_energy_adjustment=[4.e3, 0., 0.]) ]
        
class mamod(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'margarite',
                                 mineral_list = [ma()],
                                 molar_amounts = [1.],
                                 free_energy_adjustment=[6.5e3, 0., 0.]) ]
        
class fmu(CombinedMineral):
    def __init__(self):
        CombinedMineral.__init__(self,
                                 name = 'ferrimuscovite',
                                 mineral_list = [andradite(), gr(), mu()],
                                 molar_amounts = [0.5, -0.5, 1.],
                                 free_energy_adjustment=[25.e3, 0., 0.])

        


biotite
orthopyroxene
sapphirine
cordierite
staurolite
chlorite
chloritoid
spinel
magnetite
ilmenite
magnetite1
