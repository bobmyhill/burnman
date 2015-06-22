import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    DKS_2013_liquids_tweaked, \
    DKS_2013_solids, \
    SLB_2011
from burnman import constants
import numpy as np
from scipy.optimize import fsolve, curve_fit
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class MgO_SiO2_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Subregular MgO-SiO2 liquid'
        self.type='subregular'

        self.endmembers = [[DKS_2013_liquids_tweaked.MgO_liquid(), '[Mg]O2'],
                           [DKS_2013_liquids_tweaked.SiO2_liquid(), '[Si]O2']]
                           

        self.enthalpy_interaction = [[[-100000., -100000.]]]
        self.volume_interaction   = [[[0., 0.]]]
        self.entropy_interaction  = [[[0., 0.]]]
                        
        burnman.SolidSolution.__init__(self, molar_fractions)


per=SLB_2011.periclase()
fo=SLB_2011.forsterite()
en=SLB_2011.enstatite()
stv=SLB_2011.stishovite()

per.set_state(1.e5, 1000.)
print per.gibbs
# Melting checks
checks=[[per, 1.e5, 3070.],
        [fo, 1.e5, 2163.],
        [en, 1.e5, 1560.+273.15], # just unstable
        [stv, 14.e9, 3120.],
        [en, 13.e9, 2270.+273.15]]

liq=MgO_SiO2_liquid()


for check in checks:
    phase, pressure, temperature = check
    if 'Mg' in phase.params['formula']:
        nMg = phase.params['formula']['Mg']
    else:
        nMg = 0.
    if 'Si' in phase.params['formula']:
        nSi = phase.params['formula']['Si']
    else:
        nSi = 0.
    c = nSi/(nMg + nSi)
    liq.set_composition([1.-c, c])
    liq.set_state(pressure, temperature)
    phase.set_state(pressure, temperature)
    print phase.params['name'], liq.gibbs - phase.gibbs/(nSi+nMg)

#components=[fo.params['formula']]
#chem_potentials=burnman.chemicalpotentials.chemical_potentials([liq], components)

