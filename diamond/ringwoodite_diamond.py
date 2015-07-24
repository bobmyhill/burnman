# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.


import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from scipy.optimize import fsolve

# Create ringwoodite solid solution
class mg_fe_ringwoodite(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='ringwoodite'
        self.type='symmetric'
        self.endmembers = [[minerals.SLB_2011.mg_ringwoodite(), '[Mg]2SiO4'],
                           [minerals.SLB_2011.fe_ringwoodite(), '[Fe]2SiO4']]
        self.enthalpy_interaction=[[9.34084e3]]
        
        burnman.SolidSolution.__init__(self, molar_fractions)

# Create wadsleyite solid solution
class mg_fe_wadsleyite(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='wadsleyite'
        self.type='symmetric'
        self.endmembers = [[minerals.SLB_2011.mg_wadsleyite(), '[Mg]2SiO4'],
                           [minerals.SLB_2011.fe_wadsleyite(), '[Fe]2SiO4']]
        self.enthalpy_interaction=[[16.74718e3]]

        burnman.SolidSolution.__init__(self, molar_fractions)


# Initialise minerals
ringwoodite = mg_fe_ringwoodite([0.9, 0.1])
wadsleyite = mg_fe_wadsleyite([0.9, 0.1])
diamond =  minerals.HP_2011_ds62.diam()

# Pressure of inclusion
Pf=19.e9 
Tf=1500. + 273.15
diamond.set_state(Pf, Tf)
ringwoodite.set_state(Pf, Tf)

volume = diamond.V
print volume, 'm^3/mol'

moles_ringwoodite = diamond.V / ringwoodite.V
print moles_ringwoodite


# New external pressure
P=7.e9 
T=1400. + 273.15

diamond.set_state(P, T)
inclusion_volume = diamond.V # note that this is a minimum - the excess pressure inside the cell will tend to expand the volume elastically
molar_ringwoodite_volume = inclusion_volume / moles_ringwoodite

def find_pressure(pressure, volume, temperature, mineral):
    mineral.set_state(pressure[0], temperature)
    return volume - mineral.V


print fsolve(find_pressure, P, args=(molar_ringwoodite_volume, T, ringwoodite))[0]/1.e9, 'GPa'
print ringwoodite.gibbs - P*ringwoodite.V
print fsolve(find_pressure, P, args=(molar_ringwoodite_volume, T, wadsleyite))[0]/1.e9, 'GPa'
print wadsleyite.gibbs - P*wadsleyite.V
