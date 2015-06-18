import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    DKS_2013_liquids, \
    DKS_2013_solids, \
    SLB_2011
from burnman import constants
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


phases = [DKS_2013_liquids.SiO2_liquid(),
         DKS_2013_liquids.MgSiO3_liquid(),
         DKS_2013_liquids.MgSi2O5_liquid(),
         DKS_2013_liquids.MgSi3O7_liquid(),
         DKS_2013_liquids.MgSi5O11_liquid(),
         DKS_2013_liquids.Mg2SiO4_liquid(),
         DKS_2013_liquids.Mg3Si2O7_liquid(),
         DKS_2013_liquids.Mg5SiO7_liquid(),
         DKS_2013_liquids.MgO_liquid()
         ]


pressure = 25.e9 # Pa
temperature = 3000. # K

MgO_liq = DKS_2013_liquids.MgO_liquid()
SiO2_liq = DKS_2013_liquids.SiO2_liquid()

MgO_liq.set_state(pressure, temperature)
SiO2_liq.set_state(pressure, temperature)
MgO_gibbs = MgO_liq.gibbs
SiO2_gibbs = SiO2_liq.gibbs

MgO_V = MgO_liq.V
SiO2_V = SiO2_liq.V

fSis=[]
Hexs=[]
Vexs=[]
for phase in phases:
    print phase.params['name']

    nSi = phase.params['formula']['Si']
    nMg = phase.params['formula']['Mg']

    sum_cations = nSi+nMg
    fSi=nSi/sum_cations

    phase.set_state(pressure, temperature)
    Hex = ((phase.gibbs)/sum_cations - (fSi*SiO2_gibbs + (1.-fSi)*MgO_gibbs))/4.

    Vex = (phase.V)/sum_cations - (fSi*SiO2_V + (1.-fSi)*MgO_V)
    fSis.append(fSi)
    Hexs.append(Hex)
    Vexs.append(Vex)


plt.plot(fSis, Hexs, marker='o', linestyle='None')
plt.show()

