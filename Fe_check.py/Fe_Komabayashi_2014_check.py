import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals

fcc = minerals.Komabayashi_2014.fcc_iron()
hcp = minerals.Komabayashi_2014.hcp_iron()
liq = minerals.Komabayashi_2014.liquid_iron()

fcc.set_state(1.e9, 1000.)
print fcc.gibbs



list_minerals=[fcc, hcp, liq]

PT = np.array([[4.e9 , 1926],
               [24.e9, 2357],
               [38.e9, 2594],
               [10.e9, 2118],
               [16.e9, 2210]])

print 'Pressure (GPa) Temperature (K) Entropy (J/K/mol)'
for mineral in list_minerals:
    print mineral.params['name'],
print 

for (P, T) in PT:
    print P/1.e9, T,
    for mineral in list_minerals:
        mineral.set_state(P, T)
        print mineral.S,
    print 

'''
P, T = PT[3]
fcc.set_state(P, T)
S = fcc.S
gibbs0 = fcc.gibbs

dT = 1.e-1
fcc.set_state(P, T+dT)
S = fcc.S
gibbs1 = fcc.gibbs

print -(gibbs1 - gibbs0)/dT
'''
