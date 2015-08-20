# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman import tools
from burnman.mineral import Mineral
from burnman.chemicalpotentials import *
from burnman import constants

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from fitting_functions import *
from scipy import optimize


'''
First, we import the minerals we will use 
'''
B20=minerals.Fe_Si_O.FeSi_B20()
B2=minerals.Fe_Si_O.FeSi_B2()
Si_A4=minerals.Fe_Si_O.Si_diamond_A4()
Si_fcc=minerals.Fe_Si_O.Si_fcc_A1()
Si_hcp=minerals.Fe_Si_O.Si_hcp_A3()
Si_bcc=minerals.Fe_Si_O.Si_bcc_A2()

Fe_bcc=minerals.Myhill_calibration_iron.bcc_iron()
Fe_fcc=minerals.Myhill_calibration_iron.fcc_iron()
Fe_hcp=minerals.Myhill_calibration_iron.hcp_iron()





#B20+B2	2053	138	31.1	0.8	3.406	0.001	4.358	0.003	2.712	0.003
#B20+B2	2275	158	31.7	0.9	3.403	0.002	4.366	0.008	2.714	0.003
#B20+B2	2346	164	31.6	1.0	3.405	0.002	4.379	0.002	2.708	0.002
#B20+B2	2134	147	31.7	0.9	3.401	0.002	4.379	0.006	2.706	0.001
#B20+B2	1951	124	31.3	0.8	3.403	0.001	4.368	0.002	2.707	0.004
#B20+B2	1800	129	31.3	0.7	3.401	0.001	4.363	0.003	2.709	0.004
#B20+B2	1530	116	31.0	0.6	3.401	0.001	4.357	0.002	2.700	0.001
#B20+B2	1432	116	31.0	0.5	3.399	0.001	4.354	0.004	2.697	0.001

basicerror=0.01 # Angstroms
FeSi_B20_data=[]
FeSi_B2_data=[]
for line in open('data/Fischer_et_al_FeSi_PVT_S2.dat'):
    content=line.strip().split()
    if content[0] != '%' and content[8] != '*' and content[8] != '**':
        if content[0] == 'B20+B2': # T_B2, Terr_B2, P_B2, Perr_B2, aKbr_B2, aKbrerr_B2, a_B2, a_err_B2
            if float(content[10]) < 1.e-12:
                content[10]=basicerror
            FeSi_B2_data.append([float(content[1]), float(content[2]), float(content[3])*1.e9, float(content[4])*1.e9, float(content[5]), float(content[6]), float(content[9]), float(content[10])])


T_B2, Terr_B2, P_B2, Perr_B2, aKbr_B2, aKbrerr_B2, a_B2, a_err_B2 = zip(*FeSi_B2_data)

'''
Here are some important constants
'''
Pr=1.e5
nA=6.02214e23
voltoa=1.e30

Z_B2=1. # Fm-3m
Z_B20=4. # P2_13

a_B2=np.array(a_B2)
a_err_B2=np.array(a_err_B2)

# Volumes and uncertainties
V_B2_obs=a_B2*a_B2*a_B2*(nA/Z_B2/voltoa)/2. # remember B2 is FeSi/2.
Verr_B2=3.*a_B2*a_B2*a_err_B2*(nA/Z_B2/voltoa)/2. # remember B2 is FeSi/2.


for i, T in enumerate(T_B2):
    B20.set_state(P_B2[i], T_B2[i])
    B2.set_state(P_B2[i], T_B2[i])
    Fe_bcc.set_state(P_B2[i], T_B2[i])

    print P_B2[i]/1.e9, T_B2[i], (V_B2_obs[i] - Fe_bcc.V) / (B2.V-Fe_bcc.V)
