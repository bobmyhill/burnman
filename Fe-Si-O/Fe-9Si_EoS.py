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

atomic_masses=read_masses()


'''
This script takes the 9 wt% HCP data from Fischer et al., 2014 and the EoSs from HCP Fe and Si to estimate a high pressure dV for mixing
'''

class dummy (Mineral):
    def __init__(self):
        formula='Fe0.83566Si0.16434'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Fe-9Si (HCP)',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 49200., # SGTE data
            'S_0': 18.820 + 20.8, # Barin, SGTE data
            'V_0': 7.0e-06 , # hcp Fe is 6.766e-06, Si is 8.8e-6
            'Cp': [22.826, 0.003856857, -353888.416, -0.0596068], # Barin
            'a_0': 7.757256e-06 ,
            'K_0': 57.44e9 , # 72 = Duclos et al 
            'Kprime_0': 5.28 , # Fit to Mujica et al. # 3.9 for Duclos et al 
            'Kdprime_0': -5.28/57.44e9 , # Duclos et al 
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)


'''
First, we import the minerals we will use 
'''

Si_hcp=minerals.Fe_Si_O.Si_hcp_A3()
Fe_hcp=minerals.Myhill_calibration_iron.hcp_iron()
Fe9Si_hcp=dummy()

alloy=9. # wt% Si
mass_Fe=formula_mass({'Fe': 1.}, atomic_masses)
mass_Si=formula_mass({'Si': 1.}, atomic_masses)
molar_fraction_Si=(alloy/mass_Si)/((100.-alloy)/mass_Fe + alloy/mass_Si)
composition=[1.-molar_fraction_Si, molar_fraction_Si]
print composition



'''
Here are some important constants
'''
Pr=1.e5
nA=6.02214e23
voltoa=1.e30
Z_hcp=2. # HCP, P6_3/mmc

'''
Now we read in Fischer et al. PVT data for the hcp Fe-9Si
Remember that for hcp, V = np.sqrt(3)/2.*a*a*a*(c/a)
'''

Fe9Si_hcp_data=[]
basicerror=0.05
for line in open('data/Fischer_et_al_Fe9Si_PVT_S1.dat'):
    content=line.strip().split()
    if content[0] == 'hcp': # T_hcp, Terr_hcp, P_hcp, Perr_hcp, aKbr_hcp, aKbrerr_hcp, a_hcp, a_err_hcp, covera_hcp, covera_err_hcp
        if float(content[10]) < 1.e-6:
            content[10] = basicerror
        if float(content[12]) < 1.e-6:
            content[12] = basicerror
        Fe9Si_hcp_data.append([float(content[1]), float(content[2]), float(content[3])*1.e9, float(content[4])*1.e9, float(content[5]), float(content[6]), float(content[9]), float(content[10]), float(content[11]), float(content[12])])


T_hcp, Terr_hcp, P_hcp, Perr_hcp, aKbr_hcp, aKbrerr_hcp, a_hcp, a_err_hcp, covera_hcp, covera_err_hcp = zip(*Fe9Si_hcp_data)


'''
HCP PVT fitting
'''

# Pressures and Temperatures
P_hcp=np.array(P_hcp)
T_hcp=np.array(T_hcp)
PT_hcp=np.array([P_hcp,T_hcp])

a_hcp=np.array(a_hcp)
a_err_hcp=np.array(a_err_hcp)

covera_hcp=np.array(covera_hcp)
covera_err_hcp=np.array(covera_err_hcp)

# Volumes and uncertainties
V_hcp=(np.sqrt(3.)/2.)*a_hcp*a_hcp*a_hcp*covera_hcp*(nA/Z_hcp/voltoa)
Verr_hcp=V_hcp*(np.sqrt(3.)/2.)*np.sqrt((3.*a_err_hcp/a_hcp)*(3.*a_err_hcp/a_hcp)+(covera_err_hcp/covera_hcp)*(covera_err_hcp/covera_hcp))


# Guesses
guesses=[Fe9Si_hcp.params['V_0'], Fe9Si_hcp.params['K_0'], Fe9Si_hcp.params['a_0']]

popt, pcov = optimize.curve_fit(fit_PVT_data(Fe9Si_hcp), PT_hcp, V_hcp, guesses, Verr_hcp)
print 'HCP params:', popt

diff=np.empty_like(V_hcp)
for i, P in enumerate(P_hcp):
    T=T_hcp[i]
    Fe9Si_hcp.set_state(P, T)
    Si_hcp.set_state(P, T)
    Fe_hcp.set_state(P, T)

    diff[i]=(Fe9Si_hcp.V - (Fe_hcp.V*composition[0] + Si_hcp.V*composition[1]))/Fe9Si_hcp.V

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(P_hcp/1.e9, T_hcp, diff, c='r', marker='o')

ax.set_xlabel('P')
ax.set_ylabel('T')
ax.set_zlabel('Fractional volume difference')

plt.show()

# HCP Si data in Hu et al., 1986
P=41.5e9
T=300.
Si_hcp.set_state(P, T)
a=2.524
b=4.142
print a*a*b*np.sqrt(3)/2.*(nA/Z_hcp/voltoa),  Si_hcp.V

# Olijnyk et al., 1984
P=42.0e9
T=300.
Si_hcp.set_state(P, T)
a=2.444
b=4.152
print a*a*b*np.sqrt(3)/2.*(nA/Z_hcp/voltoa),  Si_hcp.V
