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

from HP_convert import *
from fitting_functions import *
from scipy import optimize


'''
First, we import the minerals we will use 
'''
B20=minerals.Fe_Si_O.FeSi_B20()
B2=minerals.Fe_Si_O.FeSi_B2()
FeSi_melt=minerals.Fe_Si_O.FeSi_liquid()
Si_A4=minerals.Fe_Si_O.Si_diamond_A4()
Si_fcc=minerals.Fe_Si_O.Si_fcc_A1()
Si_hcp=minerals.Fe_Si_O.Si_hcp_A3()
Si_bcc=minerals.Fe_Si_O.Si_bcc_A2()
Si_liq=minerals.Fe_Si_O.Si_liquid()
Fe_bcc=minerals.Myhill_calibration_iron.bcc_iron()
Fe_fcc=minerals.Myhill_calibration_iron.fcc_iron()
Fe_hcp=minerals.Myhill_calibration_iron.hcp_iron()
Fe_liq=minerals.Myhill_calibration_iron.liquid_iron_HP()

HP_convert(Si_liq, 1000., 3000., 1800., 10.e9)

# Let's get the melting curve of FeSi first of all...

FeSi_melting_data = []
for line in open('data/Lord_FeSi_melting.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if float(content[0]) < 30.:
            FeSi_melting_data.append([B2, float(content[0])*1.e9, float(content[1]), float(content[2])*1.e9, float(content[3])])
        else:
            FeSi_melting_data.append([B20, float(content[0])*1.e9, float(content[1]), float(content[2])*1.e9, float(content[3])])

FeSi_phase, P_melting, T_melting, Perr_melting, Terr_melting = zip(*FeSi_melting_data)

phaseP = zip(*[FeSi_phase, P_melting])

def melting_curve(data, V_0, K_0, Kprime_0):
    FeSi_melt.params['V_0'] = V_0
    FeSi_melt.params['K_0'] = K_0
    FeSi_melt.params['Kprime_0'] = Kprime_0
    FeSi_melt.params['Kdprime_0'] = -4.0/K_0

    print V_0, K_0, Kprime_0
    temperatures = []
    for FeSi_phase, P in data:
        if FeSi_phase == B20:
            temperatures.append(optimize.fsolve(eqm_temperature([FeSi_phase, FeSi_melt], [1., -2.]), [2000.], args=(P))[0])
        else:
            temperatures.append(optimize.fsolve(eqm_temperature([FeSi_phase, FeSi_melt], [1., -1.]), [2000.], args=(P))[0])
    return temperatures

guesses = [FeSi_melt.params['V_0'], 100.e9, 4.]


#popt, pcov = optimize.curve_fit(melting_curve, phaseP, T_melting, guesses, Terr_melting)
#print popt, pcov

#FeSi_melt.params['V_0'] = 7.8e-6
#FeSi_melt.params['K_0'] = 115.e9
FeSi_melt.set_state(1.e5, 1683.)
B20.set_state(1.e5, 1683.)

print FeSi_melt.gibbs*2., B20.gibbs
dTdP = 170.e-9
FeSi_V_melting = (FeSi_melt.S*2. - B20.S)*dTdP
print (B20.V + FeSi_V_melting)/2.

FeSi_melt.set_state(1.e5, 1683.)
Fe_liq.set_state(1.e5, 1683.)
Si_liq.set_state(1.e5, 1683.)

H_ex = 4.*(FeSi_melt.H - 0.5*(Fe_liq.H + Si_liq.H))
S_ex = 4.*(FeSi_melt.S - 0.5*(Fe_liq.S + Si_liq.S) + constants.gas_constant*np.log(0.5))
V_ex = 4.*(FeSi_melt.V - 0.5*(Fe_liq.V + Si_liq.V))
print S_ex, V_ex

class liquid_Fe_Si(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='liquid Fe-Si solid solution'
        self.type='subregular'
        self.endmembers = [[Fe_liq, '[Fe]'],[Si_liq, '[Si]']]
        self.enthalpy_interaction=[[[H_ex, H_ex]]]
        self.volume_interaction=[[[V_ex, V_ex]]]
        self.entropy_interaction=[[[S_ex, S_ex]]]
        burnman.SolidSolution.__init__(self, molar_fractions)

Fe_Si_liq = liquid_Fe_Si()
Fe_Si_liq.set_composition([0.5, 0.5])

FeSi_melt.set_state(1.e5, 1683.)
Fe_Si_liq.set_state(1.e5, 1683.)
print FeSi_melt.V, Fe_Si_liq.V
print FeSi_melt.S, Fe_Si_liq.S
print FeSi_melt.H, Fe_Si_liq.H
print FeSi_melt.K_T/1.e9, Fe_Si_liq.K_T/1.e9

pressures = np.linspace(1.e5, 170.e9, 40)
temperatures = np.empty_like(pressures)
temperatures_ss = np.empty_like(pressures)
V_ex = np.empty_like(pressures)
for i, P in enumerate(pressures):
    if P < 30.e9:
        temperatures[i] = optimize.fsolve(eqm_temperature([B20, FeSi_melt], [1., -2.]), [2000.], args=(P))[0]
        temperatures_ss[i] = optimize.fsolve(eqm_temperature([B20, Fe_Si_liq], [1., -2.]), [2000.], args=(P))[0]
    else:
        temperatures[i] = optimize.fsolve(eqm_temperature([B2, FeSi_melt], [1., -1.]), [2000.], args=(P))[0]
        temperatures_ss[i] = optimize.fsolve(eqm_temperature([B2, Fe_Si_liq], [1., -1.]), [2000.], args=(P))[0]

    print P/1.e9, temperatures[i], FeSi_melt.V, Fe_Si_liq.V
    FeSi_melt.set_state(P, temperatures[i])
    Fe_liq.set_state(P, temperatures[i])
    Si_liq.set_state(P, temperatures[i])
    V_ex[i] = FeSi_melt.V - 0.5*(Fe_liq.V - Si_liq.V)

plt.plot(pressures/1.e9, temperatures, label='Fitted curve')
plt.plot(pressures/1.e9, temperatures_ss, label='Fitted curve (solid solution)')
plt.plot(np.array(P_melting)/1.e9, np.array(T_melting), 'o', linestyle='None', label='Data')
plt.legend(loc='lower left')
plt.show()


plt.plot(pressures/1.e9, V_ex, label='V excess')
plt.legend(loc='lower left')
plt.show()





exit()



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

