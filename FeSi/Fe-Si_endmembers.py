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

'''
Here are some important constants
'''
Pr=1.e5
nA=6.02214e23
voltoa=1.e30

Z_B2=1. # Fm-3m
Z_B20=4. # P2_13


print 21.76*(nA/Z_B2/voltoa)/2.



'''
Now we read in Fischer et al. PVT data for the B20 and B2 FeSi polymorphs
'''
basicerror=0.01 # Angstroms
FeSi_B20_data=[]
FeSi_B2_data=[]
for line in open('data/Fischer_et_al_FeSi_PVT_S2.dat'):
    content=line.strip().split()
    if content[0] != '%' and content[8] != '*' and content[8] != '**':
        if content[7] != '-': # T_B20, Terr_B20, P_B20, Perr_B20, aKbr_B20, aKbrerr_B20, a_B20, a_err_B20, should be good even in the presence of B2
            if float(content[8]) < 1.e-12:
                content[8]=basicerror
            FeSi_B20_data.append([float(content[1]), float(content[2]), float(content[3])*1.e9, float(content[4])*1.e9, float(content[5]), float(content[6]), float(content[7]), float(content[8])])
        if content[0] == 'B2' and content[9] != '-': # T_B2, Terr_B2, P_B2, Perr_B2, aKbr_B2, aKbrerr_B2, a_B2, a_err_B2, only good if B2 is the only phase
            if float(content[10]) < 1.e-12:
                content[10]=basicerror
            FeSi_B2_data.append([float(content[1]), float(content[2]), float(content[3])*1.e9, float(content[4])*1.e9, float(content[5]), float(content[6]), float(content[9]), float(content[10])])


T_B20, Terr_B20, P_B20, Perr_B20, aKbr_B20, aKbrerr_B20, a_B20, a_err_B20 = zip(*FeSi_B20_data)
T_B2, Terr_B2, P_B2, Perr_B2, aKbr_B2, aKbrerr_B2, a_B2, a_err_B2 = zip(*FeSi_B2_data)


'''
B20 PVT fitting
'''

# Pressures and Temperatures
PT_B20=np.array([P_B20,T_B20])
P_B20=np.array(P_B20)

a_B20=np.array(a_B20)
a_err_B20=np.array(a_err_B20)

# Volumes and uncertainties
V_B20=a_B20*a_B20*a_B20*(nA/Z_B20/voltoa)
Verr_B20=3.*a_B20*a_B20*a_err_B20*(nA/Z_B20/voltoa)

# Guesses
guesses=[B20.params['V_0'], B20.params['K_0'], B20.params['a_0']]

popt, pcov = optimize.curve_fit(fit_PVT_data(B20), PT_B20, V_B20, guesses, Verr_B20)
print 'B20 V_0, K_0, a_0:', popt


'''
B2 PVT fitting
'''

# Pressures and Temperatures
PT_B2=np.array([P_B2,T_B2])

a_B2=np.array(a_B2)
a_err_B2=np.array(a_err_B2)

# Volumes and uncertainties
V_B2=a_B2*a_B2*a_B2*(nA/Z_B2/voltoa)/2. # remember B2 is FeSi/2.
Verr_B2=3.*a_B2*a_B2*a_err_B2*(nA/Z_B2/voltoa)/2. # remember B2 is FeSi/2.

# Guesses
guesses=[B2.params['a_0']]

popt, pcov = optimize.curve_fit(fita0(B2), PT_B2, V_B2, guesses, Verr_B2)
print 'B2 V_0, K_0, a_0:', popt

'''
Plotting PVT data at room temperature 
'''

pressures_B20=[]
volumes_calculated_B20=[]
volumes_observed_B20=[]

for i, PT in enumerate(zip(*PT_B20)):
    P, T = PT
    B20.set_state(P, T)
    if (T-300.)*(T-300.) < 0.1:
        pressures_B20.append(P)
        volumes_calculated_B20.append(B20.V)
        volumes_observed_B20.append(V_B20[i])

pressures_B20=np.array(pressures_B20)
volumes_calculated_B20=np.array(volumes_calculated_B20)
volumes_observed_B20=np.array(volumes_observed_B20)
        
pressures_B2=np.linspace(1.e5, 150.e9, 101)
volumes_calculated_B2=np.empty_like(pressures_B2)

T=300.
for i, P in enumerate(pressures_B2):
    B2.set_state(P, T)
    pressures_B2[i] = P
    volumes_calculated_B2[i] = B2.V*2.


plt.plot( pressures_B20/1.e9, volumes_calculated_B20/2., linewidth=1)
plt.plot( pressures_B20/1.e9, volumes_observed_B20/2., marker=".", linestyle="None")
plt.plot( pressures_B2/1.e9, volumes_calculated_B2/2., linewidth=1)

plt.title('Volume fit')
plt.xlabel("Pressure (GPa)")
plt.ylabel("Volume (m^3/mol)")
plt.show()


'''
1 bar properties: First, a gibbs free energy check at room temperature
'''

# Barin
S_0_B20=44.685
G_0_B20=-92175.
H_0_B20=-78852.

print 'B20 Gibbs, checked against Barin'
B20.set_state(1.e5, 298.15)
print B20.gibbs, G_0_B20


'''
Now let's fit and plot Cp
'''

FeSi_B20_Cp_data=[]
for line in open('data/Barin_FeSi_B2_Cp.dat'):
    content=line.strip().split()
    if content[0] != '%':
        FeSi_B20_Cp_data.append(map(float,content))


# Initial guess.
T_Cp_B20, Cp_B20  = zip(*FeSi_B20_Cp_data)
guesses=np.array([1, 1, 1,1])
popt, pcov = optimize.curve_fit(fitCp(B20), np.array(T_Cp_B20), Cp_B20, guesses)
print 'B20 heat capacity'
print popt

temperatures=np.linspace(200., 1700., 100)
Cps=np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    B20.set_state(1.e5, T)
    Cps[i]=B20.C_p


plt.plot( T_Cp_B20, Cp_B20, marker=".", linestyle="None")
plt.plot( temperatures, Cps, linewidth=1)
plt.title('Heat Capacity fit')
plt.xlabel("Temperature (K)")
plt.ylabel("Heat capacity (J/K/mol)")
plt.show()


print 'B20 - B2 equilibrium boundary'

def fit_H_S(temperatures, H_0, S_0):
    B2.params['H_0'] = H_0
    B2.params['S_0'] = S_0
    pressures = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        pressures[i] = optimize.fsolve(eqm_pressure([B20, B2], [1., -2.]),[20.e9], args=(T))[0]/1.e9

    return pressures


transition_pressure = lambda T: (121.6 - 0.0551*T) # Dobson et al, 2003
#transition_pressure = lambda T: 42. + 0.*T # Fischer et al, 2013

temperatures_obs=np.linspace(1600, 2100, 6)
pressures_obs = transition_pressure(temperatures_obs) # GPa

 
print optimize.curve_fit(fit_H_S, temperatures_obs, pressures_obs, [B2.params['H_0'], B2.params['S_0']])


for T in temperatures_obs:
    print T, optimize.fsolve(eqm_pressure([B20, B2], [1., -2.]),[20.e9], args=(T)) 


print ''
'''
Fitting for the Si polymorphs (diamond, bcc, fcc, hcp structures)
'''
print 'SILICON'
print ''
'''
# Check bcc unstable
pressures=np.linspace(20.e9, 300.e9, 100)
for P in pressures:
    Si_hcp.set_state(P, 300.)
    Si_fcc.set_state(P, 300.)
    Si_bcc.set_state(P, 300.)
    
    print Si_bcc.gibbs - Si_hcp.gibbs, Si_bcc.gibbs - Si_fcc.gibbs, '-ve means bcc is stable (bad)'
'''

'''
Fitting Cp for the diamond structure
We'll use this Cp for the other phases
'''

Si_A4_Cp_data=[]
for line in open('data/Barin_Si_A4_Cp.dat'):
    content=line.strip().split()
    if content[0] != '%':
        Si_A4_Cp_data.append(map(float,content))

# Initial guess.
T_Cp_A4, Cp_A4  = zip(*Si_A4_Cp_data)
guesses=np.array([1, 1, 1,1])
popt, pcov = optimize.curve_fit(fitCp(Si_A4), np.array(T_Cp_A4), Cp_A4, guesses)
print popt

temperatures=np.linspace(200., 1700., 100)
Cps=np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    Si_A4.set_state(1.e5, T)
    Cps[i]=Si_A4.C_p


plt.plot( T_Cp_A4, Cp_A4, marker=".", linestyle="None")
plt.plot( temperatures, Cps, linewidth=1)
plt.title('Si (A4 diamond structure)')
plt.xlabel("Temperature (K)")
plt.ylabel("Heat capacity (J/K/mol)")
plt.show()

'''
Fitting thermal expansivity for the diamond structure
We'll use this for the other phases too
'''

Si_A4_a_data=[]
for line in open('data/Si_thermal_expansion.dat'):
    content=line.strip().split()
    if content[0] != '%':
        Si_A4_a_data.append(map(float,content))

T_a_Si, a_Si = zip(*Si_A4_a_data)
a_Si=np.array(a_Si)*1.e-6

guesses=np.array([1.e-6, 800.])
popt, pcov = optimize.curve_fit(fitalpha(Si_A4), np.array(T_a_Si),  a_Si, guesses)
print popt, pcov


temperatures=np.linspace(5., 1700., 100)
alphastar=np.empty_like(temperatures)
Si_A4.set_state(1.e5, 293.)
L_293=np.power(Si_A4.V, 1./3.)

for i, T in enumerate(temperatures):
    deltaT=0.1
    Si_A4.set_state(1.e5, T-deltaT)
    L_T0=np.power(Si_A4.V, 1./3.)
    Si_A4.set_state(1.e5, T+deltaT)
    L_T1=np.power(Si_A4.V, 1./3.)
    DeltaL=L_T1 - L_T0
    DeltaT=2.*deltaT
    alphastar[i]=(1./L_293)*(DeltaL/DeltaT)

plt.plot( temperatures, alphastar, linewidth=1)
plt.plot( T_a_Si, a_Si, marker=".", linestyle="None")
plt.title('Si (A4 diamond structure)')
plt.xlabel("Temperature (K)")
plt.ylabel("Thermal expansivity (10^-6 K^-1)")
plt.show()

'''
Checking EOS data for the diamond structure
'''

VoverV0=0.910
Ptransition=11.3e9

Si_A4.set_state(Pr, 300.)
V0=Si_A4.V
Si_A4.set_state(Ptransition, 300.)
V1=Si_A4.V
print V1/V0, VoverV0

'''
Fitting EOS data for HCP and FCC structures
'''

Si_hcp_volume_data=[]
for line in open('data/si_eos_hcp_V.dat'):
    content=line.strip().split()
    if content[0] != '%':
        Si_hcp_volume_data.append(map(float,content))

Si_hcp_P_V_obs, Si_hcp_V_obs = zip(*Si_hcp_volume_data)

Si_hcp_volume_ab_initio=[]
for line in open('data/si_eos_mujica_hcp.dat'):
    content=line.strip().split()
    if content[0] != '%':
        Si_hcp_volume_ab_initio.append(map(float,content))

Si_hcp_P_V, Si_hcp_V = zip(*Si_hcp_volume_ab_initio)

Si_fcc_volume_data=[]
for line in open('data/si_eos_fcc_V.dat'):
    content=line.strip().split()
    if content[0] != '%':
        Si_fcc_volume_data.append(map(float,content))

Si_fcc_P_V_obs, Si_fcc_V_obs = zip(*Si_fcc_volume_data)

Si_fcc_volume_ab_initio=[]
for line in open('data/si_eos_mujica_fcc.dat'):
    content=line.strip().split()
    if content[0] != '%':
        Si_fcc_volume_ab_initio.append(map(float,content))

Si_fcc_P_V, Si_fcc_V = zip(*Si_fcc_volume_ab_initio)


guesses=np.array([70.e9, 4.])
popt, pcov = optimize.curve_fit(fitK_p0(Si_hcp, Si_A4), np.array(Si_hcp_P_V)*1.e9,  np.array(Si_hcp_V), guesses)
print 'HCP fit:', Si_hcp.params['K_0'], Si_hcp.params['Kprime_0'], Si_hcp.params['V_0']/Si_A4.params['V_0']

guesses=np.array([70.e9, 4.])
popt, pcov = optimize.curve_fit(fitK_p0(Si_fcc, Si_A4), np.array(Si_fcc_P_V)*1.e9,  np.array(Si_fcc_V), guesses)
print 'FCC fit:', Si_fcc.params['K_0'], Si_fcc.params['Kprime_0'], Si_fcc.params['V_0']/Si_A4.params['V_0']

pressures=np.linspace(1.e5, 250.e9, 100)
Si_fcc_volume=np.empty_like(pressures)
Si_hcp_volume=np.empty_like(pressures)
for i, P in enumerate(pressures):
    Si_hcp.set_state(P, 300.)
    Si_hcp_volume[i]=Si_hcp.V/Si_A4.params['V_0']
    Si_fcc.set_state(P, 300.)
    Si_fcc_volume[i]=Si_fcc.V/Si_A4.params['V_0']

plt.plot( pressures/1.e9, Si_hcp_volume, linewidth=1, label='HCP, calculated')
plt.plot( Si_hcp_P_V_obs, Si_hcp_V_obs, marker=".", linestyle="None", label='HCP')
#plt.plot( Si_hcp_P_V, Si_hcp_V, marker=".", linestyle="None", label='HCP, ab_initio')
plt.plot( pressures/1.e9, Si_fcc_volume, linewidth=1, label='FCC, calculated')
plt.plot( Si_fcc_P_V_obs, Si_fcc_V_obs, marker=".", linestyle="None", label='FCC')
plt.title('Si')
plt.xlabel("Pressure (GPa)")
plt.ylabel("Volume/V_0")
plt.legend(loc='upper right')
plt.show()


#### COMPARE FE AND SI

pressures=np.linspace(1.e5, 250.e9, 100)
FeSi_B2_volume=np.empty_like(pressures)
FeSi_fcc_volume=np.empty_like(pressures)
FeSi_hcp_volume=np.empty_like(pressures)
Fe_bcc_volume=np.empty_like(pressures)
Fe_fcc_volume=np.empty_like(pressures)
Fe_hcp_volume=np.empty_like(pressures)
T = 2000.
for i, P in enumerate(pressures):
    B2.set_state(P, T)
    Si_fcc.set_state(P, T)
    Si_hcp.set_state(P, T)

    Fe_bcc.set_state(P, T)    
    Fe_fcc.set_state(P, T)
    Fe_hcp.set_state(P, T)

    Fe_bcc_volume[i]=Fe_bcc.V
    Fe_fcc_volume[i]=Fe_fcc.V
    Fe_hcp_volume[i]=Fe_hcp.V

    FeSi_B2_volume[i]=B2.V*2.
    FeSi_fcc_volume[i]=(Si_fcc.V + Fe_fcc.V)
    FeSi_hcp_volume[i]=(Si_hcp.V + Fe_hcp.V)

plt.plot( pressures/1.e9, Fe_bcc_volume, 'r--', linewidth=1, label='Fe BCC')
plt.plot( pressures/1.e9, Fe_fcc_volume, 'g--', linewidth=1, label='Fe FCC')
plt.plot( pressures/1.e9, Fe_hcp_volume, 'b--', linewidth=1, label='Fe HCP')

plt.plot( pressures/1.e9, FeSi_B2_volume, 'r-', linewidth=1, label='FeSi B2')
plt.plot( pressures/1.e9, FeSi_fcc_volume, 'g-', linewidth=1, label='Fe+Si FCC')
plt.plot( pressures/1.e9, FeSi_hcp_volume, 'b-', linewidth=1, label='Fe+Si HCP')

plt.title('Si vs. Fe, '+str(T)+' K')
plt.xlabel("Pressure (GPa)")
plt.ylabel("Volume/V_0")
plt.legend(loc='upper right')
plt.show()



'''
Finally, we check the relative gibbs free energy between the hcp and fcc phases
hcp -> fcc transition should be at 79+/-2 GPa / 80+/-3 GPa 
'''

print optimize.fsolve(eqm_pressure([Si_fcc, Si_hcp], [1., -1.]), [100.e9], args=(298.15))
