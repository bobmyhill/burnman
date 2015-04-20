# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman import tools
from burnman.mineral import Mineral
from burnman.minerals import Myhill_calibration_iron
from burnman.processchemistry import *
from burnman.chemicalpotentials import *
from burnman import constants

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import optimize
atomic_masses=read_masses()


fo=minerals.HP_2011_ds62.fo()

temperatures=np.linspace(5., 1700., 100)
alphastar_L=np.empty_like(temperatures)
alphastar_V=np.empty_like(temperatures)
fo.set_state(1.e5, 293.)
L_293=np.power(fo.V, 1./3.)
V_293=fo.V

for i, T in enumerate(temperatures):
    deltaT=0.1
    fo.set_state(1.e5, T-deltaT)
    L_T0=np.power(fo.V, 1./3.)
    V_T0=fo.V
    fo.set_state(1.e5, T+deltaT)
    L_T1=np.power(fo.V, 1./3.)
    V_T1=fo.V
    DeltaL=L_T1 - L_T0
    DeltaV=V_T1 - V_T0
    DeltaT=2.*deltaT
    alphastar_L[i]=(1./L_293)*(DeltaL/DeltaT)
    alphastar_V[i]=(1./V_293)*(DeltaV/DeltaT)

plt.plot( temperatures, alphastar_L, linewidth=1, label='Linear')
plt.plot( temperatures, alphastar_V, linewidth=1, label='Volumetric')
plt.title('Heat Capacity fit')
plt.xlabel("Temperature (K)")
plt.ylabel("Thermal expansivity (10^-6 K^-1)")
plt.legend("lower left")
plt.show()


Pr=1.e5
nA=6.02214e23
voltoa=1.e30

Z_B2=1. # Fm-3m
Z_B20=4. # P2_13

basicerror=0.01 # Angstroms
FeSi_B20_data=[]
FeSi_B2_data=[]
for line in open('data/Fischer_et_al_FeSi_PVT_S2.dat'):
    content=line.strip().split()
    if content[0] != '%' and content[8] != '*' and content[8] != '**':
        if content[7] != '-': # T_B20, Terr_B20, P_B20, Perr_B20, aKbr_B20, aKbrerr_B20, a_B20, a_err_B20
            if float(content[8]) < 1.e-12:
                content[8]=basicerror
            FeSi_B20_data.append([float(content[1]), float(content[2]), float(content[3])*1.e9, float(content[4])*1.e9, float(content[5]), float(content[6]), float(content[7]), float(content[8])])
        if content[9] != '-': # T_B2, Terr_B2, P_B2, Perr_B2, aKbr_B2, aKbrerr_B2, a_B2, a_err_B2
            if float(content[10]) < 1.e-12:
                content[10]=basicerror
            FeSi_B2_data.append([float(content[1]), float(content[2]), float(content[3])*1.e9, float(content[4])*1.e9, float(content[5]), float(content[6]), float(content[9]), float(content[10])])


T_B20, Terr_B20, P_B20, Perr_B20, aKbr_B20, aKbrerr_B20, a_B20, a_err_B20 = zip(*FeSi_B20_data)
T_B2, Terr_B2, P_B2, Perr_B2, aKbr_B2, aKbrerr_B2, a_B2, a_err_B2 = zip(*FeSi_B2_data)

class FeSi_B20 (Mineral):
    def __init__(self):
        formula='Fe1.0Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FeSi B20',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -78852. , # Barin
            'S_0': 44.685 , # Barin
            'V_0': 1.359e-05 ,
            'Cp': [38.6770e+01, 0.0217569, -159.151, 0.0060376],
            'a_0': 2.811e-05 ,
            'K_0': 2.057e+11 ,
            'Kprime_0': 4.0 ,
            'Kdprime_0': -4.0/2.057e+11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class FeSi_B2 (Mineral):
    def __init__(self):
        formula='Fe1.0Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FeSi B2',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -78852. , # to fit
            'S_0': 44.685 , # to fit
            'V_0': 1.300e-05 ,
            'Cp': [38.6770e+01, 0.0217569, -159.151, 0.0060376],
            'a_0': 3.064e-05 ,
            'K_0': 2.199e+11 ,
            'Kprime_0': 4.0 ,
            'Kdprime_0': -4.0/2.199e+11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

B20=FeSi_B20()
B2=FeSi_B2()

def fit_PVT_data(mineral):
    def fit_data(PT, V_0, K_0, a_0):
        mineral.params['V_0'] = V_0
        mineral.params['K_0'] = K_0
        Kprime_0 = 4.0
        mineral.params['Kprime_0'] = Kprime_0
        mineral.params['Kdprime_0'] = -Kprime_0/K_0
        mineral.params['a_0'] = a_0

        volumes=[]
        for P, T in zip(*PT):
            mineral.set_state(P, T)
            volumes.append(mineral.V)

        return volumes
    return fit_data


'''
B20 fitting
'''

# Pressures and Temperatures
PT_B20=np.array([P_B20,T_B20])

a_B20=np.array(a_B20)
a_err_B20=np.array(a_err_B20)

# Volumes and uncertainties
V_B20=a_B20*a_B20*a_B20*(nA/Z_B20/voltoa)
Verr_B20=3.*a_B20*a_B20*a_err_B20*(nA/Z_B20/voltoa)

# Guesses
guesses=[B20.params['V_0'], B20.params['K_0'], B20.params['a_0']]

popt, pcov = optimize.curve_fit(fit_PVT_data(B20), PT_B20, V_B20, guesses, Verr_B20)
print 'B20 params:', popt


'''
B2 fitting
'''

# Pressures and Temperatures
PT_B2=np.array([P_B2,T_B2])

a_B2=np.array(a_B2)
a_err_B2=np.array(a_err_B2)

# Volumes and uncertainties
V_B2=a_B2*a_B2*a_B2*(nA/Z_B2/voltoa)
Verr_B2=3.*a_B2*a_B2*a_err_B2*(nA/Z_B2/voltoa)

# Guesses
guesses=[B2.params['V_0'], B2.params['K_0'], B2.params['a_0']]

popt, pcov = optimize.curve_fit(fit_PVT_data(B2), PT_B2, V_B2, guesses, Verr_B2)
print 'B2 params:', popt


# PLOTTING

pressures=[]
volumes_calculated_B20=[]
volumes_observed_B20=[]

for i, PT in enumerate(zip(*PT_B20)):
    P, T = PT
    B20.set_state(P, T)
    if (T-300.)*(T-300.) < 0.1:
        pressures.append(P)
        volumes_calculated_B20.append(B20.V)
        volumes_observed_B20.append(V_B20[i])

pressures_B2=np.linspace(10.e9, 100.e9, 101)
volumes_calculated_B2=np.empty_like(pressures_B2)

T=300.
for i, P in enumerate(pressures_B2):
    B2.set_state(P, T)
    pressures_B2[i] = P
    volumes_calculated_B2[i] = B2.V

plt.plot( np.array(pressures)/1.e9, volumes_calculated_B20, linewidth=1)
plt.plot( np.array(pressures)/1.e9, volumes_observed_B20, marker=".", linestyle="None")
plt.plot( pressures_B2/1.e9, volumes_calculated_B2, linewidth=1)

plt.title('Volume fit')
plt.xlabel("Pressure (GPa)")
plt.ylabel("Volume (m^3/mol)")
plt.show()

# 1 bar properties
# Barin
S_0_B20=44.685
G_0_B20=-92175.
H_0_B20=-78852.

B20.set_state(1.e5, 298.15)
print G_0_B20, B20.gibbs

FeSi_B20_Cp_data=[]
for line in open('data/Barin_FeSi_B2_Cp.dat'):
    content=line.strip().split()
    if content[0] != '%':
        FeSi_B20_Cp_data.append(map(float,content))

def fitCp(mineral):
    def fit(temperatures, a, b, c, d):
        mineral.params['Cp']=[a, b, c, d]
        Cp=np.empty_like(temperatures)
        for i, T in enumerate(temperatures):
            mineral.set_state(1.e5, T)
            Cp[i]=mineral.C_p
        return Cp
    return fit


# Initial guess.
T_Cp_B20, Cp_B20  = zip(*FeSi_B20_Cp_data)
guesses=np.array([1, 1, 1,1])
popt, pcov = optimize.curve_fit(fitCp(B20), np.array(T_Cp_B20), Cp_B20, guesses)
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

# B20-B2 phase boundary
class Si_diamond_A4 (Mineral):
    def __init__(self):
        formula='Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Si A4',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 0. , # Barin
            'S_0': 18.820 , # Barin
            'V_0': 1.20588e-05 , # Hallstedt, 2007
            'Cp': [22.826, 0.003856857, -353888.416, -0.0596068], # Barin
            'a_0': 7.757256e-06 , # Fit to Roberts, 1981
            'K_0': 200.e+9 , # ?
            'Kprime_0': 4.0 , # ?
            'Kdprime_0': -4.0/70.e+9 , # ?
            'T_einstein': 764., # Fit to Roberts, 1981 (would be 516. from 0.8*Tdebye (645 K); see wiki)
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class Si_fcc_A1 (Mineral):
    def __init__(self):
        formula='Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Si fcc A1',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 51000. , # SGTE data
            'S_0': 18.820 + 21.8 , # Barin, SGTE data
            'V_0': 1.300e-05 , # ?
            'Cp': [22.826, 0.003856857, -353888.416, -0.0596068], # Barin
            'a_0': 3.064e-05 , # ?
            'K_0': 2.199e+11 , # ?
            'Kprime_0': 4.0 , # ?
            'Kdprime_0': -4.0/2.199e+11 , # ?
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class Si_hcp_A3 (Mineral):
    def __init__(self):
        formula='Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Si hcp A3',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 49200., # SGTE data
            'S_0': 18.820 + 20.8, # Barin, SGTE data
            'V_0': 1.300e-05 , # ?
            'Cp': [22.826, 0.003856857, -353888.416, -0.0596068], # Barin
            'a_0': 3.064e-05 , # ?
            'K_0': 70.e+9 , # ?
            'Kprime_0': 4.0 , # ?
            'Kdprime_0': -4.0/70.e+9 , # ?
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

Si_A4=Si_diamond_A4()
Si_fcc=Si_fcc_A1()
Si_hcp=Si_hcp_A3()

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
plt.title('Heat Capacity fit')
plt.xlabel("Temperature (K)")
plt.ylabel("Heat capacity (J/K/mol)")
plt.show()


# Thermal expansivity
Si_A4_a_data=[]
for line in open('data/Si_thermal_expansion.dat'):
    content=line.strip().split()
    if content[0] != '%':
        Si_A4_a_data.append(map(float,content))

T_a_Si, a_Si = zip(*Si_A4_a_data)
a_Si=np.array(a_Si)*1.e-6

def fitalpha(mineral):
    def fit(temperatures, alpha, T_einst):
        mineral.params['a_0']=alpha
        mineral.params['T_einstein']=T_einst
        mineral.set_state(1.e5, 293.)
        L_293=np.power(mineral.V, 1./3.)
        alphastar=np.empty_like(temperatures)
        for i, T in enumerate(temperatures):
            deltaT=0.1
            mineral.set_state(1.e5, T-deltaT)
            L_T0=np.power(mineral.V, 1./3.)
            mineral.set_state(1.e5, T+deltaT)
            L_T1=np.power(mineral.V, 1./3.)
            DeltaL=L_T1 - L_T0
            DeltaT=2.*deltaT
            alphastar[i]=(1./L_293)*(DeltaL/DeltaT)
        return alphastar
    return fit


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
plt.title('Heat Capacity fit')
plt.xlabel("Temperature (K)")
plt.ylabel("Thermal expansivity (10^-6 K^-1)")
plt.show()

