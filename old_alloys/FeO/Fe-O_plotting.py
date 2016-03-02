import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import Komabayashi_2014, Myhill_calibration_iron, Fe_Si_O
import numpy as np
from fitting_functions import *
from scipy import optimize, integrate
import matplotlib.pyplot as plt
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass

from Fe_Si_O_liquid_models import *
from HP_convert import *

FeSiO_melt = metallic_Fe_Si_O_liquid()

Fe_fcc = Myhill_calibration_iron.fcc_iron()
Fe_hcp = Myhill_calibration_iron.hcp_iron()
Fe_liq = Myhill_calibration_iron.liquid_iron()

T_ref = 1809.
P_ref = 50.e9
HP_convert(Fe_fcc, 300., 2200., T_ref, P_ref)
HP_convert(Fe_hcp, 300., 2200., T_ref, P_ref)
HP_convert(Fe_liq, 1809., 2400., T_ref, P_ref)



FeO = Fe_Si_O.FeO_solid_HP()
FeO_liq = Fe_Si_O.FeO_liquid_HP()

FeO_Kom = Komabayashi_2014.FeO_solid()
FeO_liq_Kom = Komabayashi_2014.FeO_liquid()


FeSiO_melt.set_composition([0.5, 0.0, 0.5])
pressures = np.linspace(1.e9, 200.e9, 101)
excesses = np.empty_like(pressures)
for i, P in enumerate(pressures):
    FeSiO_melt.set_state(P, 3000.)
    excesses[i] = FeSiO_melt.excess_gibbs

plt.plot(pressures/1.e9, excesses/1.e3)
plt.xlabel("Pressure (GPa)")
plt.ylabel("Gibbs excesses (kJ/mol)")
plt.show()

compositions = np.linspace(0.0, 1.0, 101)
excesses = np.empty_like(pressures)
for i, c in enumerate(compositions):
    FeSiO_melt.set_composition([1.-c, 0.0, c])
    FeSiO_melt.set_state(P, 3000.)
    excesses[i] = FeSiO_melt.gibbs

plt.plot(compositions, excesses/1.e3)
plt.xlabel("X FeO")
plt.ylabel("Gibbs excesses (kJ/mol)")
plt.show()

# First, the equation of state of FeO (B1)
f=open('data/Fischer_2011_FeO_B1_EoS.dat', 'r')
FeO_volumes = []
datastream = f.read()  # We need to open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] != "%" and line[10] != '-':
        FeO_volumes.append(map(float, line))

f=open('data/Campbell_2009_FeO_B1_EoS.dat', 'r')
datastream = f.read()  # We need to open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] != "%" and line[10] != '-':
        FeO_volumes.append(map(float, line))

P, Perr, T, Terr, V_NaCl, Verr_NaCl, V_Fe, Verr_Fe, covera_Fe, coveraerr_Fe, V_FeO, Verr_FeO = zip(*FeO_volumes)

P = np.array(P)*1.e9
Perr = np.array(Perr)*1.e9
T = np.array(T)
Terr = np.array(Terr)
V_FeO = np.array(V_FeO)*1.e-6
Verr_FeO = np.array(Verr_FeO)*1.e-6

PT = [P, T]
guesses = [FeO.params['K_0'], FeO.params['a_0']]
popt, pcov = optimize.curve_fit(fit_PVT_data_Ka(FeO), PT, V_FeO, guesses, Verr_FeO)
for i, p in enumerate(popt):
    print popt[i], '+/-', np.sqrt(pcov[i][i])

# Now for the liquid

pressures = np.linspace(1.e9, 200.e9, 20)
#pressures = [10.e9, 20.e9, 30.e9, 65.e9]
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = optimize.fsolve(eqm_temperature([FeO, FeO_liq], [1., -1.]), [2000.], args=(P))[0]
    print P/1.e9, temperatures[i]

plt.plot(pressures, temperatures)
#plt.plot(pressures, temperatures_Komabayashi)
plt.show()




# READ IN DATA
eutectic_PT = []

f=open('data/Fe_FeO_eutectic_temperature.dat', 'r')
datastream = f.read()  # We need to open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] != "%":
        eutectic_PT.append([float(line[0])*1.e9, float(line[1]), 
                            float(line[2]), float(line[3])])


eutectic_PTc = []
f=open('data/Fe_FeO_eutectic.dat', 'r')
datastream = f.read()  # We need to open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] != "%":
        eutectic_PTc.append([float(line[1])*1.e9, 2.0e9, 
                             float(line[2]), float(line[3]), 
                             float(line[4])/100., float(line[5])/100.])

solvus_PTcc = []
f=open('data/Fe_FeO_solvus.dat', 'r')
datastream = f.read()  # We need to open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] != "%":
        solvus_PTcc.append([float(line[1])*1.e9, 2.0e9, 
                            float(line[2]), float(line[3]), 
                            float(line[4])/100., float(line[5])/100.,
                            float(line[8])/100., float(line[9])/100.])



####
# PLOT COMPARISONS WITH PUBLISHED DATA
####
eutectic_PT = np.array(eutectic_PT).T
eutectic_PTc = np.array(eutectic_PTc).T
solvus_PTcc = np.array(solvus_PTcc).T

def eqm_liquid(cT, P, model, Fe_phase, FeO_phase):
    c, T = cT

    FeSiO_melt.set_composition([1.-c, 0., c])
    FeSiO_melt.set_state(P, T)
    partial_excesses = model.excess_partial_gibbs
    equations = [ Fe_phase.calcgibbs(P, T) - FeSiO_melt.partial_gibbs[0],
                  FeO_phase.calcgibbs(P, T) - FeSiO_melt.partial_gibbs[2]]
    return equations



def eqm_two_liquid(cc, P, T):
    c1, c2 = cc

    FeSiO_melt.set_composition([1.-c1, 0., c1])
    FeSiO_melt.set_state(P, T)

    partial_excesses_1 = FeSiO_melt.excess_partial_gibbs
    FeSiO_melt.set_composition([1.-c2, 0., c2])
    FeSiO_melt.set_state(P, T)
    partial_excesses_2 = FeSiO_melt.excess_partial_gibbs
    equations = [ partial_excesses_1[0] - partial_excesses_2[0],
                  partial_excesses_1[2] - partial_excesses_2[2]]
    return equations


# Plot solvus
temperatures = [2173., 2273., 2373., 2473., 2573.]
pressures = np.linspace(1.e5, 28.e9, 30)

compositions_1 = np.empty_like(pressures)
compositions_2 = np.empty_like(pressures)


for T in temperatures:
    c1=0.01
    c2=0.99
    for i, P in enumerate(pressures):

        c1, c2 = optimize.fsolve(eqm_two_liquid, [c1, c2], 
                                 args=(P, T), factor = 0.1, xtol=1.e-12)
        compositions_1[i] = c1
        compositions_2[i] = c2
    plt.plot(compositions_1, pressures/1.e9, label='Metallic at '+str(T)+' K')
    plt.plot(compositions_2, pressures/1.e9, label='Ionic at '+str(T)+' K')

plt.plot(solvus_PTcc[4], solvus_PTcc[0]/1.e9, marker='o', linestyle='None')
plt.plot(solvus_PTcc[6], solvus_PTcc[0]/1.e9, marker='o', linestyle='None')

plt.legend(loc='lower right')
plt.show()



# Plot eutectic temperatures and compositions
pressures = np.linspace(30.e9, 250.e9, 100)
eutectic_compositions = np.empty_like(pressures)
eutectic_temperatures = np.empty_like(pressures)

c, T = [0.5, 4000.]
for i, P in enumerate(pressures):
    print c, T
    c, T = optimize.fsolve(eutectic_liquid, [c, T], 
                           args=(P, intermediate_0, intermediate_1, Fe_hcp, FeO))
    print P, T, model.enthalpy_interaction
    eutectic_compositions[i] = c
    eutectic_temperatures[i] = T


plt.plot(pressures/1.e9, eutectic_compositions)
plt.plot(eutectic_PTc[0]/1.e9, eutectic_PTc[4], marker='o', linestyle='None', label='Model')
plt.legend(loc='lower right')
plt.show()

plt.plot(pressures/1.e9, eutectic_temperatures)
plt.plot(eutectic_PT[0]/1.e9, eutectic_PT[2], marker='o', linestyle='None', label='Model')
plt.plot(eutectic_PTc[0]/1.e9, eutectic_PTc[2], marker='o', linestyle='None', label='Model')
plt.legend(loc='lower right')
plt.show()
