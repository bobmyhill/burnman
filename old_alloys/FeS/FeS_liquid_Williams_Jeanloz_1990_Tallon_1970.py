# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

from scipy.optimize import fsolve, curve_fit
import burnman
from HP_convert import *
from listify_xy_file import *
from fitting_functions import *
from scipy.interpolate import interp1d

tro = burnman.minerals.Fe_Si_O.FeS_IV_V_Evans()
FeS_IV_HP = burnman.minerals.Fe_Si_O.FeS_IV_HP()
FeS_VI = burnman.minerals.Fe_Si_O.FeS_VI()
liq_FeS = burnman.minerals.Fe_Si_O.FeS_liquid()

melting_curve_data = listify_xy_file('data/FeS_melting_curve_Williams_Jeanloz_1990.dat')
melting_temperature = interp1d(melting_curve_data[0]*1.e9, 
                               melting_curve_data[1], 
                               kind='cubic')


FeS_VI.set_state(1.e5, 1463.)
Cp = liq_FeS.params['Cp']
print liq_FeS.params['P_0']/1.e9, liq_FeS.params['T_0']

P = 36.e9
T = melting_temperature(P)
HP_convert(liq_FeS, 300., 2200., T, P)
liq_FeS.params['Cp'] = [43.5, 0., 0., 0.]


dP = 100. # Pa
dT = melting_temperature(P + dP/2.) - melting_temperature(P - dP/2.)
dTdP = dT/dP
FeS_VI.set_state(P, T)
aK_T = FeS_VI.alpha*FeS_VI.K_T
Sfusion = burnman.constants.gas_constant*np.log(2.) / (1. - aK_T*dTdP)
Vfusion = Sfusion*dTdP

liq_FeS.params['S_0'] = FeS_VI.S + Sfusion
liq_FeS.params['V_0'] = FeS_VI.V + Vfusion


liq_FeS.params['K_0'] = 150.e9
liq_FeS.params['Kprime_0'] = 4.9
liq_FeS.params['a_0'] = 7.0e-5


liq_FeS.set_state(1.e5, 1463.)
print liq_FeS.C_p, 'aim for 62.5'
liq_FeS.set_state(1.e5, 2463.)
print liq_FeS.C_p, 'aim for 62.5'


print P, T

FeS_VI.set_state(P, T+0.0001)
liq_FeS.set_state(P, T+0.0001)
print FeS_VI.gibbs,  liq_FeS.gibbs, liq_FeS.S
liq_FeS.params['H_0'] = FeS_VI.gibbs - T*liq_FeS.S
liq_FeS.set_state(P, T)
print liq_FeS.gibbs

# Tallon (1980) suggested that melting of simple substances was associated with an entropy change of
# Sfusion = burnman.constants.gas_constant*np.log(2.) + a*K_T*Vfusion
# Realising also that dT/dP = Vfusion/Sfusion, we can express the entropy 
# and volume of fusion in terms of the melting curve:
# Sfusion = burnman.constants.gas_constant*np.log(2.) / (1. - a*K_T*dTdP)
# Vfusion = Sfusion*dT/dP

pressures = np.linspace(36.e9, 120.e9, 101)
Sfusion = np.empty_like(pressures)
Vfusion = np.empty_like(pressures)
Smelt = np.empty_like(pressures)
Vmelt = np.empty_like(pressures)
for i, P in enumerate(pressures):
    dP = 100. # Pa
    dT = melting_temperature(P + dP/2.) - melting_temperature(P - dP/2.)
    dTdP = dT/dP
    T = melting_temperature(P)
    FeS_VI.set_state(P, T)
    aK_T = FeS_VI.alpha*FeS_VI.K_T
    Sfusion[i] = burnman.constants.gas_constant*np.log(2.) / (1. - aK_T*dTdP)
    Vfusion[i] = Sfusion[i]*dTdP

    Smelt[i] = FeS_VI.S + Sfusion[i]
    Vmelt[i] = FeS_VI.V + Vfusion[i]



pressures2 = np.linspace(1.e5, 120.e9, 101)
S_VI = np.empty_like(pressures2)
V_VI = np.empty_like(pressures2)
Smelt2 = np.empty_like(pressures2)
Vmelt2 = np.empty_like(pressures2)

for i, P in enumerate(pressures2):
    T = melting_temperature(P)
    FeS_VI.set_state(P, T)
    liq_FeS.set_state(P, T)
    Smelt2[i] = liq_FeS.S
    print liq_FeS.S
    Vmelt2[i] = liq_FeS.V 
    S_VI[i] = FeS_VI.S 
    V_VI[i] = FeS_VI.V 


plt.plot(pressures, Smelt, label='S using Tallon')
plt.plot(pressures2, Smelt2, label='model')
plt.plot(pressures2, S_VI, 'r--', label='S VI')
plt.legend(loc="lower right")
plt.show()

# Compare the equation of state with available experimental data
data = [[4.1e9, 1573.15, 4.723],# Chen et al., 2005
        [0.4e9, 1600., 4.23],
        [2.2e9, 1600., 4.58],
        [2.1e9, 1700., 4.55],
        [2.8e9, 1700., 4.62],
        [2.6e9, 1800., 4.59],
        [3.8e9, 1800., 4.71]] # Nishida et al., 2011



P_liq = np.array(zip(*data)[0])
T_liq = np.array(zip(*data)[1])
PT_liq = [P_liq, T_liq]
expt_rhos = np.array(zip(*data)[2])
V_liq = liq_FeS.params['molar_mass']*1.e-3/expt_rhos


plt.plot(P_liq, V_liq, marker='o', linestyle='None', label='experiment')
plt.plot(pressures, Vmelt, label='V using Tallon')
plt.plot(pressures2, Vmelt2, label='model')
plt.plot(pressures2, V_VI, 'r--', label='V VI')
plt.legend(loc="lower right")
plt.show()





f=open('data/FeS_melting_Williams_Jeanloz_1990.dat', 'r')
WJ1990_melt = []
WJ1990_solid = []
datastream = f.read()  # We need to open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] != "%":
        if line[3] == "melt":
            WJ1990_melt.append([float(line[0])*1.e9, 
                                float(line[1]), 
                                float(line[2])])
        if line[3] == "solid":
            WJ1990_solid.append([float(line[0])*1.e9, 
                                 float(line[1]), 
                                 float(line[2])])

WJ1990_melt = np.array(zip(*WJ1990_melt))
WJ1990_solid = np.array(zip(*WJ1990_solid))

# Plot Boehler data
data = listify_xy_file('data/FeS_melting_Boehler_1992.dat')
plt.errorbar(data[0], data[1], yerr=60., linestyle='None', marker='o')

plt.errorbar(WJ1990_melt[0]/1.e9, WJ1990_melt[1], yerr=WJ1990_melt[2], linestyle='None', marker='o')
plt.errorbar(WJ1990_solid[0]/1.e9, WJ1990_solid[1], yerr=WJ1990_solid[2], linestyle='None', marker='o')


data = listify_xy_file('data/Fe12S13_melting_Ryzhenko_Kennedy_1973.dat')
plt.plot(data[0], data[1], linestyle='None', marker='o')



Pm = np.linspace(1.e5, 120.e9, 101)
plt.plot(Pm/1.e9, melting_temperature(Pm))




pressures = np.linspace(36.e9, 120.e9, 101)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] =  fsolve(eqm_temperature([liq_FeS, FeS_VI], [1.0, -1.0]), 2000., args=(P))[0]

plt.plot(pressures/1.e9, temperatures)

plt.xlabel("P (GPa)")
plt.ylabel("T (K)")
plt.show()
