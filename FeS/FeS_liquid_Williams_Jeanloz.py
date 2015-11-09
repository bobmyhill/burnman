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


tro = burnman.minerals.Fe_Si_O.FeS_IV_V_Evans()
FeS_IV_HP = burnman.minerals.Fe_Si_O.FeS_IV_HP()
FeS_VI = burnman.minerals.Fe_Si_O.FeS_VI()
liq_FeS = burnman.minerals.Fe_Si_O.FeS_liquid()


Tfusion=1463.
Hfusion=31464.
tro.set_state(1.e5, Tfusion)
liq_FeS.set_state(1.e5, Tfusion)
print 'liquid enthalpy:', tro.H + Hfusion
print 'liquid entropy:', -(tro.gibbs - (tro.H + Hfusion))/Tfusion
print liq_FeS.gibbs - tro.gibbs

print 'dTdP:', (liq_FeS.V - tro.V)/(liq_FeS.S - tro.S)*1.e9, 'K/GPa'


data = listify_xy_file('data/Chen_et_al_2005_FeS_V_4.1GPa.dat')
plt.plot(data[0], data[1], linestyle='None', marker='o', label='Chen (XRD)')
data = listify_xy_file('data/Chen_et_al_2005_FeS_V_4.1GPa_radiograph.dat')
plt.plot(data[0], data[1], linestyle='None', marker='o', label='Chen (radiograph)')

P = 4.1e9
temperatures = np.linspace(1073., 1573., 101)
volumes = np.empty_like(temperatures)
melt_volumes = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    tro.set_state(P, T)
    liq_FeS.set_state(P, T)
    melt_volumes[i] = liq_FeS.V


V0 = 361.88 / 12 * burnman.constants.Avogadro * 1.e-30 # King and Prewitt, 1982
data = [[800., 87.92e-6/4.9583],
        [1200., 0.9942*V0]]
data = zip(*data)

plt.plot(data[0], data[1], linestyle='None', marker='o', label='data from Fei et al. and Urakawa et al.')
plt.plot(temperatures, melt_volumes, label='melt model')
plt.legend(loc='lower right')
plt.show()



##########################################
############### FeS liquid ###############
##########################################

# Volume of FeS at  the melting point at 1 bar
dTdP = 65./1.e9 # K/Pa # Rhysenko and Kennedy Fe12S13 
Smelting = 21.506 # (Barin)
P = 1.e5
T = 1463.
tro.set_state(P, T) # from Evans
liq_FeS.set_state(P, T)
print tro.gibbs
Vsolid = tro.V
Vmelt = Smelting*dTdP + tro.V
print 'FAVOURED VMELT'
print 'Vmelt:', Vmelt, 'using Rhysenko and Kennedy melting curve'

print 'ALTERNATIVELY'
Vmelt = 2.244e-05 # Tm (Kaiura and Toguri, 1979)
dTdP = (Vmelt - tro.V)/Smelting
print 'dT/dP:', dTdP*1.e9, 'K/GPa from Kaiura and Toguri data'

# First, let's plot the densities of FeS liquid at 1 bar
P = 1.e5
temperatures = np.linspace(1463., 1623.15)
densities = np.empty_like(temperatures)

for i, T in enumerate(temperatures):
    liq_FeS.set_state(P, T)
    densities[i] = liq_FeS.params['molar_mass']*1.e3/(liq_FeS.V*1.e6)

data = listify_xy_file('data/density_FeS_Kaiura_Toguri_1979.dat')
plt.plot(data[0], data[1], linestyle='None', marker='o')

plt.plot(temperatures, densities)
plt.show()

# Here's the melting temperature at 1 bar
T = fsolve(eqm_temperature([tro, liq_FeS], [1., -1]), [1400.], args=(1.e5))[0]
print 'Melting temperature at 1 bar:', T, 'K'


# And here's the slope of the melting curve
print ((tro.V - liq_FeS.V) / (tro.S - liq_FeS.S)) *1.e9, 'K/GPa'
print 'Experimental (lower bound):', (1600.-1463.)/2.9, 'K/GPa'
print 'Experimental (upper bound):', (1790.-1463.)/2.5, 'K/GPa'

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


# Compare the equation of state with available experimental data
data = [[4.1e9, 1573.15, 4.723],# Chen et al., 2005
        [0.4e9, 1600., 4.23],
        [2.2e9, 1600., 4.58],
        [2.1e9, 1700., 4.55],
        [2.8e9, 1700., 4.62],
        [2.6e9, 1800., 4.59],
        [3.8e9, 1800., 4.71]] # Nishida et al., 2011

# Final datapoint from rollover of FeS VI curve
P_n = 120.e9
T_n = 3550.
FeS_VI.set_state(P_n, T_n)
data.append([P_n, T_n, liq_FeS.params['molar_mass']*1.e-3/FeS_VI.V])


P_liq = np.array(zip(*data)[0])
T_liq = np.array(zip(*data)[1])
PT_liq = [P_liq, T_liq]
expt_rhos = np.array(zip(*data)[2])
V_liq = liq_FeS.params['molar_mass']*1.e-3/expt_rhos

liq_FeS.params['Kprime_0'] = 8.0
guesses = [5.e-6, 
           10.e9,
           7.,
           4.e-6]  # V, K
#print curve_fit(fit_EoS_data(liq_FeS, ['V_0', 'K_0', 'Kprime_0', 'a_0']), PT_liq, V_liq, guesses)

volumes = np.empty_like(P_liq)
for i, datum in enumerate(data):
    P, T, rho = datum
    liq_FeS.set_state(P, T)
    volumes[i] = liq_FeS.V
    print P/1.e9, volumes[i], liq_FeS.params['molar_mass']*1.e-3/rho

plt.plot(P_liq, volumes, linestyle='None', marker='o', label='model')
plt.plot(P_liq, V_liq, linestyle='None', marker='o', label='experiment')


# Now plot a few points of solids and liquids near the melting curve
melting_curve = [[1.e5,   1463.],
                 [10.e9,  1900.],
                 [20.e9,  2335.],
                 [30.e9,  2630.],
                 [40.e9,  2850.],
                 [50.e9,  3030.],
                 [60.e9,  3170.],
                 [80.e9,  3400.],
                 [100.e9, 3550.],
                 [120.e9, 3650.]]
            
from scipy.interpolate import interp1d
melting_temperature = interp1d(*zip(*melting_curve), kind='cubic')
     
line_segments = [[1.e5, 120.e9, liq_FeS],
                 [20.e9, 36.e9, FeS_IV_HP],
                 [36.e9, 120.e9, FeS_VI]]

for segment in line_segments:
    P0, P1, phase = segment
    pressures = np.linspace(P0,P1,101)
    volumes = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        T = melting_temperature(P)
        phase.set_state(P, T)
        volumes[i] = phase.V
    plt.plot(pressures, volumes, label=phase.params['name'])

# Also compare with the prediction of Chen et al., 2005 (Sm = 6.79 J/K/mol)
pressures = np.linspace(36.e9, 100.e9, 101)
Smelting = 6.79
melt_volumes = np.empty_like(pressures)
for i, P in enumerate(pressures):
    dP = 1.e9
    T = melting_temperature(P)
    T0 = melting_temperature(P-dP)
    T1 = melting_temperature(P+dP)
    dT = T1 - T0
    Vmelting = Smelting*dT / (2.*dP)
    print P/1.e9, T, Vmelting
    FeS_VI.set_state(P, T)
    melt_volumes[i] = FeS_VI.V + Vmelting

plt.plot(pressures, melt_volumes, label='Chen prediction for liquid')
plt.legend(loc='upper right')
plt.show()

# Now find H_0 that satisfies the estimated melting point at 36 GPa
# Convert liquid so that we can extend our study to higher pressure
P = 36.e9
T = melting_temperature(P)
print P, T
HP_convert(liq_FeS, 300., 1400., 1463., P)

tro.set_state(P, T)
liq_FeS.set_state(P, T)
print  tro.gibbs - liq_FeS.gibbs, 'J/mol'
Smelt = liq_FeS.S - tro.S
Vmelt = liq_FeS.V - tro.V
print  'Entropy of melting:', Smelt, 'J/K/mol'
print  'Volume of melting:', Vmelt, 'J/K/mol'
print 'dT/dP:', Vmelt/Smelt*1.e9, 'K/GPa'
# Find the melting curve with VI
pressures = np.linspace(36.e9, 120.e9, 101)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = fsolve(eqm_temperature([FeS_VI, liq_FeS], [1.0, -1.0]), 
                             [1400.], args=(P))[0]

# Find the melting curve with IV_HP
pressures2 = np.linspace(20.e9, 36.e9, 101)
temperatures2 = np.empty_like(pressures2)
for i, P in enumerate(pressures2):
    temperatures2[i] = fsolve(eqm_temperature([FeS_VI, liq_FeS], [1.0, -1.0]), 
                              [1400.], args=(P))[0]

# Find the melting curve with tro
pressures3 = np.linspace(1.e5, 5.e9, 101)
temperatures3 = np.empty_like(pressures3)
for i, P in enumerate(pressures3):
    temperatures3[i] = fsolve(eqm_temperature([tro, liq_FeS], [1.0, -1.0]), 
                             [1400.], args=(P))[0]

# Plot Boehler data
data = listify_xy_file('data/FeS_melting_Boehler_1992.dat')
plt.errorbar(data[0], data[1], yerr=60., linestyle='None', marker='o')

plt.errorbar(WJ1990_melt[0]/1.e9, WJ1990_melt[1], yerr=WJ1990_melt[2], linestyle='None', marker='o')
plt.errorbar(WJ1990_solid[0]/1.e9, WJ1990_solid[1], yerr=WJ1990_solid[2], linestyle='None', marker='o')


data = listify_xy_file('data/Fe12S13_melting_Ryzhenko_Kennedy_1973.dat')
plt.plot(data[0], data[1], linestyle='None', marker='o')


plt.plot(pressures3/1.e9, temperatures3)
plt.plot(pressures2/1.e9, temperatures2)
plt.plot(pressures/1.e9, temperatures)
plt.xlabel("P (GPa)")
plt.ylabel("T (K)")
plt.show()


print liq_FeS.params
