# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import numpy as np
from scipy.optimize import fsolve, curve_fit
import burnman
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
from burnman import constants
from scipy.integrate import cumtrapz, simps
atomic_masses=read_masses()


# This file adapts the equation of state of FeS IV from the model of Ohfuji,
# changing the value of K' so that the IV -> VI transition is of ~second order
# at 1200 K. At the same time, the thermal expansivity of FeS VI is estimated.
# Finally, a fit to the volume data at 300 K and 1200 K and the new 
# FeS I, II, III and IV_HP EoSes are used to estimate the 
# enthalpy and entropy of FeS VI.
 
# Note that this script requires some user iteration of the Kprime of IV and 
# thermal expansivity of VI.


FeS_I = burnman.minerals.HP_2011_ds62.lot()
#FeS_II = burnman.minerals.Fe_Si_O.FeS_II()
#FeS_III = burnman.minerals.Fe_Si_O.FeS_III()
#FeS_IV_HP = burnman.minerals.Fe_Si_O.FeS_IV_HP()
#FeS_VI = burnman.minerals.Fe_Si_O.FeS_VI()

##########################################
############# FeS I, II, III #############
##########################################

class FeS_I_new (burnman.Mineral):
    def __init__(self):
        formula='FeS'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FeS I',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': -4165.,
            'V_0': 1.2e-5, # 6.733e-6 ,
            'K_0': 161.9e9, # 166.e9 ,
            'Kprime_0': 5.15, # 5.32 ,
            'Debye_0': 594. ,
            'grueneisen_0': 1.65 ,
            'q_0': 0.0 ,
            'Cv_el': 3.0, # 2.7,
            'T_el': 6088., # 6500.
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        burnman.Mineral.__init__(self)

FeS_I2 = FeS_I_new()
# First, let's take a look at the heat capacity and entropy of FeS

temperatures = np.linspace(300., 1200., 21)
Cps = np.empty_like(temperatures)
Ss =  np.empty_like(temperatures)
Cps2 = np.empty_like(temperatures)
Ss2 =  np.empty_like(temperatures)
P = 1.e5
for i, T in enumerate(temperatures):
    FeS_I.set_state(P, T)
    Cps[i] = FeS_I.C_p
    Ss[i] = FeS_I.S
    FeS_I2.set_state(P, T)
    Cps2[i] = FeS_I2.C_p
    Ss2[i] = FeS_I2.S

PT = [temperatures*0.0 + 1.e5, temperatures]

#popt, pcov = burnman.tools.fit_PTp_data(FeS_I2, 'C_p', ['Debye_0', 'grueneisen_0', 'T_el'], PT, Cps)
#print popt
#exit()

# T Cp S H Cp S H
Gronvold_data = burnman.tools.array_from_file("data/FeS_CpSH_Gronvold_et_al_1959.dat")
JANAF_data = burnman.tools.array_from_file("data/FeS_JANAF.dat")

plt.plot(temperatures, Cps, label='HP')
plt.plot(temperatures, Cps2, label='new')
plt.plot(Gronvold_data[0], Gronvold_data[1]*4.184, marker='o', linestyle='None')
plt.plot(JANAF_data[0], JANAF_data[1], marker='o', linestyle='None')
plt.legend(loc="lower left")
plt.show()

plt.plot(temperatures, Ss, label='HP')
plt.plot(temperatures, Ss2, label='new')
plt.plot(Gronvold_data[0], Gronvold_data[2]*4.184, marker='o', linestyle='None')
plt.plot(JANAF_data[0], JANAF_data[2], marker='o', linestyle='None')
plt.legend(loc="lower left")
plt.show()

exit()



f=open('data/FeS_PV_King_Prewitt_1982_294K.dat', 'r')
data_I = []
data_II = []
data_III = []
datastream = f.read()  # We need to open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] == "I":
        data_I.append([float(line[1]), 
                     float(line[2])*burnman.constants.Avogadro/1.e30/12.])
    if line[0] == "II":
        data_II.append([float(line[1]), 
                     float(line[2])*burnman.constants.Avogadro/1.e30/8.])
    if line[0] == "III":
        data_III.append([float(line[1]), 
                     float(line[2])*burnman.constants.Avogadro/1.e30/8.])

f=open('data/FeS_VI_PTV_Ohfuji_et_al_2007.dat', 'r')
datastream = f.read()  # We need to open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] == "III" and line[2]=='300':
        data_III.append([float(line[1])*1.e9, 
                     float(line[3])/1.e6])

data_I = np.array(zip(*data_I))
data_II = np.array(zip(*data_II))
data_III = np.array(zip(*data_III))

PT = [data_II[0], 298.15*np.ones(len(data_II[0]))]
V = data_II[1]
print curve_fit(fit_EoS_data(FeS_II, ['V_0', 'K_0']), 
                PT, V, [1.648e-05, 75.e+9])
PT = [data_III[0], 298.15*np.ones(len(data_III[0]))]
V = data_III[1]
print curve_fit(fit_EoS_data(FeS_III, ['V_0', 'K_0']), 
                PT, V, [1.648e-05, 75.e+9])


plt.plot(data_I[0]/1.e9, data_I[1], linestyle='None', marker='o', label='FeS I')
plt.plot(data_II[0]/1.e9, data_II[1], linestyle='None', marker='o', label='FeS II')
plt.plot(data_III[0]/1.e9, data_III[1], linestyle='None', marker='o', label='FeS III')

pressures=np.linspace(1.e5, 36.e9, 101)
volumes_I = np.empty_like(pressures)
volumes_II = np.empty_like(pressures)
volumes_III = np.empty_like(pressures)
for i, P in enumerate(pressures):
    FeS_I.set_state(P, 298.15)
    volumes_I[i] = FeS_I.V

    FeS_II.set_state(P, 298.15)
    volumes_II[i] = FeS_II.V

    FeS_III.set_state(P, 298.15)
    volumes_III[i] = FeS_III.V

plt.plot(pressures/1.e9, volumes_I, label='FeS I (lot; HP_2011_ds62)')
plt.plot(pressures/1.e9, volumes_II, label='FeS II')
plt.plot(pressures/1.e9, volumes_III, label='FeS III')
plt.legend(loc='lower left')
plt.show()

# Now find standard enthalpies of II and III at 298 K
# Assumes that the entropy is already correct
# We can ignore this, as we're only interested in the 
# room temperature gibbs energies of II and III.
def find_enthalpy(phase):
    def enthalpy(H_0, G, P, T):
        phase.params['H_0'] = H_0[0]
        phase.set_state(P, T+1.) # reset set state
        phase.set_state(P, T)
        return [G - phase.gibbs]
    return enthalpy

P_I_II = 3.4e9 # Pa
P_II_III = 6.7e9 # Pa
P_III_VI = 36.e9 # Pa

FeS_I.set_state(P_I_II, 298.15)
print 'H_0 for FeS II:',  fsolve(find_enthalpy(FeS_II), [FeS_II.params['H_0']], args=(FeS_I.gibbs, P_I_II, 298.15))[0]

FeS_II.set_state(P_II_III, 298.15)
print 'H_0 for FeS III:', fsolve(find_enthalpy(FeS_III), [FeS_III.params['H_0']], args=(FeS_II.gibbs, P_II_III, 298.15))[0]

FeS_III.set_state(P_III_VI, 298.15)
FeS_VI_gibbs_points = [[P_III_VI, 298.15, FeS_III.gibbs]]

##########################################
################# FeS VI #################
##########################################

# First, let's get the room temperature PVT for FeS VI done
# There's not much data...

# There's a suggestion that the IV-VI reaction may be second order
# (Ohfuji et al., 2007), so it seems reasonable to pick P_ref 
# of the (metastable) IV-VI transition (~36 GPa)
P_IV_VI = 36.e9
FeS_VI.params['T_0'] = 300.
FeS_VI.params['P_0'] = P_IV_VI


f=open('data/FeS_VI_PTV_Ohfuji_et_al_2007.dat', 'r')
data = []
datastream = f.read()  # We need to open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] == "VI" and line[2]=='300':
        data.append([float(line[1])*1.e9, 
                     float(line[2]), 
                     float(line[3])/1.e6, 
                     float(line[4])/1.e6])

f=open('data/FeS_VI_PV_Ono_Kikegawa_2006.dat', 'r')
datastream = f.read()  # We need to open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] != "%":
        data.append([float(line[0])*1.e9, 
                     300., 
                     float(line[2])/4.*burnman.constants.Avogadro*1.e-30, 
                     float(line[3])/4.*burnman.constants.Avogadro*1.e-30])

P_VI, T_VI, V_VI, Verr_VI = zip(*data)

P_VI = np.array(P_VI)
T_VI = np.array(T_VI)
V_VI = np.array(V_VI)
Verr_VI = np.array(Verr_VI)

guesses = [FeS_VI.params['V_0'],
           FeS_VI.params['K_0'],
           FeS_VI.params['Kprime_0']]  # V, K, K', a
print curve_fit(fit_PV_data_full(FeS_VI), P_VI, V_VI, guesses, Verr_VI)


pressures = np.linspace(1.e5, 250.e9, 101)
volumes_VI = np.empty_like(pressures)

T = 300.
for i, P in enumerate(pressures):
    FeS_VI.set_state(P, T)
    volumes_VI[i] = FeS_VI.V

plt.plot(pressures/1.e9, volumes_VI, 'b-', label='VI (300 K)')
plt.plot(P_VI/1.e9, V_VI, marker='o', linestyle='None', label='Ohfuji et al. (2007)')
#plt.ylim(8.e-6, 14.e-6)
plt.legend(loc="upper right")
plt.show()


##########################################
################# FeS IV #################
##########################################

# Ohfuji et al. (2007) equation of state
def pressure(V, T, params):
    T0, K0, dKdP, dKdT, V0, a0, a1 = params
    
    K0T = K0 + dKdT*(T - T0)
    V0T = V0*np.exp(a0*(T-T0) + 0.5*a1*(T*T - T0*T0)) 

    PVT = 1.5*K0T*(np.power((V/V0T), -7./3.) - 
                   np.power((V/V0T), -5./3.)) * (1. - 0.75*(4. - dKdP) * (np.power((V/V0T), -2./3.) - 1.))

    return PVT
    
V_IV = 228.02/8.*constants.Avogadro*1.e-30
V_V = 59.92/2.*constants.Avogadro*1.e-30

IV_params = [600., 62.5e9, 4.0, -0.0208e9, V_IV, 7.16e-5, 6.08e-8]
V_params = [1000., 54.3e9, 4.0, -0.0117e9, V_V, 10.42e-5, 0.0]

def find_volume(V, P, T, params):
    return P - pressure(V[0], T, params)

def volume(P, T, params):
    return fsolve(find_volume, [1.e-7], args=(P, T, params))[0]

# Plot the Ohfuji et al. obtained volumes at 16.5 GPa
P = 16.5e9
temperatures = np.linspace(500., 2000., 101)
volumes_IV = np.empty_like(temperatures)
volumes_V = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    volumes_IV[i] = volume(P, T, IV_params)
    volumes_V[i] = volume(P, T, V_params)


plt.plot(temperatures, volumes_IV, label='IV')
plt.plot(temperatures, volumes_V, label='V')
plt.legend(loc='lower right')
plt.show()

# Compare volumes with experimental data from Urukawa et al. (2004)
V0 = 361.88 / 12 * constants.Avogadro * 1.e-30 # King and Prewitt, 1982
temperatures = [500., 1200., 2000.]
pressures = np.linspace(10.e9, 30.e9, 101)

for T in temperatures:
    for i, P in enumerate(pressures):
        volumes_IV[i] = volume(P, T, IV_params)
        volumes_V[i] = volume(P, T, V_params)
    plt.plot(pressures/1.e9, volumes_IV, label='IV')
    plt.plot(pressures/1.e9, volumes_V, label='V')

data = listify_xy_file('data/FeS_PV_500K_Urukawa.dat')
plt.plot(data[0], data[1]*V0, linestyle='None', marker='o')

data = listify_xy_file('data/FeS_PV_1200K_V_Urukawa.dat')
plt.plot(data[0], data[1]*V0, linestyle='None', marker='o')
plt.legend(loc='upper right')
plt.show()


# Take a bunch of pressures and temperatures, 
# and fit the model of Ohfuji et al to the Holland and Powell EoS.
temperatures = np.linspace(500., 1200., 11)
pressures = np.linspace(14.e9, 20.e9, 11)

PT = []
V = []
for P in pressures:
    for T in temperatures:
        PT.append([P, T])
        V.append(volume(P, T, IV_params))

PT = zip(*PT)

#########################################################
##### Here we fix the Kprime so that the transition #####
####### with FeS VI is of second order at 1200 K. #######
# Note that we also tweak the thermal expansivity of VI #
#########################################################

FeS_IV_HP.params['Kprime_0'] = 6.5
FeS_VI.params['a_0'] = 3.8e-05

print curve_fit(fit_EoS_data(FeS_IV_HP, ['V_0', 'K_0', 'a_0']), 
                PT, V, [1.648e-05, 75.e+9, 1.07e-4])


temperatures = [500., 1200., 2000.]
pressures = np.linspace(10.e9, 60.e9, 101)

volumes_IV = np.empty_like(pressures)
volumes_V = np.empty_like(pressures)
volumes = np.empty_like(pressures)
volumes_VI = np.empty_like(pressures)
for T in temperatures:
    for i, P in enumerate(pressures):
        volumes_IV[i] = volume(P, T, IV_params)
        volumes_V[i] = volume(P, T, V_params)
        FeS_IV_HP.set_state(P, T)
        FeS_VI.set_state(P, T)

        volumes[i] = FeS_IV_HP.V
        volumes_VI[i] = FeS_VI.V

    plt.plot(pressures/1.e9, volumes_IV, label='IV')
    plt.plot(pressures/1.e9, volumes_V, label='V')
    plt.plot(pressures/1.e9, volumes, 'r--', label='new')
    plt.plot(pressures/1.e9, volumes_VI, 'b--', label='VI')

data = listify_xy_file('data/FeS_PV_500K_Urukawa.dat')
plt.plot(data[0], data[1]*V0, linestyle='None', marker='o')

data = listify_xy_file('data/FeS_PV_1200K_IV_Urukawa.dat')
plt.plot(data[0], data[1]*V0, linestyle='None', marker='o')

data = listify_xy_file('data/FeS_PV_1200K_V_Urukawa.dat')
plt.plot(data[0], data[1]*V0, linestyle='None', marker='o')


f=open('data/FeS_VI_PTV_Ohfuji_et_al_2007.dat', 'r')
data = []
datastream = f.read()  # We need to open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] == "VI" and line[2]=='300':
        data.append([float(line[1]), 
                     float(line[3])/1.e6])
data = np.array(zip(*data))
plt.plot(data[0], data[1], linestyle='None', marker='o')
plt.xlim(0., 60.)
plt.legend(loc='upper right')
plt.show()

# Now calculate the Gibbs free energy at the FeS IV-VI transition at 1200 K
P_IV_VI_1200 = 36.e9

tro = burnman.minerals.Fe_Si_O.FeS_IV_V_Evans()
tro.set_state(1.e5, 1200.)
PVs = [[1.e5, tro.V]]

V0 = 361.88 / 12 * burnman.constants.Avogadro * 1.e-30 # King and Prewitt, 1982
data = listify_xy_file('data/FeS_PV_1200K_V_Urukawa.dat')
for datum in zip(*data):
    P = datum[0]*1.e9
    V = datum[1]*V0
    PVs.append([P, V])

data = listify_xy_file('data/FeS_PV_1200K_IV_Urukawa.dat')
for datum in zip(*data):
    P = datum[0]*1.e9
    V = datum[1]*V0
    PVs.append([P, V])


pressures = np.linspace(24.e9, P_IV_VI_1200, 101)
for P in pressures:
    FeS_IV_HP.set_state(P, 1200.)
    PVs.append([P, FeS_IV_HP.V])


PVs = np.array(zip(*PVs))
plt.plot(PVs[0], PVs[1])
plt.show()

VdP = cumtrapz(PVs[1], PVs[0])[-1]
G_IV_VI_1200 = tro.gibbs + VdP
FeS_VI_gibbs_points.append([P_IV_VI_1200, 1200., G_IV_VI_1200])


def find_enthalpy_entropy(phase):
    def enthalpy_entropy(HS, points):
        phase.params['H_0'] = HS[0]
        phase.params['S_0'] = HS[1]

        Gdiff = []
        for point in points:
            P, T, G = point
            phase.set_state(P, T+1.) # reset set state
            phase.set_state(P, T)
            Gdiff.append(G - phase.gibbs)
        return Gdiff
    return enthalpy_entropy

print '[H_0, S_0] for FeS VI:', fsolve(find_enthalpy_entropy(FeS_VI), 
                                       [FeS_VI.params['H_0'], FeS_VI.params['S_0']], 
                                       args=(FeS_VI_gibbs_points))


