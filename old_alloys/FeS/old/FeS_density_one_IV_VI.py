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

liq_FeS = burnman.minerals.Fe_Si_O.FeS_liquid()
FeS_II = burnman.minerals.Fe_Si_O.FeS_II()
FeS_dummy = burnman.minerals.Fe_Si_O.FeS_dummy()
FeS_IV_HP = burnman.minerals.Fe_Si_O.FeS_IV_HP()
FeS_VI = burnman.minerals.Fe_Si_O.FeS_VI()
tro = burnman.minerals.Fe_Si_O.FeS_IV_V_1bar()
trot = burnman.minerals.HP_2011_ds62.trot()
lot = burnman.minerals.HP_2011_ds62.lot()


##########################################
############# FeS I, II, III #############
##########################################

f=open('data/FeS_PV_King_Prewitt_1982_294K.dat', 'r')
data = []
datastream = f.read()  # We need to open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] == "I":
        data.append([float(line[1]), 
                     float(line[2])*burnman.constants.Avogadro/1.e30/12.])
    if line[0] == "II":
        data.append([float(line[1]), 
                     float(line[2])*burnman.constants.Avogadro/1.e30/8.])
    if line[0] == "III":
        data.append([float(line[1]), 
                     float(line[2])*burnman.constants.Avogadro/1.e30/8.])

data = np.array(zip(*data))

pressures=np.linspace(1.e5, 10.e9, 101)
volumes = np.empty_like(pressures)
volumes_HP = np.empty_like(pressures)
for i, P in enumerate(pressures):
    FeS_II.set_state(P, 294.)
    lot.set_state(P, 294.)
    volumes[i] = FeS_II.V
    volumes_HP[i] = lot.V

plt.plot(data[0]/1.e9, data[1], linestyle='None', marker='o', label='King and Prewitt, 1982')
plt.plot(pressures/1.e9, volumes, label='FeS II')
plt.plot(pressures/1.e9, volumes_HP, label='lot (HP_2011_ds62)')
plt.legend(loc='lower left')
plt.show()


##########################################
############## FeS IV/VI HP ##############
##########################################

# There's a suggestion that the IV - VI reaction may 
# be second order (Ohfuji et al., 2007), so it seems
# reasonable to start with the assumption that
# FeS IV/VI can be modelled as a single phase at 
# high pressure

f=open('data/FeS_IV_PTV.dat', 'r')
data = []
datastream = f.read()  # We need to open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] != "%":
        data.append([float(line[0])*1.e9, 
                     float(line[1]), 
                     float(line[2]),
                     float(line[2])/1000.])


f=open('data/FeS_VI_PTV_Ohfuji_et_al_2007.dat', 'r')
datastream = f.read()  # We need to open the file
data_VI = []
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] == "VI" and line[2]=='300':
        data.append([float(line[1])*1.e9, 
                     float(line[2]), 
                     float(line[3])/1.e6, 
                     float(line[4])/1.e6])
        data_VI.append([float(line[1])*1.e9, 
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
        data_VI.append([float(line[0])*1.e9, 
                        300., 
                        float(line[2])/4.*burnman.constants.Avogadro*1.e-30, 
                        float(line[3])/4.*burnman.constants.Avogadro*1.e-30])



P_IV, T_IV, V_IV, Verr_IV = zip(*data)

P_IV = np.array(P_IV)
T_IV = np.array(T_IV)
V_IV = np.array(V_IV)
Verr_IV = np.array(Verr_IV)

PT_IV = [P_IV, T_IV]

guesses = [FeS_IV_HP.params['V_0'],
           FeS_IV_HP.params['K_0'],
           FeS_IV_HP.params['Kprime_0'],
           FeS_IV_HP.params['a_0']]  
print curve_fit(fit_PVT_data_full(FeS_IV_HP), PT_IV, V_IV, guesses, Verr_IV)





# Now find the Gibbs free energies at the crossover points 
# between the low pressure and high pressure phases

# The boundaries are pretty sharp, so we can take the crossover pressure 
# as the intersection of the two volume curves

def find_gibbs(T_0, P_crossover, K_0, Kprime_0):
    P_0 = 1.e5
    tro.set_state(P_0, T_0)
    H_0 = tro.H
    S_0 = tro.S
    V_0 = tro.V

    FeS_dummy.params['P_0'] = P_0
    FeS_dummy.params['T_0'] = T_0
    FeS_dummy.params['H_0'] = tro.H
    FeS_dummy.params['S_0'] = tro.S
    FeS_dummy.params['V_0'] = tro.V
    FeS_dummy.params['K_0'] = K_0
    FeS_dummy.params['Kprime_0'] = Kprime_0
    FeS_dummy.params['Kdprime_0'] = -FeS_dummy.params['Kprime_0']/FeS_dummy.params['K_0']

    FeS_dummy.set_state(P_crossover, T_0)
    return FeS_dummy.gibbs


data = listify_xy_file('data/FeS_PV_800K_Fei_et_al_1995.dat')
plt.plot(data[0], 87.92e-6/data[1], linestyle='None', marker='o', label='Fei et al. (1995)')

for datum in zip(*data):
    print datum[0], '800.', 87.92e-6/datum[1], 'Fei et al. (1995)'

T = 800.
pressures=np.linspace(1.e5, 25.e9, 101)
volumes_IV_LP = np.empty_like(pressures)
volumes_V_LP = np.empty_like(pressures)
volumes_HP = np.empty_like(pressures)
for i, P in enumerate(pressures):
    FeS_IV_HP.set_state(P, T)
    volumes_HP[i] = FeS_IV_HP.V

plt.plot(pressures/1.e9, volumes_HP, label='FeS IV (HP)')
plt.title(str(T)+'K')
plt.legend(loc='lower right')
plt.show()

gibbs_array = []
T = 500.
P_crossover = 10.2e9
gibbs=find_gibbs(T, P_crossover, 45.e9, 0.5)
gibbs_array.append([P_crossover, T, gibbs])

pressures = np.linspace(1.e5, 25.e9, 101)
volumes_LP = np.empty_like(pressures)
volumes_HP = np.empty_like(pressures)
for i, P in enumerate(pressures):
   FeS_dummy.set_state(P, T)
   volumes_LP[i] = FeS_dummy.V
   FeS_IV_HP.set_state(P, T)
   volumes_HP[i] = FeS_IV_HP.V

V0 = 361.88 / 12 * burnman.constants.Avogadro * 1.e-30 # King and Prewitt, 1982
data = listify_xy_file('data/FeS_PV_500K_Urukawa.dat')
plt.plot(data[0], data[1]*V0, linestyle='None', marker='o')

for datum in zip(*data):
    print datum[0], '500.', datum[1]*V0, 'Urukawa et al. (2004)'

data = listify_xy_file('data/FeS_PV_473K_Kusaba.dat')
plt.plot(data[0], data[1]*V0, linestyle='None', marker='o')


for datum in zip(*data):
    print datum[0], '473.', datum[1]*V0, 'Kusaba et al. (1998)'

plt.plot(pressures/1.e9, volumes_LP, label='IV (LP)')
plt.plot(pressures/1.e9, volumes_HP, label='IV (HP)')
plt.title(str(T)+'K')
plt.legend(loc='lower right')
plt.show()

T = 1200.
P_crossover = 14.e9
gibbs = find_gibbs(T, P_crossover, 52.e9, 0.5)
gibbs_array.append([P_crossover, T, gibbs])


pressures = np.linspace(1.e5, 25.e9, 101)
volumes_LP = np.empty_like(pressures)
volumes_HP = np.empty_like(pressures)
for i, P in enumerate(pressures):
   FeS_dummy.set_state(P, T)
   volumes_LP[i] = FeS_dummy.V
   FeS_IV_HP.set_state(P, T)
   volumes_HP[i] = FeS_IV_HP.V

data = listify_xy_file('data/FeS_PV_1200K_V_Urukawa.dat')
plt.plot(data[0], data[1]*V0, linestyle='None', marker='o')
data = listify_xy_file('data/FeS_PV_1200K_IV_Urukawa.dat')
plt.plot(data[0], data[1]*V0, linestyle='None', marker='o')

for datum in zip(*data):
    print datum[0], '1200.', datum[1]*V0, 'Urukawa et al. (2004)'

plt.plot(pressures/1.e9, volumes_LP, label='V (LP)')
plt.plot(pressures/1.e9, volumes_HP, label='IV (HP)')
plt.title(str(T)+'K')
plt.legend(loc='lower right')
plt.show()


pressures = np.linspace(1.e5, 250.e9, 101)
volumes_IV_HP = np.empty_like(pressures)
volumes_IV_HP2 = np.empty_like(pressures)

T = 300.
T2 = 1200.
for i, P in enumerate(pressures):
    FeS_IV_HP.set_state(P, T)
    volumes_IV_HP[i] = FeS_IV_HP.V

    FeS_IV_HP.set_state(P, T2)
    volumes_IV_HP2[i] = FeS_IV_HP.V


plt.plot(pressures/1.e9, volumes_IV_HP, 'r-', label='IV (300 K)')
plt.plot(pressures/1.e9, volumes_IV_HP2, 'r--', label='IV (1200 K)')

data = listify_xy_file('data/FeS_PV_1200K_V_Urukawa.dat')
plt.plot(data[0], data[1]*V0, linestyle='None', marker='o')
data = listify_xy_file('data/FeS_PV_1200K_IV_Urukawa.dat')
plt.plot(data[0], data[1]*V0, linestyle='None', marker='o')

plt.plot(np.array(zip(*data_VI)[0])/1.e9, 
         zip(*data_VI)[2], 
         marker='o', linestyle='None', label='Ohfuji et al. (2007)')
#plt.ylim(8.e-6, 14.e-6)
plt.legend(loc="upper right")
plt.show()




# Now we can estimate H0 and S0 of our high pressure FeS IV phase
# using the two Gibbs free energies obtained above
def HS(args, points):
    H_0, S_0 = args

    FeS_IV_HP.params['H_0'] = H_0
    FeS_IV_HP.params['S_0'] = S_0

    diffs = []   
    for point in points:
        P, T, gibbs = point
        FeS_IV_HP.set_state(P, T)
        diffs.append(gibbs - FeS_IV_HP.gibbs)
    return diffs

print fsolve(HS, [-100000., 70.], args=(gibbs_array))

##########################################
############### FeS liquid ###############
##########################################


data = [[1478.7442021, 3.90925925926],
        [1483.9631360, 3.94074074074],
        [1484.5584977, 3.86666666667],
        [1523.2466251, 3.87407407407],
        [1536.1360332, 3.89074074074],
        [1536.5289027, 3.85],
        [1573.7251644, 3.85],
        [1578.2120110, 3.84814814815],
        [1577.4721356, 3.83148148148],
        [1622.7101073, 3.82222222222],
        [1626.0737279, 3.82407407407]]

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


def eqm(T, P, min1, min2):
    min1.set_state(P, T[0])
    min2.set_state(P, T[0])
    return [min1.gibbs - min2.gibbs]


FeS_IV_V.set_state(1.e5, 1463.)
liq_FeS.set_state(1.e5, 1463.)
print FeS_IV_V.K_T/1.e9, FeS_IV_V.V, FeS_IV_V.alpha
print liq_FeS.K_T/1.e9, liq_FeS.V, liq_FeS.alpha

T = fsolve(eqm, [1400.], args=(1.e5, FeS_IV_V, liq_FeS))[0]
print P/1.e9, T, liq_FeS.V - FeS_IV_V.V, liq_FeS.S - FeS_IV_V.S


HP_convert(liq_FeS, 300., 1400., 1463., 50.e9)
#HP_convert(tro, 1000., 1500., 1463., 50.e9)


data = listify_xy_file('data/FeS_melting_Boehler_1992.dat')


'''
def fit_modulus_to_Tm(pressures, Kprime, Kdprime):
    liq_FeS.params['Kprime_0'] = Kprime
    liq_FeS.params['Kdprime_0'] = Kdprime
    temperatures=[]
    for pressure in pressures:
        temperatures.append(fsolve(eqm, [1400.], args=(P, FeS_IV_V, liq_FeS))[0])
    return temperatures

guesses = [liq_FeS.params['Kprime_0'], 
           liq_FeS.params['Kdprime_0']]
print curve_fit(fit_modulus_to_Tm, data[0], data[1], guesses)
'''
pressures = np.linspace(1.e5, 50.e9, 50)
temperatures = np.empty_like(pressures)
volumes = np.empty_like(pressures)
volumes_2 = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = fsolve(eqm, [1400.], args=(P, FeS_IV_V, liq_FeS))[0]
    liq_FeS.set_state(P, 1600.)
    volumes[i] = liq_FeS.params['molar_mass']*1.e-3/liq_FeS.V
    liq_FeS.set_state(P, 2600.)
    volumes_2[i] = liq_FeS.params['molar_mass']*1.e-3/liq_FeS.V

plt.errorbar(data[0], data[1], yerr=60., linestyle='None', marker='o')
plt.plot(pressures/1.e9, temperatures)
plt.xlabel("P (GPa)")
plt.ylabel("T (K)")
plt.show()

plt.plot(pressures/1.e9, volumes)
plt.plot(pressures/1.e9, volumes_2)
plt.xlabel("P (GPa)")
plt.ylabel("V (m^3)")
plt.show()

