import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
from listify_xy_file import *
atomic_masses=read_masses()

'''
from scipy.constants import physical_constants
r_B = physical_constants['Bohr radius'][0]
print r_B
print 47.e-30 / np.power(r_B, 3.) / 4.
exit()
'''
print 'NB: also possibility of an anharmonic contribution'

class fcc_iron (burnman.Mineral):
    def __init__(self):
        formula='Fe'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FCC iron',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': 0.,
            'V_0': 46.35/1.e30*burnman.constants.Avogadro/4. ,
            'K_0': 145.0e9 ,
            'Kprime_0': 5.3 ,
            'Debye_0': 417. ,
            'grueneisen_0': 1.8 , # 2. ok
            'q_0': 0.5 , # 0., ok
            'Cv_el': 2.7,
            'T_el': 10000., # 10000. ok
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        burnman.Mineral.__init__(self)


fcc = fcc_iron()


temperatures = np.linspace(1185., 1665., 49)
Cps = np.empty_like(temperatures)
Ss = np.empty_like(temperatures)
P = 1.e5
for i, T in enumerate(temperatures):
    fcc.set_state(P, T)
    Cps[i] = fcc.C_p
    Ss[i] = fcc.S
    print T, fcc.C_p, fcc.S, fcc.gibbs

plt.plot(temperatures, Cps)
fcc_Cp_data = listify_xy_file('data/fcc_Cp_Chen_Sundman_2001.dat')
plt.plot(fcc_Cp_data[0], fcc_Cp_data[1], marker='o', linestyle='None')
fcc_Cp_data = listify_xy_file('data/fcc_Cp_Rogez_le_Coze_1980.dat')
plt.plot(fcc_Cp_data[0], fcc_Cp_data[1], marker='o', linestyle='None')
plt.show()

P = 1.e5
temperatures = np.linspace(1., 1700., 101)
volumes = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    fcc.set_state(P, T)
    volumes[i] = fcc.V
plt.plot(temperatures, volumes)

Z_fcc = 4.
T_Onink_et_al_1993 = np.linspace(1180., 1250., 101)
def V(T):
    return np.power(0.36320*(1+24.7e-6*(T - 1000.)),3.)*1e-27*burnman.constants.Avogadro/Z_fcc

plt.plot(T_Onink_et_al_1993, V(T_Onink_et_al_1993))


fcc_V_data = listify_xy_file('data/Basinski_et_al_1955_fcc_volumes_RP.dat')
plt.plot(fcc_V_data[0], fcc_V_data[1]/11.7024*7.17433e-6, marker='o', linestyle='None')
plt.plot(fcc_V_data[0], 0.9905*fcc_V_data[1]/11.7024*7.17433e-6, marker='o', linestyle='None')

plt.show()




# High temperature data from Nishihara et al., 2012
fcc_V_data = listify_xy_file('data/Nishihara_et_al_2012_fcc_volumes.dat')
P_N, T_N, V_N, Verr_N = fcc_V_data
P_N = P_N*1.e9
V_N = V_N/1.e30*burnman.constants.Avogadro/4.
Verr_N = Verr_N/1.e30*burnman.constants.Avogadro/4.

for T in [1223., 1273., 1323.]:
    pressures = np.linspace(1.e5, 30.e9, 101)
    for i, P in enumerate(pressures):
        fcc.set_state(P, T)
        volumes[i] = fcc.V


    plt.plot(pressures/1.e9, volumes)
plt.plot(P_N/1.e9, V_N, marker='o', linestyle='None')
plt.show()
