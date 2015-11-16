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
            'V_0': 46./1.e30*burnman.constants.Avogadro/4. ,
            'K_0': 133.0e9 ,
            'Kprime_0': 5.0 ,
            'Debye_0': 417. ,
            'grueneisen_0': 1.72 ,
            'q_0': 1.6 ,
            'Cv_el': 2.7,
            'T_el': 9000.,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        burnman.Mineral.__init__(self)

class hcp_iron (burnman.Mineral):
    def __init__(self):
        formula='Fe'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'HCP iron',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': 0.,
            'V_0': 6.733e-6 ,
            'K_0': 166.e9 ,
            'Kprime_0': 5.32 ,
            'Debye_0': 422. ,
            'grueneisen_0': 1.72 ,
            'q_0': 1 ,
            'Cv_el': 2.7,
            'T_el': 10000,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        burnman.Mineral.__init__(self)

fcc = fcc_iron()
hcp = hcp_iron()


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

plt.show()



P = 1.e5
temperatures = np.linspace(1., 2000., 101)
gibbs0 = np.empty_like(temperatures)
gibbs1 = np.empty_like(temperatures)
volumes0 = np.empty_like(temperatures)
volumes1 = np.empty_like(temperatures)
Vcheck0 = np.empty_like(temperatures)
Vcheck1 = np.empty_like(temperatures)
Cps0 = np.empty_like(temperatures)
Cps1 = np.empty_like(temperatures)
Cpcheck0 = np.empty_like(temperatures)
Cpcheck1 = np.empty_like(temperatures)
Ss0 = np.empty_like(temperatures)
Ss1 = np.empty_like(temperatures)
gr0 = np.empty_like(temperatures)
gr1 = np.empty_like(temperatures)
Scheck0 = np.empty_like(temperatures)
Scheck1 = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    hcp.set_state(P, T)
    volumes1[i] = hcp.V
    Cps1[i] = hcp.C_p
    Ss1[i] = hcp.S
    gibbs1[i] = hcp.gibbs
    gr1[i] = hcp.gr
    
    dT = 1.
    hcp.set_state(P, T-0.5*dT)
    G0 = hcp.gibbs
    S0 = hcp.S
    hcp.set_state(P, T+0.5*dT)
    G1 = hcp.gibbs
    S1 = hcp.S
    Scheck1[i] = (G0 - G1)/dT
    Cpcheck1[i] = T*(S1 - S0)/dT

    dP = 10.
    hcp.set_state(P-0.5*dP, T)
    G0 = hcp.gibbs
    hcp.set_state(P+0.5*dP, T)
    G1 = hcp.gibbs
    Vcheck1[i] = (G1-G0)/dP

hcp.params['Cv_el']= 0.0
hcp.params['T_el']= 1000.
for i, T in enumerate(temperatures):
    hcp.set_state(P, T)
    volumes0[i] = hcp.V
    Cps0[i] = hcp.C_p
    Ss0[i] = hcp.S
    gibbs0[i] = hcp.gibbs
    gr0[i] = hcp.gr

    dT = 1.
    hcp.set_state(P, T-0.5*dT)
    G0 = hcp.gibbs
    S0 = hcp.S
    hcp.set_state(P, T+0.5*dT)
    G1 = hcp.gibbs
    S1 = hcp.S
    Scheck0[i] = (G0 - G1)/dT
    Cpcheck0[i] = T*(S1 - S0)/dT

    dP = 100.
    hcp.set_state(P-0.5*dP, T)
    G0 = hcp.gibbs
    hcp.set_state(P+0.5*dP, T)
    G1 = hcp.gibbs
    Vcheck0[i] = (G1-G0)/dP

plt.plot(temperatures, gibbs1-gibbs0, label='Electronic contribution')
plt.legend(loc='lower right')
plt.title("Gibbs")
plt.show()


plt.plot(temperatures, gr0, label='No Cv_el')
plt.plot(temperatures, gr1, label='Cv_el')
plt.legend(loc='lower right')
plt.title("Grueneisen")
plt.show()

plt.plot(temperatures, volumes0, label='No Cv_el')
plt.plot(temperatures, volumes1, label='Cv_el')
plt.plot(temperatures, Vcheck0, 'r--', label='No Cv_el check')
plt.plot(temperatures, Vcheck1, 'b--', label='Cv_el check')
plt.legend(loc='lower right')
plt.title("Volume")
plt.show()


plt.plot(temperatures, Ss0, label='No Cv_el')
plt.plot(temperatures, Ss1, label='Cv_el')
plt.plot(temperatures, Scheck0, 'r--', label='No Cv_el check')
plt.plot(temperatures, Scheck1, 'b--', label='Cv_el check')
plt.legend(loc='lower right')
plt.title("Entropy")
plt.show()


plt.plot(temperatures, Cps0, label='No Cv_el')
plt.plot(temperatures, Cps1, label='Cv_el')
plt.plot(temperatures, Cpcheck0, 'r--', label='No Cv_el check')
plt.plot(temperatures, Cpcheck1, 'b--', label='Cv_el check')
plt.legend(loc='lower right')
plt.ylim(0., 50.)
plt.title("Heat capacity")
plt.show()
