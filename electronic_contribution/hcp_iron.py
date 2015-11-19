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
            'T_el': 6500.,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        burnman.Mineral.__init__(self)

hcp = hcp_iron()


pressures = np.linspace(50.e9, 170.e9, 51)
arr_Cv = []
arr_alpha = []
arr_aKT = []
arr_gamma = []
for T in [2000., 4000., 6000.]:
    Cvs = np.empty_like(pressures)
    alphas = np.empty_like(pressures)
    aKTs = np.empty_like(pressures)
    gammas = np.empty_like(pressures)
    print T, 'K'
    for i, P in enumerate(pressures):
        hcp.set_state(P, T)
        Cvs[i] = hcp.C_v/burnman.constants.gas_constant
        alphas[i] = hcp.alpha
        aKTs[i] = hcp.alpha*hcp.K_T
        gammas[i] = hcp.gr

    arr_Cv.append([T, Cvs])
    arr_alpha.append([T, alphas])
    arr_aKT.append([T, aKTs])
    arr_gamma.append([T, gammas])

for T, arr in arr_Cv:
    print T
    plt.plot(pressures/1.e9, arr, label=str(T)+' K')
plt.title("C_V")
plt.legend(loc="lower right")
plt.ylim(3.5, 5.5)
plt.xlim(50., 350.)
plt.show()

for T, arr in arr_alpha:
    plt.plot(pressures/1.e9, arr*1.e5, label=str(T)+' K')
plt.title("Thermal expansivity")
plt.legend(loc="lower right")
plt.ylim(0.5, 4.)
plt.xlim(0., 350.)
plt.show()


for T, arr in arr_aKT:
    plt.plot(pressures/1.e9, arr, label=str(T)+' K')
plt.title("alpha * K_T")
plt.legend(loc="lower right")
plt.ylim(9.e6, 16.e6)
plt.xlim(50., 350.)
plt.show()

for T, arr in arr_gamma:
    plt.plot(pressures/1.e9, arr, label=str(T)+' K')
plt.title("Gruneisen")
plt.legend(loc="lower right")
plt.ylim(1.36, 1.6)
plt.xlim(50., 350.)
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
