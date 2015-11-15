import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()

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


hcp = hcp_iron()
'''
P = 1.e5
T = 2000.

#eos = burnman.eos.slb.SLB3()
eos = burnman.eos.slb_with_el.SLBEL3()

hcp.set_state(P, T)
V = hcp.V

F = eos.helmholtz_free_energy(P, T, V, hcp.params)
H = eos.enthalpy(P, T, V, hcp.params)
S = eos.entropy(P, T, V, hcp.params)
C_v = eos.heat_capacity_v(P, T, V, hcp.params)
K_S = eos.adiabatic_bulk_modulus(P, T, V, hcp.params)
K_T = eos.isothermal_bulk_modulus(P, T, V, hcp.params)

dT = 1.
F0 = eos.helmholtz_free_energy(P, T-0.5*dT, V, hcp.params)
F1 = eos.helmholtz_free_energy(P, T+0.5*dT, V, hcp.params)
print 'S:', S, -(F1-F0)/dT

dFdT0 = (F-F0)/(0.5*dT)
dFdT1 = (F1-F)/(0.5*dT)
Td2FdT2 = -T*(dFdT1 - dFdT0)/(0.5*dT)
print 'C_v:', C_v, Td2FdT2


dV = 1.e-9
F0 = eos.helmholtz_free_energy(P, T, V-0.5*dV, hcp.params)
F1 = eos.helmholtz_free_energy(P, T, V+0.5*dV, hcp.params)

dFdV0 = (F - F0)/(0.5*dV)
dFdV1 = (F1 - F)/(0.5*dV)

Vd2FdV2 = V*(dFdV1 - dFdV0)/(0.5*dV)
print 'K_T:', K_T/1.e9, Vd2FdV2/1.e9
'''

pressures = [50.e9, 100.e9, 150.e9]
temperatures = np.linspace(1., 6000., 101)
alphas = np.empty_like(temperatures)
for P in pressures:
    for i, T in enumerate(temperatures):
        hcp.set_state(P, T)
        alphas[i] = hcp.alpha

    plt.plot(temperatures, alphas, label=str(P/1.e9)+' GPa')
plt.legend(loc="lower left")
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
