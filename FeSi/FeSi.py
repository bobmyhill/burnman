import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import Mineral
from burnman import minerals
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()


class FeSi_B20 (Mineral):
    def __init__(self):
        formula='FeSi'
        formula = dictionarize_formula(formula)
        a = 0.448663
        Z = 4.
        V_0 = a*a*a*1.e-27*burnman.constants.Avogadro/Z # Acker
        self.params = {
            'name': 'FeSi',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': -242000.0 ,
            'V_0': V_0 ,
            'K_0': 1.79e+11 ,
            'Kprime_0': 4.9 ,
            'Debye_0': 596.0 , # 596. Acker
            'grueneisen_0': 1.53 ,
            'q_0': 1. ,
            'G_0': 59000000000.0 ,
            'Gprime_0': 1.4 ,
            'eta_s_0': -0.1 ,
            'T_el': 3500.,
            'Cv_el': 2.7,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)


FeSi = FeSi_B20()


P = 1.e5
temperatures = np.linspace(20., 2500., 101)
volumes = np.empty_like(temperatures)
Ss = np.empty_like(temperatures)
Cps = np.empty_like(temperatures)

for i, T in enumerate(temperatures):
    FeSi.set_state(P, T)
    volumes[i] = FeSi.V
    Ss[i] = FeSi.S
    Cps[i] = FeSi.C_p


Acker_data = burnman.tools.array_from_file("data/FeSi_Acker.dat")
T, Cp, DS, DH, phi = Acker_data
Cp = Cp*burnman.constants.gas_constant
DS = DS*burnman.constants.gas_constant

Barin_data = burnman.tools.array_from_file("data/Barin_FeSi_B20.dat")
T_B, Cp_B = Barin_data

plt.plot(temperatures, volumes, label='model')
plt.legend(loc='lower right')
plt.title("Volumes")
plt.xlabel("Temperature (K)")
plt.show()

plt.plot(temperatures, Cps, label='model')
plt.plot(T, Cp, marker='o', linestyle='None')
plt.plot(T_B, Cp_B, marker='o', linestyle='None')
plt.legend(loc='lower right')
plt.title("Cps")
plt.xlabel("Temperature (K)")
plt.show()


plt.plot(temperatures, Ss, label='model')
plt.plot(T, DS, marker='o', linestyle='None')
plt.legend(loc='lower right')
plt.title("Ss")
plt.xlabel("Temperature (K)")
plt.show()

