import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import Mineral
from burnman import minerals
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()


class wuestite (Mineral):
    def __init__(self):
        formula='FeO'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Wuestite',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': -242000.0 ,
            'V_0': 1.206e-05 ,
            'K_0': 1.79e+11 ,
            'Kprime_0': 4.9 ,
            'Debye_0': 500.0 ,
            'grueneisen_0': 1.4 ,
            'q_0': 0. ,
            'G_0': 59000000000.0 ,
            'Gprime_0': 1.4 ,
            'eta_s_0': -0.1 ,
            'T_el': 4500.,
            'Cv_el': 2.7,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        self.landau = {
            'Tc_0':195.,
            'S_D':11.,
            'V_D': 0.} 
        # V_D ~ 5e-8 (Sumino et al., 1980; Table 3b)
        # but likely to get smaller with increasing pressure
        self.uncertainties = {
            'err_F_0': 1000.0 ,
            'err_V_0': 0.0 ,
            'err_K_0': 1000000000.0 ,
            'err_K_prime_0': 0.2 ,
            'err_Debye_0': 21.0 ,
            'err_grueneisen_0': 0.13 ,
            'err_q_0': 1.0 ,
            'err_G_0': 1000000000.0 ,
            'err_Gprime_0': 0.1 ,
            'err_eta_s_0': 1.0 }
        Mineral.__init__(self)


fper = wuestite()
fper_HP = minerals.HP_2011_ds62.fper()


P = 1.e5
temperatures = np.linspace(1., 300., 101)
volumes = np.empty_like(temperatures)
Ss = np.empty_like(temperatures)
Cps = np.empty_like(temperatures)
K_Ss = np.empty_like(temperatures)
volumes_HP = np.empty_like(temperatures)
Ss_HP = np.empty_like(temperatures)
Cps_HP = np.empty_like(temperatures)

for i, T in enumerate(temperatures):
    fper.set_state(P, T)
    fper_HP.set_state(P, T)
    volumes[i] = fper.V
    Ss[i] = fper.S
    Cps[i] = fper.C_p
    K_Ss[i] = fper.K_S
    volumes_HP[i] = fper_HP.V
    Ss_HP[i] = fper_HP.S
    Cps_HP[i] = fper_HP.C_p


Stolen_data = burnman.tools.array_from_file("data/FeO_Cp.py")
T, Cp, DH, DS, phi = Stolen_data
Stolen_data = burnman.tools.array_from_file("data/Fe0.9374O_Cp.dat")
T_nonstoic, Cp_nonstoic = Stolen_data

JANAF_data = burnman.tools.array_from_file("data/FeO_Cp_JANAF.py")
T_J, Cp_J, DS_J, GHT_J, H_J, fH_J, fG_J, logKf_J = JANAF_data

plt.plot(temperatures, volumes, label='model')
plt.plot(temperatures, volumes_HP, label='model HP')
plt.legend(loc='lower right')
plt.title("Volumes")
plt.xlabel("Temperature (K)")
plt.show()


plt.plot(T_nonstoic, (2./1.9374)*Cp_nonstoic, marker='.', linestyle='None')
plt.plot(T, Cp, marker='.', linestyle='None')
plt.plot(temperatures, Cps, label='model')
plt.plot(temperatures, Cps_HP, label='model HP')
#plt.plot(T_J, Cp_J, marker='o', linestyle='None')
plt.legend(loc='lower right')
plt.title("Cps")
plt.xlabel("Temperature (K)")
plt.ylim(0., 100.)
plt.show()


plt.plot(temperatures, Ss, label='model')
plt.plot(temperatures, Ss_HP, label='model HP')
plt.plot(T, DS, marker='o', linestyle='None')
plt.plot(T_J, DS_J, marker='o', linestyle='None')
plt.legend(loc='lower right')
plt.title("Ss")
plt.xlabel("Temperature (K)")
plt.ylim(0., 200.)
plt.show()


plt.plot(temperatures-273.15, K_Ss, label='model')
plt.legend(loc='lower right')
plt.title("K_Ss")
plt.xlabel("Temperature (C)")
plt.xlim(-100., 30.)
plt.show()
