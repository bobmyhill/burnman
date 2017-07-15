import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg

#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import Mineral
from burnman import minerals
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()

class B2_FeSi (Mineral):
    def __init__(self):
        formula='FeSi'
        formula = dictionarize_formula(formula)
        Z = 1.
        V_0 = 2.*6.414e-6 # 21.74*1.e-30*burnman.constants.Avogadro/Z # Dobson et al., 2003
        self.params = {
            'name': 'B2 FeSi',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': -71746. ,
            'V_0': V_0 ,
            'K_0': 230.6e9, # Fischer et al., 2014 #184.e+9 , # Dobson et al., 2003
            'Kprime_0': 4.17, # Fischer et al., 2014 #4.2 , # Dobson et al., 2003
            'Debye_0': 490.0 , # To change entropies
            'grueneisen_0': 2.5 ,
            'q_0': 0.5 ,
            'G_0': 59000000000.0 ,
            'Gprime_0': 1.4 ,
            'eta_s_0': -0.1 ,
            'T_el': 7000.,
            'Cv_el': 2.7,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)
'''
class B2_FeSi (Mineral):
    def __init__(self):
        formula='FeSi'
        formula = dictionarize_formula(formula)
        Z = 1.
        V_0 = 2.*6.414e-6 # 21.74*1.e-30*burnman.constants.Avogadro/Z # Dobson et al., 2003
        self.params = {
            'name': 'FeSi',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': -221356. ,
            'V_0': V_0 ,
            'K_0': 230.6e9, # Fischer et al., 2014 #184.e+9 , # Dobson et al., 2003
            'Kprime_0': 4.17, # Fischer et al., 2014 #4.2 , # Dobson et al., 2003
            'Debye_0': 417.0 , # Fischer
            'grueneisen_0': 1.3 , # Fischer
            'q_0': 1.7 , # Fischer
            'G_0': 59000000000.0 ,
            'Gprime_0': 1.4 ,
            'eta_s_0': -0.1 ,
            'T_el': 7000.,
            'Cv_el': 2.7,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)
'''
if __name__ == "__main__":
    from B20_FeSi import B20_FeSi

    B20 = B20_FeSi()
    B2  = B2_FeSi()
    
    # Transition pressure = 36.e9 Pa at 300 K
    P = 30.e9
    T = 1500.
    B2.set_state(P, T)
    B20.set_state(P, T)

    B2.params['F_0'] = B2.params['F_0'] - (B2.gibbs - B20.gibbs)
    print B2.params['F_0']
    exit()
    T = 300.
    pressures = np.linspace(1.e5, 100.e9, 101)
    V_B2 = np.empty_like(pressures)
    V_B20 = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        B2.set_state(P, T)
        B20.set_state(P, T)

        V_B2[i] = B2.V
        V_B20[i] = B20.V

    plt.plot(pressures/1.e9, V_B2, label='B2 (HP)')
    plt.plot(pressures/1.e9, V_B20, label='B20 (LP)')
    plt.legend(loc='upper right')
    plt.show()

    
    temperatures = np.linspace(300., 3000., 20.)
    pressures = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        pressures[i] = burnman.tools.equilibrium_pressure([B2, B20], [1., -1.], T)

    plt.plot(pressures/1.e9, temperatures)
    plt.xlim(0., 50.)
    plt.show()
    
