from __future__ import absolute_import

# Here we import standard python modules that are required for
# usage of BurnMan.  In particular, numpy is used for handling
# numerical arrays and mathematical operations on them, and
# matplotlib is used for generating plots of results of calculations
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import minerals

class andradite (burnman.Mineral):

    def __init__(self):
        formula = 'Ca3Fe2Si3O12'
        formula = burnman.processchemistry.dictionarize_formula(formula)
        self.params = {
            'name': 'Andradite',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -5769.08e3 - 300.*318.5 + 1.e5*13.204e-6, # To fit
            'V_0': 132.04e-6, # Holland and Powell, 2011
            'K_0': 158.8e9 , # Holland and Powell, 2011
            'Kprime_0': 4.71, # Jiang et al., 2004
            'Debye_0': 721., # Robie et al., 1987
            'grueneisen_0': 1.15, # To fit thermal expansion
            'q_0': 1.4, # Standard value in SLB2011
            'G_0': 89.7e9, # Jiang et al., 2004
            'Gprime_0': 1.25, # Jiang et al., 2004
            'eta_s_0': 2.4,  # Grossular value in SLB2011
            'n': sum(formula.values()),
            'molar_mass': burnman.processchemistry.formula_mass(formula, burnman.processchemistry.read_masses())}

        self.property_modifiers = [
            ['landau', {'Tc_0': 11.7, 'S_D': 15., 'V_D': 0.}]]
        # To fit
        
        self.uncertainties = {
            'err_F_0': 1000.0,
            'err_V_0': 0.0,
            'err_K_0': 1000000000.0,
            'err_K_prime_0': 0.1,
            'err_Debye_0': 31.0,
            'err_grueneisen_0': 0.05,
            'err_q_0': 1.0,
            'err_G_0': 1000000000.0,
            'err_Gprime_0': 0.1,
            'err_eta_s_0': 1.0}
        burnman.Mineral.__init__(self)


if __name__ == "__main__":
    andr = andradite()
    
    P = 1.e5
    T = 300.
    
    andr.set_state(P, T)
    print(andr.K_S/1.e9, andr.rho/1.e3, andr.C_p, andr.alpha)
    
    
    temperatures = np.linspace(5., 1000., 501)
    Cps = np.empty_like(temperatures)
    Ss = np.empty_like(temperatures)
    Vs = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        andr.set_state(P, T)
        Cps[i] = andr.C_p
        Ss[i] = andr.S
        Vs[i] = andr.V
        #print(T, Cps[i], andr.alpha*1.e6)
        
    T_obs, Cp_obs, S_obs = np.loadtxt(fname = 'andr_Cp_S.dat', unpack=True)
    T_obs2, Cp_obs2 = np.loadtxt(fname = 'andr_Cp.dat', unpack=True)
    
    plt.plot(temperatures, Cps)
    plt.plot(T_obs, Cp_obs, marker='.', linestyle='None')
    plt.plot(T_obs2, Cp_obs2, marker='.', linestyle='None')
    plt.show()
    
    
    plt.plot(temperatures, Ss)
    plt.plot(T_obs, S_obs, marker='.', linestyle='None')
    plt.show()
    
    plt.plot(temperatures, Vs)
    plt.show()
    
