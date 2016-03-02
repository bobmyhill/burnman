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
        V_0 = 21.33*1.e-30*burnman.constants.Avogadro/Z
        self.params = {
            'name': 'FeSi',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': -242000.0 ,
            'V_0': V_0 ,
            'K_0': 223.3e+9 ,
            'Kprime_0': 5.5 ,
            'Debye_0': 460.0 , # 596. Acker
            'grueneisen_0': 2.5 ,
            'q_0': 0. ,
            'G_0': 59000000000.0 ,
            'Gprime_0': 1.4 ,
            'eta_s_0': -0.1 ,
            'T_el': 7000.,
            'Cv_el': 2.7,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

if __name__ == "__main__":

    FeSi = B2_FeSi()

    print FeSi.params['V_0']

    FeSi.set_state(1.e5, 300.)
    
    print FeSi.K_T/1.e9, FeSi.alpha*FeSi.K_T/1.e9, 
    exit()
    
    P = 1.e5
    temperatures = np.linspace(20., 1683., 101)
    volumes = np.empty_like(temperatures)
    alphas = np.empty_like(temperatures)
    Hs = np.empty_like(temperatures)
    Ss = np.empty_like(temperatures)
    Cps = np.empty_like(temperatures)

    
    FeSi.set_state(P, 273.15)
    H273 = FeSi.H
    for i, T in enumerate(temperatures):
        FeSi.set_state(P, T)
        volumes[i] = FeSi.V
        alphas[i] = FeSi.alpha
        Ss[i] = FeSi.S
        Cps[i] = FeSi.C_p
        Hs[i] = FeSi.H - H273

    Acker_data = burnman.tools.array_from_file("data/FeSi_Acker.dat")
    T, Cp, DS, DH, phi = Acker_data
    Cp = Cp*burnman.constants.gas_constant
    DS = DS*burnman.constants.gas_constant
    
    Barin_data = burnman.tools.array_from_file("data/Barin_FeSi_B20.dat")
    T_B, Cp_B = Barin_data
    
    fig1 = mpimg.imread('data/Vocadlo_et_al_FeSi_alphas.png')  # Uncomment these two lines if you want to overlay the plot on a screengrab from SLB2011
    plt.imshow(fig1, extent=[0., 1200., 0., 6.e-5], aspect='auto')
    plt.plot(temperatures, alphas, label='model')
    plt.legend(loc='lower right')
    plt.title("alphas")
    plt.xlabel("Temperature (K)")
    plt.show()
    
    
    fig1 = mpimg.imread('data/Vocadlo_et_al_FeSi_volumes.png')  # Uncomment these two lines if you want to overlay the plot on a screengrab from SLB2011
    plt.imshow(fig1, extent=[0., 1200., 89.*1.e-30*burnman.constants.Avogadro/4., 95.*1.e-30*burnman.constants.Avogadro/4.], aspect='auto')
    plt.plot(temperatures, volumes, linewidth=4., label='model')
    plt.legend(loc='lower right')
    plt.title("Volumes")
    plt.xlabel("Temperature (K)")
    plt.show()
    
    
    plt.plot(temperatures, Cps, label='model')
    plt.plot(T, Cp, marker='o', linestyle='None')
    #plt.plot(T_B, Cp_B, marker='o', linestyle='None')
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

    fig1 = mpimg.imread('data/H_H273_FeSi_Acker_200_1800_0_90.png')  # Uncomment these two lines if you want to overlay the plot on a screengrab from SLB2011
    plt.imshow(fig1, extent=[200., 1800., 0., 90.], aspect='auto')
    plt.plot(temperatures, Hs*1.e-3, label='model')
    plt.legend(loc='lower right')
    plt.title("H - H273 (kJ/mol)")
    plt.xlim(200., 1800.)
    plt.ylim(0., 90.)
    plt.xlabel("Temperature (K)")
    plt.show()
    
