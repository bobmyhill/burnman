import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg

#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import Mineral
from burnman import minerals
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()


class B20_FeSi (Mineral):
    def __init__(self):
        formula='FeSi'
        formula = dictionarize_formula(formula)
        Z = 4.
        V_0 = 90.25*1.e-30*burnman.constants.Avogadro/Z # To fit Vocadlo
        #a = 0.448663
        #V_0 = a*a*a*1.e-27*burnman.constants.Avogadro/Z # Acker
        self.params = {
            'name': 'B20 FeSi',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': -88733. ,
            'V_0': V_0 ,
            'K_0': 2.02e+11 ,
            'Kprime_0': 4.4 ,
            'Debye_0': 460.0 , # 596. Acker
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

if __name__ == "__main__":

    FeSi = B20_FeSi()
    
    B20_V_data_RT = burnman.tools.array_from_file('data/Fischer_et_al_FeSi_PTaaerr_B20.dat')
    pressures, temperatures, a, aerr = B20_V_data_RT
    pressures = pressures*1.e9
    PT = [pressures, temperatures]
    Z = 4.
    V = a*a*a*1.e-30*burnman.constants.Avogadro/Z
    V_sigma = V*3.*(aerr/a)
    
    popt, pcov = burnman.tools.fit_PVT_data(FeSi, ['K_0'], PT, V, V_sigma)
    for i, p in enumerate(popt):
        print p, np.sqrt(pcov[i][i])

    plt.errorbar(pressures/1.e9, V, yerr=V_sigma, marker='x', linestyle='None')
    volumes = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        FeSi.set_state(P, temperatures[i])
        volumes[i] = FeSi.V
    plt.plot(pressures/1.e9, volumes)
    plt.show()

    P = 1.e5
    temperatures = np.linspace(20., 1683., 101)
    volumes = np.empty_like(temperatures)
    volumes2 = np.empty_like(temperatures)
    alphas = np.empty_like(temperatures)
    Hs = np.empty_like(temperatures)
    Ss = np.empty_like(temperatures)
    Cps = np.empty_like(temperatures)
    grs = np.empty_like(temperatures)

    
    FeSi.set_state(P, 273.15)
    H273 = FeSi.H

    
    #FeSi.set_state(P, 300.)
    #print FeSi.H, 'should be -73890 (Acker) or -78852. (Barin)'
    #print FeSi.params['F_0'] - (FeSi.H + 73890)
    #exit()
    
    for i, T in enumerate(temperatures):
        FeSi.set_state(P, T)
        volumes[i] = FeSi.V
        alphas[i] = FeSi.alpha
        Ss[i] = FeSi.S
        Cps[i] = FeSi.C_p
        Hs[i] = FeSi.H - H273
        grs[i] = FeSi.gr

        FeSi.set_state(40.e9, T)
        volumes2[i] = FeSi.V
        
    plt.plot(temperatures, grs, label='model')
    plt.legend(loc='lower right')
    plt.title("grueneisen")
    plt.xlabel("Temperature (K)")
    plt.show()
    
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
    
    Fischer_volumes = np.array([[1205,	39.3,	1.14501E-05],
                                [1373,	39.1,	1.15016E-05],
                                [1532,	39.9,	1.15663E-05],
                                [1667,	39.6,	1.16833E-05],
                                [2084,	39.9,	1.17225E-05]]).T
    
    fig1 = mpimg.imread('data/Vocadlo_et_al_FeSi_volumes.png')  # Uncomment these two lines if you want to overlay the plot on a screengrab from SLB2011
    plt.imshow(fig1, extent=[0., 1200., 89.*1.e-30*burnman.constants.Avogadro/4., 95.*1.e-30*burnman.constants.Avogadro/4.], aspect='auto')
    plt.plot(temperatures, volumes, linewidth=4., label='model B20 at 1 bar')
    plt.plot(temperatures, volumes2, linewidth=4., label='model B20 at 40 GPa')
    plt.plot(Fischer_volumes[0], Fischer_volumes[2], marker='o', linestyle='None', label='Fischer data B2 at 40 GPa')
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
    
