import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import Mineral
from burnman import minerals
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()


class B1_wuestite (Mineral):
    def __init__(self):
        formula='FeO'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'B1 FeO',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': -275450. ,
            'V_0': 1.2239e-05 , # 1.2239 from Simons (1980); 1.222 from Stolen et al., 1996; 1.216 from Katsura et al., 1967; 1.225 Hentschel, 1970
            'K_0': 1.80e+11 , #147 GPa from Hazen and Finger 1982;  179 GPa in good agreement with K_S of Sumino et al; 184.3 from Will et al., 1980
            'Kprime_0': 4.93 , # Fixed (Stixrude fit suggests 4.93)
            'Debye_0': 460.0 ,
            'grueneisen_0': 1.40 ,
            'q_0': 1.5 ,
            'T_el': 1400., # 4000.
            'Cv_el': 0.98, # 3
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        self.property_modifiers = [
            ['landau', {'P_0' : 1.e5,
                        'T_0' : 298.15,
                        'Tc_0':200.,
                        'S_D':12.,
                        'V_D': 0.}]] 
        # V_D ~ 5e-8 (Sumino et al., 1980; Table 3b)
        # but likely to get smaller with increasing pressure
        Mineral.__init__(self)
        
'''
class B1_wuestite (Mineral):
    def __init__(self):
        formula='FeO'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Wuestite',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': -275450. ,
            'V_0': 1.2239e-05 , # 1.2239 from Simons (1980); 1.222 from Stolen et al., 1996; 1.216 from Katsura et al., 1967; 1.225 Hentschel, 1970
            'K_0': 1.79e+11 , # Hazen and Finger 1982;  179 GPa in good agreement with K_S of Sumino et al; 184.3 from Will et al., 1980
            'Kprime_0': 4.0 , # Fixed
            'Debye_0': 460.0 ,
            'grueneisen_0': 1.72 ,
            'q_0': 2.2 ,
            'T_el': 6000., # 4000.
            'Cv_el': 2.7, # 3
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        self.landau = {
            'Tc_0':200.,
            'S_D':12.,
            'V_D': 0.} 
        # V_D ~ 5e-8 (Sumino et al., 1980; Table 3b)
        # but likely to get smaller with increasing pressure
        Mineral.__init__(self)
'''

if __name__ == "__main__":
    temperatures, volumes, compositions = np.loadtxt(fname='data/Hayakawa_CR_1972_wustite_TVx.dat', unpack=True)
    def volume(args, V_0, a, b):
        X, T = args
        return V_0 + X*a + T*b 
    
    from scipy.optimize import curve_fit
    
    popt, pcov = curve_fit(volume, [compositions, temperatures], volumes)

    fper = B1_wuestite()
    fper_HP = minerals.HP_2011_ds62.fper()
    fper_SLB = minerals.SLB_2011.wuestite()


    P = 1.e5
    T = 298.15

    fper.set_state(P, T)
    fper_HP.set_state(P, T)
    print fper.S
    print fper.alpha
    #print fper.V


    fper.set_state(P, 1200)
    print fper.alpha

    
    B1_EoS_data = burnman.tools.array_from_file('data/Fischer_2011_FeO_B1_EoS.dat')
    PTs = [B1_EoS_data[0]*1.e9, B1_EoS_data[2]]
    Vs = B1_EoS_data[10]*1.e-6
    V_sigmas = B1_EoS_data[11]*1.e-6
    #popt, pcov = burnman.tools.fit_PVT_data(fper, ['K_0', 'Kprime_0'], PTs, Vs, V_sigmas)
    #print popt
    
    
    temperatures = np.linspace(1., 1650., 501)
    volumes = np.empty_like(temperatures)
    Ss = np.empty_like(temperatures)
    Cps = np.empty_like(temperatures)
    K_Ss = np.empty_like(temperatures)
    volumes_HP = np.empty_like(temperatures)
    Ss_HP = np.empty_like(temperatures)
    Cps_HP = np.empty_like(temperatures)
    volumes_SLB = np.empty_like(temperatures)
    Ss_SLB = np.empty_like(temperatures)
    Cps_SLB = np.empty_like(temperatures)
    
    for i, T in enumerate(temperatures):
        fper.set_state(P, T)
        fper_HP.set_state(P, T)
        fper_SLB.set_state(P, T)
        volumes[i] = fper.V
        Ss[i] = fper.S
        Cps[i] = fper.C_p
        K_Ss[i] = fper.K_S
        volumes_HP[i] = fper_HP.V
        Ss_HP[i] = fper_HP.S
        Cps_HP[i] = fper_HP.C_p
        volumes_SLB[i] = fper_SLB.V
        Ss_SLB[i] = fper_SLB.S
        Cps_SLB[i] = fper_SLB.C_p

    fper.set_state(1.e5, 298.15)
    S0 = fper.S
    
    T_C, Hdiff, Sdiff = np.loadtxt(unpack='True', fname="data/Coughlin_KB_1951_FeO_HS_solid_liquid.dat")
    DS_C = 4.184*Sdiff*(2./1.947) + S0
    T, Cp, DH, DS, phi = burnman.tools.array_from_file("data/FeO_Cp.py")
    T_nonstoic, Cp_nonstoic = burnman.tools.array_from_file("data/Fe0.9374O_Cp.dat")
    T_J, Cp_J, DS_J, GHT_J, H_J, fH_J, fG_J, logKf_J = burnman.tools.array_from_file("data/FeO_Cp_JANAF.py")


    np.savetxt(header='Temperatures (K), Cp (J/K/mol)', X=zip(*[temperatures, Cps_SLB]), fname='output/wus_SLB_T_Cp.dat')
    np.savetxt(header='Temperatures (K), Cp (J/K/mol)', X=zip(*[temperatures, Cps]), fname='output/wus_current_T_Cp.dat')

    V_McCammon = lambda T: fper.params['V_0']*(1. - 3.78e-3 + 4.529e-5*T) 
    
    plt.plot(temperatures, volumes, label='model')
    plt.plot(temperatures, volumes_HP, label='model HP')
    plt.plot(temperatures, volumes_SLB, label='model SLB')
    plt.plot(1223.15, volume([1.0, 1223.15], *popt), marker='o', label='Hayakawa et al., 950C')
    plt.plot(1323.15, volume([1.0, 1323.15], *popt), marker='o', label='Hayakawa et al., 1050C')
    plt.plot(temperatures, V_McCammon(temperatures), label='McCammon model')
    plt.legend(loc='lower right')
    plt.title("Volumes")
    plt.xlabel("Temperature (K)")
    plt.show()

    
    plt.plot(T_nonstoic, (2./1.9374)*Cp_nonstoic, marker='.', linestyle='None')
    plt.plot(T, Cp, marker='.', linestyle='None')
    plt.plot(temperatures, Cps, label='model')
    plt.plot(temperatures, Cps_HP, label='model HP')
    plt.plot(temperatures, Cps_SLB, label='model SLB')
    plt.plot(T_J, Cp_J, marker='o', linestyle='None')
    plt.legend(loc='lower right')
    plt.title("Cps")
    plt.xlabel("Temperature (K)")
    plt.ylim(0., 100.)
    plt.show()
    

    plt.plot(temperatures, Ss, label='model')
    plt.plot(temperatures, Ss_HP, label='model HP')
    plt.plot(temperatures, Ss_SLB, label='model SLB')
    plt.plot(T, DS, marker='o', linestyle='None')
    plt.plot(T_J, DS_J, marker='o', linestyle='None')
    plt.plot(T_C, DS_C, marker='o', linestyle='None')
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


    T = 298.15
    pressures = np.linspace(1.e5, 100.e9, 101)
    volumes = np.empty_like(pressures)
    volumes_HP = np.empty_like(pressures)
    volumes_SLB = np.empty_like(pressures)

    for i, P in enumerate(pressures):
        fper.set_state(P, T)
        fper_HP.set_state(P, T)
        fper_SLB.set_state(P, T)
        volumes[i] = fper.V
        volumes_HP[i] = fper_HP.V
        volumes_SLB[i] = fper_SLB.V

    
    plt.plot(pressures/1.e9, volumes, label='model')
    plt.plot(pressures/1.e9, volumes_HP, label='model HP')
    plt.plot(pressures/1.e9, volumes_SLB, label='model SLB')
    plt.legend(loc='upper right')
    plt.title("Volumes")
    plt.xlabel("P (GPa)")
    plt.show()

