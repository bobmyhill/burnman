import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from scipy.optimize import fsolve
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
import matplotlib.image as mpimg
atomic_masses=read_masses()

'''
from scipy.constants import physical_constants
r_B = physical_constants['Bohr radius'][0]
print r_B
print 47.e-30 / np.power(r_B, 3.) / 4.
exit()
'''
#print 'NB: also possibility of an anharmonic contribution'

class fcc_iron (burnman.Mineral):
    def __init__(self):
        formula='Fe'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FCC iron',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': -2784.,
            'V_0': 6.97e-6 ,
            'K_0': 145.e9 ,
            'Kprime_0': 5.3 ,
            'Debye_0': 285. ,
            'grueneisen_0': 1.90 , # 2. ok
            'q_0': 0.06 , # 0., ok
            'Cv_el': 2.7,
            'T_el': 9200., # 10000. ok
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        burnman.Mineral.__init__(self)

if __name__ == "__main__":
    fcc = fcc_iron()

    # First, compare gibbs free energy and entropy at the alpha -> gamma transition
    fcc.set_state(1.e5, 1184.)
    print fcc.gibbs, 'should be -55437. Correction:', fcc.gibbs + 55437.
    print fcc.S, 'should be 75.962. Correction:', fcc.S - 75.962


    fig1 = mpimg.imread('data/Fe_Cp_Desai_1986.png')  # Uncomment these two lines if you want to overlay the plot on a screengrab from SLB2011
    plt.imshow(fig1, extent=[800., 2000.0, 20., 50.], aspect='auto')
    
    temperatures = np.linspace(1185., 1667., 21)
    Cps = np.empty_like(temperatures)
    Ss = np.empty_like(temperatures)
    P = 1.e5
    for i, T in enumerate(temperatures):
        fcc.set_state(P, T)
        Cps[i] = fcc.C_p
        Ss[i] = fcc.S
        #print T, fcc.C_p, fcc.S, fcc.gibbs

    plt.plot(temperatures, Cps, linewidth=4)
    #fcc_Cp_data = burnman.tools.array_from_file('data/fcc_Cp_Chen_Sundman_2001.dat')
    #plt.plot(fcc_Cp_data[0], fcc_Cp_data[1], marker='o', linestyle='None')
    #fcc_Cp_data = burnman.tools.array_from_file('data/fcc_Cp_Rogez_le_Coze_1980.dat')
    #plt.plot(fcc_Cp_data[0], fcc_Cp_data[1], marker='o', linestyle='None')
    plt.xlim(1000., 1800.)
    plt.ylim(30., 45.)
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


    fcc_V_data_B = burnman.tools.array_from_file('data/Basinski_et_al_1955_fcc_volumes_RP.dat')
    plt.plot(fcc_V_data_B[0], fcc_V_data_B[1]/11.7024*7.17433e-6, marker='o', linestyle='None')
    plt.plot(fcc_V_data_B[0], 0.990*fcc_V_data_B[1]/11.7024*7.17433e-6, marker='o', linestyle='None')


    fcc_V_data_F = burnman.tools.array_from_file('data/Feng_2015_fcc_volumes_RP.dat')
    plt.plot(fcc_V_data_F[0], np.power(fcc_V_data_F[1], 3.)/1.e27*burnman.constants.Avogadro/4., marker='o', linestyle='None')

    plt.show()


    # High temperature data from Nishihara et al., 2012
    fcc_V_data = burnman.tools.array_from_file('data/Nishihara_et_al_2012_fcc_volumes.dat')
    P_N, T_N, V_N, Verr_N = fcc_V_data
    P_N = P_N*1.e9
    V_N = V_N/1.e30*burnman.constants.Avogadro/4.
    Verr_N = Verr_N/1.e30*burnman.constants.Avogadro/4.

    for T in [1223., 1273., 1323.]:
        pressures = np.linspace(1.e5, 30.e9, 101)
        for i, P in enumerate(pressures):
            fcc.set_state(P, T)
            volumes[i] = fcc.V


        plt.plot(pressures/1.e9, volumes)
    plt.plot(P_N/1.e9, V_N, marker='o', linestyle='None')
    plt.show()


    Ps = fcc_V_data_B[0]*0. + 1.e5
    Ts = fcc_V_data_B[0]
    Vs = 0.99*fcc_V_data_B[1]/11.7024*7.17433e-6 # the 0.9905 correction is to fit the Onink et al data

    Ps = fcc_V_data_F[0]*0. + 1.e5
    Ts = fcc_V_data_F[0]
    Vs = np.power(fcc_V_data_F[1], 3.)/1.e27*burnman.constants.Avogadro/4.
    
    Ps = np.concatenate((Ps, P_N))
    Ts = np.concatenate((Ts, T_N))
    PTs = [Ps, Ts]
    Vs = np.concatenate((Vs, V_N))
    
    popt, pcov = burnman.tools.fit_PVT_data(fcc, ['V_0', 'K_0', 'grueneisen_0', 'q_0'], PTs, Vs)
    print popt, pcov
