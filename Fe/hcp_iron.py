import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()


from scipy.constants import physical_constants
r_B = physical_constants['Bohr radius'][0]
V_B = np.power(r_B, 3.)

print 'NB: also possibility of an anharmonic contribution'

class hcp_iron (burnman.Mineral):
    def __init__(self):
        formula='Fe'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'HCP iron',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': -4165.,
            'V_0': 6.764e-6, # 6.733e-6 ,
            'K_0': 161.9e9, # 166.e9 ,
            'Kprime_0': 5.15, # 5.32 ,
            'Debye_0': 395. ,
            'grueneisen_0': 2.0 ,
            'q_0': 1.0 ,
            'Cv_el': 3.0, # 2.7,
            'T_el': 6000., # 6500.
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        burnman.Mineral.__init__(self)


if __name__ == "__main__":
    hcp = hcp_iron()
    print hcp.params['V_0']/V_B/burnman.constants.Avogadro

    """
    Room temperature EoS: data
    """  

    PV_data = burnman.tools.array_from_file('data/Dewaele_et_al_2006_iron_using_ruby.dat')
    T_D, P_D, V_D, V_D_err = PV_data
    P_D = P_D*1.e9
    V_D = V_D/1.e30*burnman.constants.Avogadro
    V_D_err = V_D_err/1.e30*burnman.constants.Avogadro
    
    """
    Room temperature EoS: fitting 
    """
    
    PT_D = [P_D, T_D]
    popt, pcov = burnman.tools.fit_PVT_data(hcp, ['V_0', 'K_0', 'Kprime_0'], PT_D, V_D, V_D_err)
    print popt
    print pcov
    
    
    """
    Room temperature EoS: Plotting
    """
    
    pressures = np.linspace(1.e5, 350.e9, 101)
    volumes = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        hcp.set_state(P, 298.15)
        volumes[i] = hcp.V
        
    
    PV_data = burnman.tools.array_from_file('data/Dewaele_et_al_2006_iron_using_tungsten.dat')
    T_Dall, P_Dall, V_Dall, V_Dall_err = PV_data
    P_Dall = P_Dall*1.e9
    V_Dall = V_Dall/1.e30*burnman.constants.Avogadro
    V_Dall_err = V_Dall_err/1.e30*burnman.constants.Avogadro

    plt.plot(pressures/1.e9, volumes)
    plt.plot(P_Dall/1.e9, V_Dall, marker='o', linestyle='None')
    plt.plot(P_D/1.e9, V_D, marker='o', linestyle='None')
    plt.show()


    '''
    pressures = np.linspace(50.e9, 200.e9, 101)
    volumes = np.empty_like(pressures)
    gammas = np.empty_like(pressures)
    qs = np.empty_like(pressures)
    for T in [1000., 2000., 3000.]:
    for i, P in enumerate(pressures):
    hcp.set_state(P, T)
    volumes[i] = hcp.V
    gammas[i] = hcp.gr
    dP = 1.e6
    hcp.set_state(P+dP, T)
    qs[i] = (np.log(gammas[i]) - np.log(hcp.gr))/(np.log(volumes[i]) - np.log(hcp.V))
    plt.plot(volumes/V_B/burnman.constants.Avogadro, gammas, label=str(T)+' K (gr)')
    plt.plot(volumes/V_B/burnman.constants.Avogadro, qs, label=str(T)+' K (q)')
    
    
    plt.legend(loc="upper left")
    plt.xlim(45, 65)
    plt.ylim(1., 3.)
    plt.show()
    '''
    
    """
    Figure 12a of Wasserman et al.
    """
    pressures = np.linspace(1.e5, 200.e9, 101)
    alphas = np.empty_like(pressures)
    for T in [715., 1000., 2000.]:
        for i, P in enumerate(pressures):
            hcp.set_state(P, T)
            alphas[i] = hcp.alpha
        plt.plot(pressures/1.e9, alphas, label=str(T)+' K')
        
    """
    Duffy and Ahrens shock data
    """
    P = 202.e9
    T_0 = 300.
    T = 5200.
    hcp.set_state(P, T_0)
    V_0 = hcp.V
    hcp.set_state(P, T)
    V = hcp.V
    
    abar = np.log(V/V_0)/(T - T_0)
    print abar, 'should be 9.1 +/- 2 x 10^-6 /K'
    plt.plot(P/1.e9, abar, marker='o', label='Duffy and Ahrens')
    
    plt.legend(loc="upper left")
    plt.xlim(0, 300)
    plt.ylim(0., 10.e-5)
    plt.show()
    

    pressures = np.linspace(50.e9, 350.e9, 51)
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


    """
    HUGONIOT
    """

    hugoniot_data = burnman.tools.array_from_file('data/iron_hugoniot.dat')
    P_obs, P_err_obs, rho_obs, rho_err_obs = hugoniot_data
    P_obs = 1.e9*P_obs
    P_err_obs = 1.e9*P_err_obs
    V_obs = 55.845*1.e-6/rho_obs # Mg/m^3
    V_err_obs = V_obs*rho_err_obs/rho_obs # Mg/m^3

    bcc = burnman.minerals.HP_2011_ds62.iron()

    pressures = np.linspace(1.e5, 300.e9, 21)
    for T in [298.15, 500.]:
        temperatures, volumes = burnman.tools.hugoniot(hcp, 1.e5, T, pressures, bcc)
        plt.plot(pressures/1.e9, volumes, label=str(T)+' K')

    plt.errorbar(P_obs/1.e9, V_obs, xerr=P_err_obs/1.e9, yerr=V_err_obs, marker='o', linestyle='None')
    plt.xlim(0., 300.)
    plt.legend(loc='upper left')
    plt.title("Hugoniot")
    plt.show()

####
# Checks
####


'''

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
'''

'''
print 'slb'
hcp_slb.set_state(370.e9, 298.15)

print 'slbel3'
hcp.set_state(370.e9, 298.15)
print hcp.gibbs, hcp_slb.gibbs
print hcp.V, hcp_slb.V
print hcp.C_v, hcp_slb.C_v
print hcp.C_p, hcp_slb.C_p
print hcp.S, hcp_slb.S
exit()
'''

'''
def volume_dependent_q(x):
    """
    Finite strain approximation for :math:`q`, the isotropic volume strain
    derivative of the grueneisen parameter.
    """
    grueneisen_0 = 1.72
    q_0 = 1.
    f = 1./2. * (pow(x, 2./3.) - 1.)
    a1_ii = 6. * grueneisen_0 # EQ 47
    a2_iikk = -12.*grueneisen_0+36.*pow(grueneisen_0,2.) - 18.*q_0*grueneisen_0 # EQ 47
    nu_o_nu0_sq = 1.+ a1_ii*f + (1./2.)*a2_iikk * f*f # EQ 41
    gr = 1./6./nu_o_nu0_sq * (2.*f+1.) * ( a1_ii + a2_iikk*f )
    q = 1./9.*(18.*gr - 6. - 1./2. / nu_o_nu0_sq * (2.*f+1.)*(2.*f+1.)*a2_iikk/gr)
    return q


for x in [0.7, 0.8, 0.9, 1.0]:
    print volume_dependent_q(1./x)

exit()
'''
