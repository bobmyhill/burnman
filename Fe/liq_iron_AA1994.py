# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

# Tallon (1980) suggested that melting of simple substances was associated with an entropy change of
# Sfusion = burnman.constants.gas_constant*np.log(2.) + a*K_T*Vfusion
# Realising also that dT/dP = Vfusion/Sfusion, we can express the entropy 
# and volume of fusion in terms of the melting curve:
# Sfusion = burnman.constants.gas_constant*np.log(2.) / (1. - a*K_T*dTdP)
# Vfusion = Sfusion*dT/dP

from scipy.interpolate import interp1d
from scipy.optimize import fsolve, curve_fit
import burnman
from burnman import minerals
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()



class liq_iron (burnman.Mineral):
    def __init__(self):
        formula='Fe'
        formula = dictionarize_formula(formula)
        m = formula_mass(formula, atomic_masses)
        rho_0 = 7019.
        V_0 = m/rho_0
        D = 7766.
        Lambda = 1146.
        self.params = {
            'name': 'liquid iron',
            'formula': formula,
            'equation_of_state': 'aa',
            'P_0': 1.e5,
            'T_0': 1809.,
            'S_0': 99.823, # to fit
            'molar_mass': m,
            'V_0': V_0,
            'E_0': 72700.,
            'K_S': 109.7e9,
            'Kprime_S': 4.661,
            'Kprime_prime_S': -0.043e-9,
            'grueneisen_0': 1.735,
            'grueneisen_prime': -0.130/m*1.e-6,
            'grueneisen_n': -1.870,
            'a': [248.92*m, 289.48*m],
            'b': [0.4057*m, -1.1499*m],
            'Theta': [1747.3, 1.537],
            'theta': 5000.,
            'lmda': [302.07*m, -325.23*m, 30.45*m],
            'xi_0': 282.67*m,
            'F': [D/rho_0, Lambda/rho_0],
            'n': sum(formula.values()),
            'molar_mass': m}
        burnman.Mineral.__init__(self)



if __name__ == "__main__":
    from fcc_iron import fcc_iron
    from hcp_iron import hcp_iron
    
    bcc = minerals.HP_2011_ds62.iron()
    fcc = fcc_iron()
    hcp = hcp_iron()
    liq = liq_iron()

    
    fcc.set_state(5.2e9, 1991.)
    liq.set_state(5.2e9, 1991.)
    liq.params['E_0'] = liq.params['E_0'] + (fcc.gibbs - liq.gibbs)
    print liq.params['E_0'], 'REMEMBER TO CHANGE THIS!'

    fcc.set_state(5.2e9, 1991.0001)
    liq.set_state(5.2e9, 1991.0001)
    print fcc.gibbs, liq.gibbs
    
    dTdP = 3.85
    dTdP_err = 0.1
    DV = 0.352 # cm^3/mol
    print 'DV:', (liq.V - fcc.V)*1.e6, 'should be', DV
    print 'dT/dP:', (liq.V - fcc.V)/(liq.S - fcc.S), 'should be', dTdP, '+/-', dTdP_err 
    

    temperatures = np.linspace(1800., 4000., 101)
    aKT = np.empty_like(temperatures)
    volumes = np.empty_like(temperatures)
    grs = np.empty_like(temperatures)
    for P in [1.e9, 50.e9, 100.e9, 200.e9]:
        for i, T in enumerate(temperatures):
            liq.set_state(P, T)
            aKT[i] = liq.alpha*liq.K_T
            volumes[i] = liq.V
            
            grs[i] = liq.gr
    
        plt.plot(temperatures, grs)
    plt.show()
    exit()

    
    temperatures = np.linspace(1800., 4000., 101)
    Cps = np.empty_like(temperatures)
    volumes = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        liq.set_state(1.e5, T)
        Cps[i] = liq.C_p
        volumes[i] = liq.V
    
    plt.plot(temperatures, Cps)
    plt.show()


    Hixson_data = burnman.tools.array_from_file("data/Fe_1bar_rho_Hixson_et_al_1990.dat")
    Mizuno_data = burnman.tools.array_from_file("../FeSi/data/Mizuno_Fe_melt_VT.dat")
    
    H, T, rho, VoverV0, rhoel = Hixson_data
    V = 55.845/(rho*1.e6)
    V_Mizuno = lambda T: 0.055845/(7162 - 0.735*(T - 1808))
    #plt.plot(temperatures, V_Mizuno(temperatures))
    plt.plot(Mizuno_data[0], 0.055845/Mizuno_data[1], marker='o', linestyle='None')
    plt.plot(T, V, marker='o', linestyle='None')
    plt.plot(temperatures, volumes)
    plt.show()


    melting_curve_data = burnman.tools.array_from_file('data/Anzellini_2013_Fe_melting_curve.dat')
    melting_temperature = interp1d(melting_curve_data[0]*1.e9, 
                                   melting_curve_data[1], 
                                   kind='cubic')
    
    '''
    Now we plot the entropy and volume of the liquid phase along the melting curve
    '''
    pressures = np.linspace(1.e5, 250.e9, 26)
    Sfusion = np.empty_like(pressures)
    Vfusion = np.empty_like(pressures)
    Smelt = np.empty_like(pressures)
    Smelt_model = np.empty_like(pressures)
    Vmelt = np.empty_like(pressures)
    Vmelt_model = np.empty_like(pressures)
    Tmelt_model = np.empty_like(pressures)
    
    Cp_1bar = 46.024
    S_1bar = 99.765
    Tm_1bar = 1809.

    P_inv, T_inv = burnman.tools.invariant_point([hcp, liq], [1.0, -1.0],
                                                 [fcc, liq], [1.0, -1.0],
                                                 [100.e9, 3000.])
    
    for i, P in enumerate(pressures):
        if P > P_inv:
            Fe_phase = hcp
        else:
            Fe_phase = fcc

        dP = 100. # Pa
        


        T2 = burnman.tools.equilibrium_temperature([Fe_phase, liq], [1.0, -1.0], P+dP, 1800.)
        T = burnman.tools.equilibrium_temperature([Fe_phase, liq], [1.0, -1.0], P, 1800.)
        Tmelt_model[i] = T
        dTdP = (T2-T)/dP
        
        aK_T = Fe_phase.alpha*Fe_phase.K_T
        Sfusion[i] = burnman.constants.gas_constant*np.log(2.) / (1. - aK_T*dTdP)
        Vfusion[i] = Sfusion[i]*dTdP
        
        Smelt[i] = Fe_phase.S + Sfusion[i]
        Vmelt[i] = Fe_phase.V + Vfusion[i]

        liq.set_state(P, T)
        Smelt_model[i] = liq.S
        Vmelt_model[i] = liq.V
    
        print int(P/1.e9), T, Sfusion[i], Vfusion[i]*1.e6, aK_T/1.e9, Fe_phase.S, (Fe_phase.K_T - liq.K_T)/1.e9
    


    plt.plot(pressures/1.e9, Smelt, marker='o', linestyle='None')
    plt.plot(pressures/1.e9, Smelt_model)
    plt.show()


    plt.plot(pressures/1.e9, Vmelt, marker='o', linestyle='None')
    plt.plot(pressures/1.e9, Vmelt_model)
    plt.show()


    fig1 = mpimg.imread('data/Anzellini_2013_Fe_melting.png')  # Uncomment these two lines if you want to overlay the plot on a screengrab from SLB2011
    plt.imshow(fig1, extent=[0., 230., 1200., 5200.], aspect='auto')
    #plt.plot(pressures/1.e9, melting_temperature(pressures), marker='o', linestyle='None')
    plt.plot(pressures/1.e9, Tmelt_model)
    plt.show()

    
