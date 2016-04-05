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


class liq_FeSi (burnman.Mineral):
    def __init__(self):
        formula='FeSi'
        formula = dictionarize_formula(formula)
        m = formula_mass(formula, atomic_masses)
        rho_0 = 5060.
        V_0 = m/rho_0
        D = 7766.
        Lambda = 1146.
        self.params = {
            'name': 'liquid FeSi',
            'formula': formula,
            'equation_of_state': 'aa',
            'P_0': 1.e5, # 1 bar
            'T_0': 1683., # melting temperature
            'S_0': 183.619, # Barin
            'molar_mass': m, # mass
            'V_0': V_0,  # Fit to standard state data
            'E_0': -84840., # Fit to standard state data
            'K_S': 94.e9, # Fit to standard state data
            'Kprime_S': 4.661, # High? Williams et al., 2015
            'Kprime_prime_S': -0.043e-9, # ?
            'grueneisen_0': 2.80, # Fit to standard state data
            'grueneisen_prime': -0.130/0.055845*1.e-6, # ?
            'grueneisen_n': -1.870, # ?
            'a': [0., 0.], #[248.92*m, 289.48*m], # ? (goes into electronic term)
            'b': [0., 0.], #[0.04057*m, -0.11499*m], # ? (goes into electronic term)
            'Theta': [1747.3, 1.537], # ? (goes into potential term)
            'theta': 1000., # ? (goes into potential term)
            'lmda': [0., 0., 0.], # [302.07*m, -325.23*m, 30.45*m], # ? (goes into potential term)
            'xi_0': 65., # ? (goes into potential term)
            'F': [D/rho_0, Lambda/rho_0],
            'n': sum(formula.values()),
            'molar_mass': m}
        burnman.Mineral.__init__(self)

# H_0 = 75015 # Barin

if __name__ == "__main__":
    liq = liq_FeSi()
    
    liq.set_state(1.e5, liq.params['T_0'])
    
    formula = dictionarize_formula('FeSi')
    m = formula_mass(formula, atomic_masses)
    alpha = 1.5e-4 # Mizuno
    V_phi = 4310. # Williams et al., 2015
    VK_S = V_phi*V_phi*m # V_phi = sqrt(K_S/rho) = sqrt(K_S*V/m)
    Cp = 83.680 # Barin
    
    grueneisen_0 = VK_S*alpha/Cp
    print 'properties at Tm: modelled vs. experimental estimate'
    print 'gr', liq.gr, grueneisen_0
    print 'K_S', liq.K_S, VK_S/liq.params['V_0']
    print 'alpha', liq.alpha, alpha
    print 'C_p', liq.C_p, Cp
    print 'V_phi', np.sqrt(liq.K_S/liq.rho), V_phi
    
    # Williams
    mFe=formula_mass({'Fe': 1.}, atomic_masses)
    mNi=formula_mass({'Ni': 1.}, atomic_masses)
    mSi=formula_mass({'Si': 1.}, atomic_masses)
    
    Si_contents = np.array([0., 6., 10., 14., 20., 33.3])
    V_phi_1683 = np.array([3939., 4030., 4086., 4132., 4196., 4310.])
    mol_Si = np.empty_like(Si_contents)
    
    for i, wtSi in enumerate(Si_contents):
        mass = wtSi/mSi + 5./mNi + (95.-wtSi)/mFe
        mol_Si[i] = wtSi/mSi/mass
    
    plt.plot(mol_Si, V_phi_1683, marker='o')
    plt.show()
    
    
    # For volumes see also Dumay and Cramb (1995)
    V_Mizuno = lambda T: 0.0839305/(5051. - 0.773*(T - 1693.))
    
    temperatures = np.linspace(1683., 2273., 101)
    volumes = np.empty_like(temperatures)
    Vps = np.empty_like(temperatures)
    Cps = np.empty_like(temperatures)
    Cvs = np.empty_like(temperatures)
    Cvs_el = np.empty_like(temperatures)
    Cvs_kin = np.empty_like(temperatures)
    Cvs_pot = np.empty_like(temperatures)
    grs = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        liq.set_state(1.e5, T)
        volumes[i] = liq.V
        Vps[i] = np.sqrt(liq.K_S/liq.rho)
        Cps[i] = liq.C_p
        Cvs[i] = liq.C_v
        Cvs_kin[i] = liq.method._C_v_kin(liq.V, T, liq.params)
        Cvs_el[i] = liq.method._C_v_el(liq.V, T, liq.params)
        Cvs_pot[i] = liq.method._C_v_pot(liq.V, T, liq.params)
        grs[i] = liq.gr
        
    plt.plot(temperatures, V_Mizuno(temperatures))
    plt.plot(temperatures, volumes, marker='o')
    plt.title('Volumes')
    plt.show()
    
    
    plt.title('Grueneisen')
    plt.plot(temperatures, grs, marker='o')
    plt.show()
    
    plt.title('Bulk sound velocities')
    fig1 = mpimg.imread('figures/Fe_Si_liquid_alloy_velocities.png')
    plt.imshow(fig1, extent=[1400., 2000., 3900., 4350.], aspect='auto')
    plt.plot(temperatures, Vps, marker='o')
    plt.show()
    
    plt.title('Heat capacities')
    plt.plot(temperatures, Cps, marker='o', label='Cp')
    plt.plot(temperatures, Cvs, marker='o', label='Cv')
    plt.plot(temperatures, Cvs_el, label='el')
    plt.plot(temperatures, Cvs_kin, label='kin')
    plt.plot(temperatures, Cvs_pot, label='pot')
    plt.legend(loc='upper left')
    plt.show()
    
    
    
    from B20_FeSi import B20_FeSi
    
    B20 = B20_FeSi()

    pressures = np.linspace(1.e5, 40.e9, 101)
    temperatures = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        temperatures[i] = burnman.tools.equilibrium_temperature([B20, liq], [1.0, -1.0], P, 1800.)
        #print liq.S - B20.S


    
    fig1 = mpimg.imread('figures/FeSi_melting_curve_Lord_2010.png')
    plt.imshow(fig1, extent=[0., 160., 1600., 4200.], aspect='auto')
    plt.plot(pressures/1.e9, temperatures, linewidth=4.)
    plt.show()
    '''
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
                                   
    '''
    Now we plot the entropy and volume of the liquid phase along the melting curve
    '''
    pressures = np.linspace(1.e5, 30.e9, 31)
    Sfusion = np.empty_like(pressures)
    Vfusion = np.empty_like(pressures)
    Smelt = np.empty_like(pressures)
    Smelt_model = np.empty_like(pressures)
    Vmelt = np.empty_like(pressures)
    Vmelt_model = np.empty_like(pressures)
    Tmelt_model = np.empty_like(pressures)
    
    Cp_1bar = 83.68
    S_1bar = 183.619
    Tm_1bar = 1683.

    
    for i, P in enumerate(pressures):
        dP = 100. # Pa

        T2 = burnman.tools.equilibrium_temperature([B20, liq], [1.0, -1.0], P+dP, 1800.)
        T = burnman.tools.equilibrium_temperature([B20, liq], [1.0, -1.0], P, 1800.)
        Tmelt_model[i] = T
        dTdP = (T2-T)/dP
        
        aK_T = B20.alpha*B20.K_T
        Sfusion[i] = burnman.constants.gas_constant*np.log(2.) / (1. - aK_T*dTdP)
        Vfusion[i] = Sfusion[i]*dTdP
        
        Smelt[i] = B20.S + Sfusion[i]
        Vmelt[i] = B20.V + Vfusion[i]

        liq.set_state(P, T)
        Smelt_model[i] = liq.S
        Vmelt_model[i] = liq.V
    
        print int(P/1.e9), T, Sfusion[i], Vfusion[i]*1.e6, aK_T/1.e9, B20.S, (B20.K_T - liq.K_T)/1.e9
    


    plt.plot(pressures/1.e9, Smelt, marker='o', linestyle='None')
    plt.plot(pressures/1.e9, Smelt_model)
    plt.show()


    plt.plot(pressures/1.e9, Vmelt, marker='o', linestyle='None')
    plt.plot(pressures/1.e9, Vmelt_model)
    plt.show()
    exit()

    fig1 = mpimg.imread('data/Anzellini_2013_Fe_melting.png')  # Uncomment these two lines if you want to overlay the plot on a screengrab from SLB2011
    plt.imshow(fig1, extent=[0., 230., 1200., 5200.], aspect='auto')
    #plt.plot(pressures/1.e9, melting_temperature(pressures), marker='o', linestyle='None')
    plt.plot(pressures/1.e9, Tmelt_model)
    plt.show()

    
