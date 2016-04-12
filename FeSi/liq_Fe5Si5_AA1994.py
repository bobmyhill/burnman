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


class liq_Fe5Si5 (burnman.Mineral):
    def __init__(self):
        formula='Fe0.5Si0.5'
        formula = dictionarize_formula(formula)
        m = formula_mass(formula, atomic_masses)
        rho_0 = 5120. # 5060. # Mizuno et al., accounting for a small difference in Tmelt (1683 vs 1693), see also very similar result by Dumay and Cramb (1995). Kawai et al get ~5090 at 1723 K (so at 1683 K could be 5120 kg/m^3)
        V_0 = m/rho_0
        self.params = {
            'name': 'liquid Fe0.5Si0.5',
            'formula': formula,
            'equation_of_state': 'aamod',
            'P_0': 1.e5, # 1 bar
            'T_0': 1683., # melting temperature
            'S_0': 183.619/2., # Entropy at melting point, Barin
            'molar_mass': m, # mass
            'V_0': V_0,  # See rho_0, above
            'E_0': 68247./2., # Energy at melting point
            'K_S': 94.5e9, # Fit to Williams et al., 2015 (extrapolation)
            'Kprime_S': 4.66, # High? Williams et al., 2015
            'Kprime_prime_S': -0.035e-9, # To fit high pressure melting curve (Lord et al., 2010)
            'grueneisen_0': 2.8, # To fit alpha (Mizuno et al.)
            'grueneisen_prime': -2.*0.130/0.055845*1.e-6, # ?
            'grueneisen_n': -1.870, # ?
            'T_el': 7000.,
            'Cv_el': 2.7/2.,
            'theta':  150., #2000., # ? To fit C_p (goes into potential term)
            'xi_0': 200./2., 
            'n': sum(formula.values()),
            'molar_mass': m}
        burnman.Mineral.__init__(self)


if __name__ == "__main__":
    from liq_FeSi_AA1994 import liq_FeSi
    from B20_FeSi import B20_FeSi
    from B2_FeSi import B2_FeSi

    liq = liq_Fe5Si5()
    liq_full = liq_FeSi()    
    B20 = B20_FeSi()
    B2 = B2_FeSi() # high pressure phase

    liq.set_state(20.e9, 2000.)
    liq_full.set_state(20.e9, 2000.)

    print 'Check Fe0.5Si0.5 = 0.5 FeSi'
    print 2.*liq.gibbs, liq_full.gibbs
    
    temperatures = np.linspace(500., 4000., 21)
    Ss = np.empty_like(temperatures)
    Ss2 = np.empty_like(temperatures)
    for P in [10.e9, 50.e9, 100.e9, 200.e9]:
        for i, T in enumerate(temperatures):
            B2.set_state(P, T)
            liq.set_state(P, T)
            Ss[i] = B2.S
            Ss2[i] = 2.*liq.S
        plt.plot(temperatures, Ss2 - Ss, label=str(P/1.e9)+' GPa')
    plt.legend(loc='upper right')
    plt.show()
    
    
    liq.set_state(1.e5, liq.params['T_0'])
    B20.set_state(1.e5, liq.params['T_0'])

    print 'DeltaV at 1 bar:', liq.V - B20.V, B20.V, liq.V
    print 'DeltaS at 1 bar:', liq.S - B20.S, B20.S, liq.S
    
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
    

    P_B20_B2, T_B20_B2 = burnman.tools.invariant_point([B20, B2], [1., -1.],
                                                       [B2, liq], [1., -1.],
                                                       [23.e9, 2000.])


    temperatures = np.linspace(300., T_B20_B2, 21)
    pressures = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        print T
        pressures[i] = burnman.tools.equilibrium_pressure([B20, B2], [1.0, -1.0], T, 20.e9)
    plt.plot(pressures/1.e9, temperatures, linewidth=4.)
      
    pressures = np.linspace(1.e5, P_B20_B2, 21)
    temperatures = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        print P
        temperatures[i] = burnman.tools.equilibrium_temperature([B20, liq], [1.0, -1.0], P, 1800.)
    plt.plot(pressures/1.e9, temperatures, linewidth=4.)

    pressures = np.linspace(P_B20_B2, 160.e9, 21)
    temperatures = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        print P
        temperatures[i] = burnman.tools.equilibrium_temperature([B2, liq], [1.0, -1.0], P, 2500.)
    plt.plot(pressures/1.e9, temperatures, linewidth=4.)

    
    fig1 = mpimg.imread('figures/FeSi_melting_curve_Lord_2010.png')
    plt.imshow(fig1, extent=[0., 160., 1600., 4200.], aspect='auto')
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
    pressures = np.linspace(30.e9, 150.e9, 31)
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

        T2 = burnman.tools.equilibrium_temperature([B2, liq], [1.0, -1.0], P+dP, 1800.)
        T = burnman.tools.equilibrium_temperature([B2, liq], [1.0, -1.0], P, 1800.)
        Tmelt_model[i] = T
        dTdP = (T2-T)/dP
        
        aK_T = B2.alpha*B2.K_T
        Sfusion[i] = burnman.constants.gas_constant*np.log(2.) / (1. - aK_T*dTdP)
        Vfusion[i] = Sfusion[i]*dTdP
        
        Smelt[i] = B2.S + Sfusion[i]
        Vmelt[i] = B2.V + Vfusion[i]

        liq.set_state(P, T)
        Smelt_model[i] = liq.S
        Vmelt_model[i] = liq.V
    
        print int(P/1.e9), T, Sfusion[i], Vfusion[i]*1.e6, aK_T/1.e9, B2.S, (B2.K_T - liq.K_T)/1.e9
    


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

    
