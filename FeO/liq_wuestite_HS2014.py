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

class FeO_liquid(burnman.Mineral):
    def __init__(self):
        self.params = {'xi': 0.4, 'Tel_0': 2000.0, 'equation_of_state': 'dks_l', 'zeta_0': 0.0024257065617444393, 'T_0': 8000.0, 'O_f': 4, 'el_V_0': 1.81044909456887e-05, 'a': [[-121067.64971185905, -146030.3431653101, -333418.70122137183], [327392.3159095949, 734700.3123169115, -2152192.059275142], [6090297.376045127, -1303771.1057723137, 2985350.1952913064], [-9726174.025790948, -587092.0412079562, 0.0], [90383884.86072995, 0.0, 0.0]], 'V_0': 1.81044909456887e-05, 'name': 'FeO_liquid', 'molar_mass': 0.071844, 'm': 0.63, 'O_theta': 2, 'eta': -1.0, 'formula': {'Fe': 1.0, 'O': 1.0}}
        burnman.Mineral.__init__(self)

# H_0 = 75015 # Barin

if __name__ == "__main__":
    from B1_wuestite import B1_wuestite

    #B1 = burnman.minerals.HP_2011_ds62.fper()
    B1 = B1_wuestite()    
    liq = FeO_liquid()

    liq.set_state(1.e5, 1650.)
    B1.set_state(1.e5, 1650.)
    print liq.gibbs - B1.gibbs
    print liq.S - B1.S
    print (liq.V - B1.V)/(liq.S - B1.S) * 1.e9, 'K/GPa'

    # Find heat capacities
    temperatures = np.linspace(1000., 15000., 101)
    Cvs = np.empty_like(temperatures)
    m = 0.055845
    rhos = np.empty_like(temperatures)
    densities = [5.e3,10.e3, 15.e3]
    for rho in densities:
        V = m/rho
        for i, T in enumerate(temperatures):
            Cvs[i] = liq.method.heat_capacity_v(0., T, V, liq.params)/burnman.constants.gas_constant
            #Cvs[i] = liq.method._C_v_el(V, T, liq.params)/burnman.constants.gas_constant

        plt.plot(temperatures, Cvs)

    plt.ylim(0., 6.)
    plt.show()
    
    pressures = np.linspace(1.e5, 120.e9, 21)
    temperatures = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        temperatures[i] = burnman.tools.equilibrium_temperature([B1, liq], [1.0, -1.0], P, 1800.)
        print P/1.e9, temperatures[i]

    fig1 = mpimg.imread('figures/FeO_melting_curve.png')
    plt.imshow(fig1, extent=[0., 120., 1000., 6000.], aspect='auto')
    plt.plot(pressures/1.e9, temperatures, linewidth=4.)
    #plt.xlim(0., 20.)
    #plt.ylim(1500., 4000.)
    plt.show()
    #exit()
    
    for P in [1.e9, 50.e9, 100.e9]:
        temperatures = np.linspace(100., 4000., 101)
        Ss = np.empty_like(temperatures)
        Ss2 = np.empty_like(temperatures)
        for i, T in enumerate(temperatures):
            liq.set_state(P, T)
            B1.set_state(P, T)
            Ss[i] = liq.S
            Ss2[i] = B1.S

        plt.plot(temperatures, Ss-Ss2, label=str(P/1.e9)+' GPa')
        #plt.plot(temperatures, Ss2, label=str(P/1.e9)+' GPa, B1')
    plt.legend(loc='lower right')
    plt.show()
    
    liq.set_state(1.e5, 1673.)
    print liq.V*1.e6, liq.rho
    liq.set_state(1.e5, 1773.)
    print liq.V*1.e6, liq.rho
    liq.set_state(1.e5, 1873.)
    print liq.V*1.e6, liq.rho

    liq.set_state(1.e5, liq.params['T_0'])
    B1.set_state(1.e5, liq.params['T_0'])
    
    dTdP_Lindsley_1966 = 70.
    Sfusion_Coughlin_1951 = 19.5
    V_CL =  (B1.V + (dTdP_Lindsley_1966/1.e9*Sfusion_Coughlin_1951))
    print 'V, rho from Coughlin/Lindsley:', V_CL*1.e6, B1.params['molar_mass']/V_CL
    print 'dTdP from Coughlin/Hara:', -(B1.V - B1.params['molar_mass']/4634.)/Sfusion_Coughlin_1951*1.e9
    print 'S from Lindsley/Hara:', B1.S-(B1.V - B1.params['molar_mass']/4634.)/dTdP_Lindsley_1966*1.e9
    
    
    print 'dT/dP melting (model, K/GPa):', (liq.V - B1.V)/(liq.S - B1.S)*1.e9
    
    print 'Constraints are density of FeO liquid (extrapolated), entropy of liquid, corrected (Coughlin et al., 1951),'
    print 'Density of FeO solid (thermal expansion from Hazen and Jeanloz), entropy, corrected (Coughlin et al., 1951)'
    
    
    # Density data from Hara, Irie, Gaskell and Ogino, 1988
    temperatures_H = [1673., 1773., 1873.]
    rhos_H = [4.624e3, 4.588e3, 4.548e3]
    
    
    formula = dictionarize_formula('FeO')
    m = formula_mass(formula, atomic_masses)
    V = 1.551e-5
    alpha = 1./liq.params['V_0'] * (m/rhos_H[1] - m/rhos_H[0]) / (temperatures_H[1] - temperatures_H[0]) # ?
    Cp = 68.199 # Barin
    
    V_phi = 0. # ?
    VK_S = V_phi*V_phi*m # V_phi = sqrt(K_S/rho) = sqrt(K_S*V/m)
    
    
    grueneisen_0 = VK_S*alpha/Cp
    print 'properties at Tm: modelled vs. experimental estimate'
    print 'V', liq.V, V
    print 'alpha', liq.alpha, alpha
    print 'C_p', liq.C_p, Cp
    print 'K_S', liq.K_S, VK_S/liq.params['V_0']
    print 'gr', liq.gr, grueneisen_0
    
    print 'V_phi', np.sqrt(liq.K_S/liq.rho), V_phi


    
    temperatures = np.linspace(1650., 1900., 21)
    rhos = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        liq.set_state(1.e5, T)
        rhos[i] = liq.rho
        
    plt.plot(temperatures_H, rhos_H, marker='o', linestyle='None')
    plt.plot(temperatures, rhos)
    plt.show()
    
    
    
    B1_temperatures = np.linspace(300., 1650., 101)
    B1_Ss = np.empty_like(B1_temperatures)
    for i, T in enumerate(B1_temperatures):
        B1.set_state(1.e5, T)
        B1_Ss[i] = B1.S
    
    temperatures = np.linspace(1650., 2250., 101)
    volumes = np.empty_like(temperatures)
    Vps = np.empty_like(temperatures)
    Ss = np.empty_like(temperatures)
    Cps = np.empty_like(temperatures)
    Cvs = np.empty_like(temperatures)
    grs = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        liq.set_state(1.e5, T)
        volumes[i] = liq.V
        Vps[i] = np.sqrt(liq.K_S/liq.rho)
        Ss[i] = liq.S
        Cps[i] = liq.C_p
        Cvs[i] = liq.C_v
        grs[i] = liq.gr
        
    plt.plot(temperatures, volumes, marker='o')
    plt.title('Volumes')
    plt.show()
    
    T_C, Hdiff, Sdiff = np.loadtxt(unpack='True', fname="data/Coughlin_KB_1951_FeO_HS_solid_liquid.dat")
    B1.set_state(1.e5, 298.15)
    S0 = B1.S
    DS_C = 4.184*Sdiff*(2./1.947) + S0
    
    plt.title('Entropies')
    plt.plot(T_C, DS_C, marker='o', linestyle='None')
    plt.plot(B1_temperatures, B1_Ss)
    plt.plot(temperatures, Ss)
    plt.show()
    
    
    
    
    plt.title('Grueneisen')
    plt.plot(temperatures, grs, marker='o')
    plt.show()
    
    plt.title('Bulk sound velocities')
    plt.plot(temperatures, Vps, marker='o')
    plt.show()
    
    plt.title('Heat capacities')
    plt.plot(temperatures, Cps, marker='o', label='Cp')
    plt.plot(temperatures, Cvs, marker='o', label='Cv')
    plt.legend(loc='upper left')
    plt.show()

    
                                   
    '''
    Now we plot the entropy and volume of the liquid phase along the melting curve
    '''
    pressures = np.linspace(1.e5, 100.e9, 26)
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

        T2 = burnman.tools.equilibrium_temperature([B1, liq], [1.0, -1.0], P+dP, 1800.)
        T = burnman.tools.equilibrium_temperature([B1, liq], [1.0, -1.0], P, 1800.)
        Tmelt_model[i] = T
        dTdP = (T2-T)/dP
        
        aK_T = B1.alpha*B1.K_T
        Sfusion[i] = burnman.constants.gas_constant*np.log(2.) / (1. - aK_T*dTdP)
        Vfusion[i] = Sfusion[i]*dTdP
        
        Smelt[i] = B1.S + Sfusion[i]
        Vmelt[i] = B1.V + Vfusion[i]

        liq.set_state(P, T)
        Smelt_model[i] = liq.S
        Vmelt_model[i] = liq.V
    
        print int(P/1.e9), T, Sfusion[i], Vfusion[i]*1.e6, aK_T/1.e9, B1.S, (B1.K_T - liq.K_T)/1.e9
    


    plt.plot(pressures/1.e9, Smelt, marker='o', linestyle='None')
    plt.plot(pressures/1.e9, Smelt_model)
    plt.show()


    plt.plot(pressures/1.e9, Vmelt, marker='o', linestyle='None')
    plt.plot(pressures/1.e9, Vmelt_model)
    plt.show()
    
