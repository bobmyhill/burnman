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

# From Hara
class liq_FeO (burnman.Mineral):
    def __init__(self):
        formula='FeO'
        formula = dictionarize_formula(formula)
        m = formula_mass(formula, atomic_masses)
        rho_0 = 4634. # 5000. #4632. # 4632 is extrapolated from Hara et al., 1988a, b, (BaO and CaO solutions) similar to Ji et al., 1997 (4700)
        V_0 = m/rho_0
        self.params = {
            'name': 'liquid FeO',
            'formula': formula,
            'equation_of_state': 'aa',
            'P_0': 1.e5, # 1 bar
            'T_0': 1650., # melting temperature for FeO
            'S_0':  195.74, # 178.78 is JANAF for FeO
            'molar_mass': m, # mass
            'V_0': V_0,  # Fit to standard state data
            'E_0': -84840., # Fit to standard state data
            'K_S': 75.e9, # Fit to standard state data, remember, gr = a*K_S*V/Cp
            'Kprime_S': 4.4, # ?
            'Kprime_prime_S': -0.040e-9, # ?
            'grueneisen_0': 1.30, # controls alpha
            'grueneisen_prime': -0.130/0.055845*1.e-6, # ?
            'grueneisen_n': -1.870, # ?
            'a': [0., 0.], # (goes into electronic term)
            'b': [0., 0.], # (goes into electronic term)
            'Theta': [1747.3, 1.537], # ? (goes into potential term)
            'theta': 2000., # ? (goes into potential term)
            'lmda': [0., 0., 0.], # [302.07*m, -325.23*m, 30.45*m], # ? (goes into potential term)
            'xi_0': 67., # ? (goes into potential term)
            'F': [1., 1.],
            'n': sum(formula.values()),
            'molar_mass': m}
        burnman.Mineral.__init__(self)

# From density estimate from Bhattacharyya and Gaskell (1996)
class liq_FeO (burnman.Mineral):
    def __init__(self):
        formula='FeO'
        formula = dictionarize_formula(formula)
        m = formula_mass(formula, atomic_masses)
        rho_0 = 4800. # 5000. #4632. # 4632 is extrapolated from Hara et al., 1988a, b, (BaO and CaO solutions) similar to Ji et al., 1997 (4700)
        V_0 = m/rho_0
        self.params = {
            'name': 'liquid FeO',
            'formula': formula,
            'equation_of_state': 'aa',
            'P_0': 1.e5, # 1 bar
            'T_0': 1650., # melting temperature for FeO
            'S_0':  188., # 178.78 is JANAF for FeO
            'molar_mass': m, # mass
            'V_0': V_0,  # Fit to standard state data
            'E_0': -84840., # Fit to standard state data
            'K_S': 82.e9, # Fit to standard state data, remember, gr = a*K_S*V/Cp
            'Kprime_S': 4.4, # ?
            'Kprime_prime_S': -0.040e-9, # ?
            'grueneisen_0': 1.30, # controls alpha
            'grueneisen_prime': -0.130/0.055845*1.e-6, # ?
            'grueneisen_n': -1.870, # ?
            'a': [0., 0.], # (goes into electronic term)
            'b': [0., 0.], # (goes into electronic term)
            'Theta': [1747.3, 1.537], # ? (goes into potential term)
            'theta': 2000., # ? (goes into potential term)
            'lmda': [0., 0., 0.], # [302.07*m, -325.23*m, 30.45*m], # ? (goes into potential term)
            'xi_0': 67., # ? (goes into potential term)
            'F': [1., 1.],
            'n': sum(formula.values()),
            'molar_mass': m}
        burnman.Mineral.__init__(self)
'''
# Density estimate from dTdP, sort-of agrees with Mori and Suzuki, 1968
class liq_FeO (burnman.Mineral):
    def __init__(self):
        formula='FeO'
        formula = dictionarize_formula(formula)
        m = formula_mass(formula, atomic_masses)
        rho_0 = 5018. # Coughlin-derived
        V_0 = m/rho_0
        self.params = {
            'name': 'liquid FeO',
            'formula': formula,
            'equation_of_state': 'aa',
            'P_0': 1.e5, # 1 bar
            'T_0': 1650., # melting temperature for FeO
            'S_0':  178.78, # 178.78 is JANAF for FeO
            'molar_mass': m, # mass
            'V_0': V_0,  # Fit to standard state data
            'E_0': -84840., # Fit to standard state data
            'K_S': 100.e9, # Fit to standard state data, remember, gr = a*K_S*V/Cp
            'Kprime_S': 4.4, # ?
            'Kprime_prime_S': -0.040e-9, # ?
            'grueneisen_0': 1.30, # controls alpha
            'grueneisen_prime': -0.130/0.055845*1.e-6, # ?
            'grueneisen_n': -1.870, # ?
            'a': [0., 0.], # (goes into electronic term)
            'b': [0., 0.], # (goes into electronic term)
            'Theta': [1747.3, 1.537], # ? (goes into potential term)
            'theta': 2000., # ? (goes into potential term)
            'lmda': [0., 0., 0.], # [302.07*m, -325.23*m, 30.45*m], # ? (goes into potential term)
            'xi_0': 67., # ? (goes into potential term)
            'F': [1., 1.],
            'n': sum(formula.values()),
            'molar_mass': m}
        burnman.Mineral.__init__(self)
'''
# H_0 = 75015 # Barin

from B1_wuestite import B1_wuestite
    
liq = liq_FeO()


liq.set_state(1.e5, 1673.)
print liq.V*1.e6, liq.rho
liq.set_state(1.e5, 1773.)
print liq.V*1.e6, liq.rho
liq.set_state(1.e5, 1873.)
print liq.V*1.e6, liq.rho


liq.set_state(1.e5, liq.params['T_0'])

B1 = B1_wuestite()
#B1 = burnman.minerals.SLB_2011.wuestite()
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

#exit()

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
Cvs_el = np.empty_like(temperatures)
Cvs_kin = np.empty_like(temperatures)
Cvs_pot = np.empty_like(temperatures)
grs = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    liq.set_state(1.e5, T)
    volumes[i] = liq.V
    Vps[i] = np.sqrt(liq.K_S/liq.rho)
    Ss[i] = liq.S
    Cps[i] = liq.C_p
    Cvs[i] = liq.C_v
    Cvs_kin[i] = liq.method._C_v_kin(liq.V, T, liq.params)
    Cvs_el[i] = liq.method._C_v_el(liq.V, T, liq.params)
    Cvs_pot[i] = liq.method._C_v_pot(liq.V, T, liq.params)
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
plt.plot(temperatures, Cvs_el, label='el')
plt.plot(temperatures, Cvs_kin, label='kin')
plt.plot(temperatures, Cvs_pot, label='pot')
plt.legend(loc='upper left')
plt.show()

  

if __name__ == "__main__":
    from B1_wuestite import B1_wuestite
    
    B1 = B1_wuestite()

    B1.set_state(1.e5, 1650.)
    liq.set_state(1.e5, 1650.)
    liq.params['E_0'] = liq.params['E_0'] - liq.gibbs + B1.gibbs
    pressures = np.linspace(1.e5, 100.e9, 41)
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
    exit()

    fig1 = mpimg.imread('data/Anzellini_2013_Fe_melting.png')  # Uncomment these two lines if you want to overlay the plot on a screengrab from SLB2011
    plt.imshow(fig1, extent=[0., 230., 1200., 5200.], aspect='auto')
    #plt.plot(pressures/1.e9, melting_temperature(pressures), marker='o', linestyle='None')
    plt.plot(pressures/1.e9, Tmelt_model)
    plt.show()

    
