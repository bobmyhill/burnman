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


class liq_FeS (burnman.Mineral):
    def __init__(self):
        formula='FeS'
        formula = dictionarize_formula(formula)
        m = formula_mass(formula, atomic_masses)
        rho_0 = 3905. # Kaiura and Toguri, 1979 (with >200 kg/m^3 uncertainties)
        V_0 = m/rho_0
        self.params = {
            'name': 'liquid FeS',
            'formula': formula,
            'equation_of_state': 'aa',
            'P_0': 1.e5, # 1 bar
            'T_0': 1463., # melting temperature for FeS
            'S_0': 190.066, # JANAF for FeS
            'molar_mass': m, # mass
            'V_0': V_0,  # Fit to standard state data
            'E_0': -84840., # Fit to standard state data
            'K_S': 16.5e9, # Fit to standard state data
            'Kprime_S': 4.661, # ?
            'Kprime_prime_S': -0.043e-9, # ?
            'grueneisen_0': 0.87, # ?
            'grueneisen_prime': -0.130/0.055845*1.e-6, # ?
            'grueneisen_n': -1.870, # ?
            'a': [0., 0.], #[248.92*m, 289.48*m], # goes into electronic term
            'b': [0., 0.], #[0.4057*m, -1.1499*m], # goes into electronic term
            'Theta': [1747.3, 1.537], # ? (goes into potential term)
            'theta': 5000., # ? (goes into potential term)
            'lmda': [0., 0., 0.], # [302.07*m, -325.23*m, 30.45*m], # ? (goes into potential term)
            'xi_0': 36., # ? (goes into potential term)
            'F': [1., 1.],
            'n': sum(formula.values()),
            'molar_mass': m}
        burnman.Mineral.__init__(self)

'''
# for 1400 C, from Jing et al. 2014
mS = formula_mass({'S':1.}, atomic_masses)
mFe = formula_mass({'Fe': 1.}, atomic_masses)
Vps = np.array([3900., 3165., 2840., 2460., 2050.])
rhos = np.array([6910., 5220., 4400., 4070., 2050.])
Xs = np.empty_like(Vps)
for i, wtS in enumerate([0., 10., 20., 27., 36.4748]):
    wtFe = 100. - wtS
    Xs[i] = (wtS/mS) / ((wtFe/mFe) + (wtS/mS))


plt.plot(Xs, Vps, marker='o')
plt.show()
'''


liq = liq_FeS()

liq.set_state(1.e5, liq.params['T_0'])

formula = dictionarize_formula('FeS')
m = formula_mass(formula, atomic_masses)
V0 = 3909./m
V1 = 3860./m
dT = 100.
alpha =  -1./V0*(V1 - V0)/dT 
V_phi = 2050. # Approximate, from extrapolation of Jing et al., 2014. at 1673 K (i.e. not the reference temperature of 1463 K).
# In comparison, the data of Nishida et al., 2013 look weird.
# Nishida and Jing both show essentially no temperature dependence on V_phi at high S contents
VK_S = V_phi*V_phi*m # V_phi = sqrt(K_S/rho) = sqrt(K_S*V/m)
Cp = 62.551 # JANAF

grueneisen_0 = VK_S*alpha/Cp
print 'properties at Tm: modelled vs. experimental estimate'
print 'gr', liq.gr, grueneisen_0
print 'K_S', liq.K_S/1.e9, VK_S/liq.params['V_0']/1.e9
print 'alpha', liq.alpha, alpha
print 'C_p', liq.C_p, Cp
print 'V_phi', np.sqrt(liq.K_S/liq.rho), V_phi


fig1 = mpimg.imread('data/density_FeS_Kaiura_Toguri_1979_crop.png')
plt.imshow(fig1, extent=[1473., 1623., 3400., 4000.], aspect='auto')

temperatures = np.linspace(1373., 1773., 101)
rhos = np.empty_like(temperatures)
Vps = np.empty_like(temperatures)
Cps = np.empty_like(temperatures)
Cvs = np.empty_like(temperatures)

for i, T in enumerate(temperatures):
    liq.set_state(1.e5, T)
    rhos[i] = liq.rho
    Vps[i] = np.sqrt(liq.K_S/liq.rho)
    Cps[i] = liq.C_p
    Cvs[i] = liq.C_v
    
plt.plot(temperatures, rhos, marker='o')
plt.title('Densities at 1 bar')
plt.show()

plt.plot(temperatures, Vps, marker='o')
plt.title('Velocities at 1 bar')
plt.show()

plt.plot(temperatures, Cps, marker='o', label='Cp')
plt.plot(temperatures, Cvs, marker='o', label='Cv')
plt.legend(loc="lower left")
plt.title('Heat capacities at 1 bar')
plt.show()

exit()  

if __name__ == "__main__":
    from FeS_EoSes_and_FeS_VI_properties import FeS_I_new
    
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

    
