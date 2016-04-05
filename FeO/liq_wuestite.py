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

# For high melting temperatures, see
# Kato and Ringwood, 1989
# Knittle and Jeanloz, 1986; 1991
# Tallon rule of thumb

# For intermediate, see
# Seagle et al., 2008
# Shen et al., 1993

# For low temperature, see
# Fischer and Campbell, 2010
# Shen et al, 1993
# Ringwood and Hibberson, 1990

'''
class liq_wuestite (burnman.Mineral):
    def __init__(self):
        formula='FeO'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Wuestite liquid',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': -257587 + 1481.,
            'V_0': 1.299e-05 , 
            'K_0': 1.20e+11 , 
            'Kprime_0': 4.0 ,
            'Debye_0': 251.0 ,
            'grueneisen_0': 1.4 ,
            'q_0': 2.2 ,
            'T_el': 6000., 
            'Cv_el': 1.5, 
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        burnman.Mineral.__init__(self)
'''
'''
class liq_wuestite (burnman.Mineral):
    def __init__(self):
        formula='FeO'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Wuestite liquid',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': -257587 + 1481.,
            'V_0': 1.302e-05 , 
            'K_0': 1.22e+11 , 
            'Kprime_0': 4. ,
            'Debye_0': 250.0 ,
            'grueneisen_0': 1.4 ,
            'q_0': 2.2 ,
            'T_el': 6000., 
            'Cv_el': 1.5, 
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        burnman.Mineral.__init__(self)
'''
'''

class liq_wuestite (burnman.Mineral):
    def __init__(self):
        formula='FeO'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Wuestite liquid',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': -257587 + 1611. ,
            'V_0': 1.305e-05 , 
            'K_0': 1.25e+11 , 
            'Kprime_0': 4.0 ,
            'Debye_0': 249. ,
            'grueneisen_0': 1.4 ,
            'q_0': 2.2 ,
            'T_el': 6000., 
            'Cv_el': 1.5, 
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        burnman.Mineral.__init__(self)
'''

if __name__ == "__main__":
    from B1_wuestite import B1_wuestite

    B1_FeO = B1_wuestite()
    liq = liq_wuestite()

    # First, let's plot Cp and S for the liquid
    temperatures = np.linspace(1650., 2650., 101)
    B1_FeO.set_state(1.e5, 1650.)
    liq.set_state(1.e5, 1650.)

    dTdP0 = 67.e-9
    print liq.C_p, 'Cp should be 68.199' 
    print liq.S - B1_FeO.S, 'Sdiff should be 14.6'
    print liq.V, 'liq V should be', (liq.S - B1_FeO.S)*dTdP0 + B1_FeO.V
    print liq.gibbs - B1_FeO.gibbs, 'gibbs diff should be 0.'


    # Millot et al., 2009; density almost constant between 1865-2155 K, FeO1.05-1.35
    # 4.35 gm/cm^3
    # 4350 kg/m^3
    V_Millot_melt = 0.071844/4350
    print B1_FeO.V, V_Millot_melt, (V_Millot_melt - B1_FeO.V)/14.6*1.e9
    print (1. - 4350/(0.071844/B1_FeO.V))*100., '% density reduction'

    
    Cps = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        liq.set_state(1.e5, T)
        Cps[i] = liq.C_p

    plt.plot(temperatures, Cps)
    plt.show()

    # Need to correct F_0 for wuestite
    # Fit F_0 for the liquid from the melting point
    # Find volume and thermal expansion for FeO liquid
    # Fit S and V for the melting curve

    Cp_1bar = 68.199
    S_1bar = 171.990
    Tm_1bar = 1650.
    

    '''
    Now we plot the entropy and volume of the liquid phase along the melting curve
    '''
    pressures = np.linspace(1.e5, 120.e9, 25)
    Sfusion = np.empty_like(pressures)
    Vfusion = np.empty_like(pressures)
    Smelt = np.empty_like(pressures)
    Smelt_model = np.empty_like(pressures)
    Vmelt = np.empty_like(pressures)
    Vmelt_model = np.empty_like(pressures)
    Tmelt_model = np.empty_like(pressures)
    

    for i, P in enumerate(pressures):
        FeO_phase = B1_FeO

        dP = 100. # Pa
        Tmelt_model[i] = burnman.tools.equilibrium_temperature([FeO_phase, liq], [1.0, -1.0], P, 1800.)
        T = Tmelt_model[i]
        T2 = burnman.tools.equilibrium_temperature([FeO_phase, liq], [1.0, -1.0], P+dP, 1800.)
        dTdP =  (T2 - T)/dP
        
        FeO_phase.set_state(P, T)
        aK_T = FeO_phase.alpha*FeO_phase.K_T
        Sfusion[i] = burnman.constants.gas_constant*np.log(2.) / (1. - aK_T*dTdP)
        Vfusion[i] = Sfusion[i]*dTdP
        
        Smelt[i] = FeO_phase.S + Sfusion[i]
        Vmelt[i] = FeO_phase.V + Vfusion[i]
        
        liq.set_state(P, T)
        Smelt_model[i] = liq.S
        Vmelt_model[i] = liq.V
    
        print int(P/1.e9), T, Sfusion[i], Vfusion[i]*1.e6, aK_T/1.e9, FeO_phase.S, (FeO_phase.K_T - liq.K_T)/1.e9
    


    plt.plot(pressures/1.e9, Smelt, marker='o', linestyle='None')
    plt.plot(pressures/1.e9, Smelt_model)
    plt.show()


    burnman.tools.array_to_file(['Pressures (GPa)', 'Entropy (J/K/mol)'], [pressures/1.e9, Smelt], 'output/Smelt_Tallon_prediction.dat')
    burnman.tools.array_to_file(['Pressures (GPa)', 'Entropy (J/K/mol)'], [pressures/1.e9, Smelt_model], 'output/Smelt_model.dat')
    
    plt.plot(pressures/1.e9, Vmelt, marker='o', linestyle='None')
    plt.plot(pressures/1.e9, Vmelt_model)
    plt.show()

    burnman.tools.array_to_file(['Pressures (GPa)', 'Volume (cm^3/mol)'], [pressures/1.e9, Vmelt*1.e6], 'output/Vmelt_Tallon_prediction.dat')
    burnman.tools.array_to_file(['Pressures (GPa)', 'Volume (cm^3/mol)'], [pressures/1.e9, Vmelt_model*1.e6], 'output/Vmelt_model.dat')

    
    burnman.tools.array_to_file(['Pressures (GPa)', 'Temperatures (K)'], [pressures/1.e9, Tmelt_model], 'output/Tmelt_model.dat')

    
    fig1 = mpimg.imread('data/Fischer_Campbell_2010_FeO_melting_0_80_1500_3500.png')  # Uncomment these two lines if you want to overlay the plot on a screengrab from SLB2011
    plt.imshow(fig1, extent=[0., 80., 1500., 3500.], aspect='auto')


    KJ1991_data = burnman.tools.array_from_file('data/Knittle_Jeanloz_1991_FeO_melting.dat')
    plt.plot(KJ1991_data[0], KJ1991_data[1], marker='o', linestyle='None', label='Lindsley (1966)') 
    KJ1991_data = burnman.tools.array_from_file('data/Knittle_Jeanloz_1991_FeO_melting_solid.dat')
    plt.plot(KJ1991_data[0], KJ1991_data[1], marker='o', linestyle='None', label='Knittle and Jeanloz (1991; solid)') 
    KJ1991_data = burnman.tools.array_from_file('data/Knittle_Jeanloz_1991_FeO_melting_liquid.dat')
    plt.plot(KJ1991_data[0], KJ1991_data[1], marker='o', linestyle='None', label='Knittle and Jeanloz (1991; melt)')
    
    solid = burnman.tools.array_from_file('data/Ozawa_2011_FeO_B1.dat')
    plt.plot(solid[0], solid[1], marker='o', linestyle='None', label='Ozawa et al (2011; B1)')
    solid = burnman.tools.array_from_file('data/Ozawa_2011_FeO_B2.dat')
    plt.plot(solid[0], solid[1], marker='o', linestyle='None', label='Ozawa et al (2011; B2)')
    solid = burnman.tools.array_from_file('data/Ozawa_2011_FeO_B8.dat')
    plt.plot(solid[0], solid[1], marker='o', linestyle='None', label='Ozawa et al (2011; B8)')

    
    plt.plot(pressures/1.e9, Tmelt_model)
    plt.legend(loc="lower right")
    plt.show()
