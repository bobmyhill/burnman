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

class liq_wuestite (burnman.Mineral):
    def __init__(self):
        formula='FeO'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Wuestite',
            'formula': formula,
            'equation_of_state': 'slbel3',
            'F_0': -242000.0 ,
            'V_0': 1.2239e-05 , 
            'K_0': 1.79e+11 , 
            'Kprime_0': 4.9 ,
            'Debye_0': 500.0 ,
            'grueneisen_0': 1.45 ,
            'q_0': 0.3 ,
            'G_0': 59000000000.0 ,
            'Gprime_0': 1.4 ,
            'eta_s_0': -0.1 ,
            'T_el': 2250., 
            'Cv_el': 1.5, 
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        burnman.Mineral.__init__(self)


if __name__ == "__main__":
    from B1_wuestite import B1_wuestite

    B1_FeO = B1_wuestite()
    liq = liq_wuestite()

    # Need Cp and S at the melting point.
    # Need to correct F_0 for wuestite
    # Fit F_0 for the liquid from the melting point
    # Find volume and thermal expansion for FeO liquid
    # Fit S and V for the melting curve

    Cp_1bar = 46.024
    S_1bar = 99.765
    Tm_1bar = 1650.
    

    melting_curve_data = burnman.tools.array_from_file('data/Fe0.94O_melting_curve.dat')
    melting_temperature = interp1d(melting_curve_data[0]*1.e9, 
                                   melting_curve_data[1], 
                                   kind='cubic')
    
    '''
    Now we plot the entropy and volume of the liquid phase along the melting curve
    '''
    pressures = np.linspace(1.e5, 75.e9, 26)
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
        dT = melting_temperature(P + dP/2.) - melting_temperature(P - dP/2.)
        dTdP = dT/dP
        T = melting_temperature(P)
        FeO_phase.set_state(P, T)
        aK_T = FeO_phase.alpha*FeO_phase.K_T
        Sfusion[i] = burnman.constants.gas_constant*np.log(2.) / (1. - aK_T*dTdP)
        Vfusion[i] = Sfusion[i]*dTdP
        
        Smelt[i] = FeO_phase.S + Sfusion[i]
        Vmelt[i] = FeO_phase.V + Vfusion[i]
        
        liq.set_state(P, T)
        Smelt_model[i] = liq.S
        Vmelt_model[i] = liq.V
    
        print int(P/1.e9), Sfusion[i], Vfusion[i]*1.e6, aK_T/1.e9, FeO_phase.S, (FeO_phase.K_T - liq.K_T)/1.e9
        #Tmelt_model[i] = burnman.tools.equilibrium_temperature([FeO_phase, liq], [1.0, -1.0], P, 1800.)
    


    plt.plot(pressures/1.e9, Smelt, marker='o', linestyle='None')
    plt.plot(pressures/1.e9, Smelt_model)
    plt.show()


    plt.plot(pressures/1.e9, Vmelt, marker='o', linestyle='None')
    plt.plot(pressures/1.e9, Vmelt_model)
    plt.show()


    fig1 = mpimg.imread('data/Fischer_Campbell_2010_FeO_melting_0_80_1500_3500.png')  # Uncomment these two lines if you want to overlay the plot on a screengrab from SLB2011
    plt.imshow(fig1, extent=[0., 80., 1500., 3500.], aspect='auto')
    #plt.plot(pressures/1.e9, melting_temperature(pressures), marker='o', linestyle='None')
    #plt.plot(pressures/1.e9, Tmelt_model)
    plt.show()
