import numpy as np
from scipy import optimize

def Cp(T, a, b, c, d):
    return a + b*T + c/T/T + d/np.sqrt(T)

def HP_convert (mineral, Cp_LT, Cp_HT, T_ref, P_ref):
    # First, let's fit the heat capacity at the new reference pressure
    # over the calibration range 
    
    mineral.set_state(P_ref, T_ref)
    
    Cps = []
    temperatures = np.linspace(Cp_LT, Cp_HT, 100)
    for T in temperatures:
        mineral.set_state(P_ref, T)
        Cps.append(mineral.C_p)

    Cp_new = optimize.curve_fit(Cp, temperatures, Cps)[0]
    a, b, c, d = Cp_new


    # Now comes the work of finding all the values at the reference pressure and temperature
    dP = 100000.
    mineral.set_state(P_ref-dP, T_ref)
    
    K0 = mineral.K_T 
    mineral.set_state(P_ref, T_ref)
    K1 = mineral.K_T
    mineral.set_state(P_ref+dP, T_ref)
    K2 = mineral.K_T
    
    grad0 = (K1 - K0)/dP
    grad1 = (K2 - K1)/dP
    
    mineral.set_state(P_ref, T_ref)
    mineral.params['T_0'] = T_ref
    mineral.params['P_0'] = P_ref
    mineral.params['H_0'] = mineral.H
    mineral.params['S_0'] = mineral.S
    mineral.params['V_0'] = mineral.V
    mineral.params['Cp'] = [a, b, c, d]
    mineral.params['K_0'] = mineral.K_T 
    mineral.params['a_0'] = mineral.alpha
    mineral.params['Kprime_0'] = (K2 - K0)/(2.*dP)
    mineral.params['Kdprime_0'] = (grad1 - grad0)/dP
