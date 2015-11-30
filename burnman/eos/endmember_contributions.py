import numpy as np
from burnman.constants import gas_constant

def _Landau_excesses(pressure, temperature, params):
    Tc = params['Tc_0'] + params['V_D']*pressure/params['S_D']
    if temperature < Tc:
        # Wolfram input to check partial differentials
        # x = T, y = P, a = S, c = Tc0, d = V
        # D[a ((-c + x - (d y)/a) (1 - x/(c + (d y)/a))^2 + c (1 - x/(c + (d y)/(a)))^6)/3., x]
        # Note mistake in Stixrude and Lithgow Bertelloni (2011) for the Vex term (equation 32)
        Q = np.power(1. - temperature/Tc, 0.25)
        G = params['S_D']*((temperature - Tc)*Q*Q - params['Tc_0']*np.power(Q, 6.)/3.)
        dGdT = params['S_D']*(-2.*params['Tc_0']*np.power(Q, 5.)/Tc + Q*Q - 2.*(temperature - Tc)*Q / Tc)
        dGdP = -params['V_D']*Q*(Q - temperature/Tc/Tc*(2.*Tc*np.power(Q, 4.) + 2.*(temperature - Tc)))
        d2GdT2 = params['S_D']*(10.*params['Tc_0']*np.power(Q, 4.)/Tc/Tc - 4.*Q/Tc + 2.*(temperature - Tc)/Tc/Tc)
        d2GdP2 = 2.*params['V_D']*params['V_D']*temperature / (params['S_D']*Tc*Tc) \
            * (5.*params['Tc_0']*temperature*np.power(Q, 4.)/Tc/Tc + temperature*(temperature - Tc)/Tc/Tc \
                   - 2.*params['Tc_0']*np.power(Q, 5.)/Tc - 2.*Q - 2.*Q*(temperature - Tc)/Tc)
        d2GdPdT = (2.*params['V_D']/Tc/Tc)*(params['Tc_0']*
                                            (np.power(Q, 5.) - 5.*temperature/Tc*np.power(Q, 4.)) 
                                            + 3.*temperature*Q)
    else:
        Q = 0.
        G = 0.
        dGdT = 0.
        dGdP = 0.
        d2GdT2 = 0.
        d2GdP2 = 0.
        d2GdPdT = 0.
    return G, dGdT, dGdP, d2GdT2, d2GdP2, d2GdPdT

def landau(mineral):
    """
    Applies a tricritical Landau correction to the properties 
    of an endmember which undergoes a displacive phase transition. 
    This correction is done relative to the totally disordered 
    state, and therefore differs from the implementation 
    of Holland and Powell (1998).
    This implementation is preferred, as the heat capacity 
    curve must be fit to the disordered state anyway.
    The Holland and Powell implementation may be accessed 
    from the function landau_HP
    """
    G, dGdT, dGdP, d2GdT2, d2GdP2, d2GdPdT = _Landau_excesses(mineral.pressure, mineral.temperature, mineral.landau)

    # Gibbs
    mineral.gibbs = mineral.gibbs + G

    # Second derivatives first
    mineral.C_p = - mineral.temperature * ((- mineral.C_p / mineral.temperature) + d2GdT2) # -T*d2G/dT2
    mineral.K_T = - (mineral.V + dGdP) / ((mineral.V / mineral.K_T) + d2GdP2) # - dGdP / (d2G/dP2)
    mineral.alpha = ((mineral.alpha*mineral.V) + d2GdPdT) / (mineral.V + dGdP) # d2GdPdT / dGdP

    # Now first derivatives 
    mineral.S = mineral.S - dGdT # dGdT
    mineral.V = mineral.V + dGdP # dGdP
    mineral.H = mineral.gibbs + mineral.temperature*mineral.S # H = G + TS

    # Now volume derivatives
    mineral.helmholtz = mineral.gibbs - mineral.pressure*mineral.V
    mineral.C_v = mineral.C_p - mineral.V*mineral.temperature*mineral.alpha*mineral.alpha*mineral.K_T
    mineral.gr = mineral.alpha*mineral.K_T*mineral.V/mineral.C_v
    mineral.K_S = mineral.K_T*mineral.C_p/mineral.C_v

    print "WARNING: Cp doesn't peak with this landau function! Problem with differentiation?"
    return None

def landau_HP(mineral):
    """
    Applies a tricritical Landau correction to the properties 
    of an endmember which undergoes a displacive phase transition. 
    This correction is done relative to the standard state, as per
    Holland and Powell (1998).

    Note that this formalism is rather inconsistent, as it predicts that 
    the order parameter can be greater than one...
    """
    params = mineral.landau_HP
    P = mineral.pressure
    T = mineral.temperature
    P_0 = mineral.params['P_0']
    T_0 = mineral.params['T_0']

    if T_0 < params['Tc_0']:
        Q_0 = np.power((params['Tc_0'] - T_0)/params['Tc_0'], 0.25)
    else:
        Q_0 = 0.

    Tc = params['Tc_0'] + params['V_D']*(P-P_0)/params['S_D']
    if T < Tc:
        Q = np.power((Tc - T)/params['Tc_0'], 0.25)
    else:
        Q = 0.

    # Gibbs
    G = params['Tc_0']*params['S_D']*(Q_0*Q_0 - np.power(Q_0, 6.)/3.) \
        - params['S_D']*(Tc*Q*Q - params['Tc_0']*np.power(Q, 6.)/3.) \
        - T*params['S_D']*(Q_0*Q_0 - Q*Q) + (P-P_0)*params['V_D']*Q_0*Q_0

    dGdT = params['S_D']*(Q*Q - Q_0*Q_0)
    dGdP = -params['V_D']*(Q*Q - Q_0*Q_0)

    if Q > 1.e-12:
        d2GdT2 = -params['S_D']/(2.*params['Tc_0']*Q*Q)
        d2GdP2 = -params['V_D']*params['V_D']/(2.*params['S_D']*params['Tc_0']*Q*Q)
        d2GdPdT = -params['V_D']/(2.*params['Tc_0']*Q*Q)
    else:
        d2GdT2 = 0.
        d2GdP2 = 0.
        d2GdPdT = 0.



    mineral.gibbs = mineral.gibbs + G
    # Second derivatives first
    mineral.C_p = - mineral.temperature * ((- mineral.C_p / mineral.temperature) + d2GdT2) # -T*d2G/dT2
    mineral.K_T = - (mineral.V + dGdP) / ((mineral.V / mineral.K_T) + d2GdP2) # - dGdP / (d2G/dP2)
    mineral.alpha = ((mineral.alpha*mineral.V) + d2GdPdT) / (mineral.V + dGdP) # d2GdPdT / dGdP

    # Now first derivatives 
    mineral.S = mineral.S - dGdT # dGdT
    mineral.V = mineral.V + dGdP # dGdP
    mineral.H = mineral.gibbs + mineral.temperature*mineral.S # H = G + TS

    # Now volume derivatives
    mineral.helmholtz = mineral.gibbs - mineral.pressure*mineral.V
    mineral.C_v = mineral.C_p - mineral.V*mineral.temperature*mineral.alpha*mineral.alpha*mineral.K_T
    mineral.gr = mineral.alpha*mineral.K_T*mineral.V/mineral.C_v
    mineral.K_S = mineral.K_T*mineral.C_p/mineral.C_v

    return None

def DQF(mineral):
    """
    Applies a 'Darken's quadratic formalism' correction 
    to the thermodynamic properties of a mineral endmember.
    This correction is linear in P and T, and therefore 
    corresponds to a constant volume and entropy correction.
    """

    G = mineral.dqf['H'] \
        - (mineral.temperature - mineral.params['T_0'])*mineral.dqf['S'] \
        + (mineral.pressure - mineral.params['P_0'])*mineral.dqf['V']
    dGdT = -mineral.dqf['S']
    dGdP = mineral.dqf['V']
    d2GdT2 = 0.
    d2GdP2 = 0.
    d2GdPdT = 0.

    # Gibbs
    mineral.gibbs = mineral.gibbs + G

    # Second derivatives first
    mineral.C_p = - temperature * ((- mineral.C_p / T) + d2G/dT2) # -T*d2G/dT2
    mineral.K_T = - (mineral.V + dGdP) / ((mineral.V / mineral.K_T) + d2GdP2) # - dGdP / (d2G/dP2)
    mineral.alpha = ((mineral.alpha*mineral.V) + d2GdPdT) / (mineral.V + dGdP) # d2GdPdT / dGdP

    # Now first derivatives
    mineral.H = mineral.gibbs + temperature*mineral.S # H = G + TS 
    mineral.S = mineral.S - dGdT # dGdT
    mineral.V = mineral.V + dGdP # dGdP

    # Now volume derivatives
    mineral.helmholtz = mineral.gibbs - pressure*mineral.V
    mineral.C_v = mineral.C_p - mineral.V*temperature*mineral.alpha*mineral.alpha*mineral.K_T
    mineral.gr = mineral.alpha*mineral.K_T*mineral.V/mineral.C_v
    mineral.K_S = mineral.K_T*mineral.C_p/mineral.C_v

    return None

'''
def magnetic(mineral):
    """
    Returns the magnetic contribution to the Gibbs free energy [J/mol]
    Expressions are those used by Chin, Hertzman and Sundman (1987)
    as reported in Sundman in the Journal of Phase Equilibria (1991)
    """
    
    structural_parameter=mineral.magnetic['magnetic_structural_parameter']
    tau=temperature/(mineral.magnetic['curie_temperature'][0] + pressure*mineral.magnetic['curie_temperature'][1])
    magnetic_moment=mineral.magnetic['magnetic_moment'][0] + pressure*mineral.magnetic['magnetic_moment'][1]

    A = (518./1125.) + (11692./15975.)*((1./structural_parameter) - 1.)
    if tau < 1: 
        f=1.-(1./A)*(79./(140.*structural_parameter*tau) + (474./497.)*(1./structural_parameter - 1.)*(np.power(tau, 3.)/6. + np.power(tau, 9.)/135. + np.power(tau, 15.)/600.))
    else:
        f=-(1./A)*(np.power(tau,-5)/10. + np.power(tau,-15)/315. + np.power(tau, -25)/1500.)


    G = gas_constant*temperature*np.log(magnetic_moment + 1.)*f
    dGdT = 
    dGdP = 
    d2GdT2 = 
    d2GdP2 = 
    d2GdPdT = 

    # Gibbs
    mineral.gibbs = mineral.gibbs + G

    # Second derivatives first
    mineral.C_p = - temperature * ((- mineral.C_p / T) + d2G/dT2) # -T*d2G/dT2
    mineral.K_T = - (mineral.V + dGdP) / ((mineral.V / mineral.K_T) + d2GdP2) # - dGdP / (d2G/dP2)
    mineral.alpha = ((mineral.alpha*mineral.V) + d2GdPdT) / (mineral.V + dGdP) # d2GdPdT / dGdP

    # Now first derivatives
    mineral.H = mineral.gibbs + temperature*mineral.S # H = G + TS 
    mineral.S = mineral.S - dGdT # dGdT
    mineral.V = mineral.V + dGdP # dGdP

    # Now volume derivatives
    mineral.helmholtz = mineral.gibbs - pressure*mineral.V
    mineral.C_v = mineral.C_p - mineral.V*temperature*mineral.alpha*mineral.alpha*mineral.K_T
    mineral.gr = mineral.alpha*mineral.K_T*mineral.V/mineral.C_v
    mineral.K_S = mineral.K_T*mineral.C_p/mineral.C_v

    return None
'''
