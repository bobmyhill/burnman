import numpy as np

def calc(pressure, temperature, params):
    Tc = params['Tc_0'] + params['V_D']*pressure/params['S_D']
    if T < Tc:
        Q = 1. - temperature/Tc
        G = params['S_D']*((temperature - Tc)*Q*Q + params['Tc_0']*np.power(Q, 6.))
        dGdT = params['S_D']*(-6.*params['Tc_0']*np.power(Q, 5.)/Tc + Q*Q - 2.*(temperature - Tc)*Q / Tc)
        dGdP = -params['V_D']*Q*(Q - temperature/Tc/Tc*(6.*Tc*np.power(Q, 4.) + 2.*(temperature - Tc)))
        d2GdT2 = params['S_D']*(30.*params['Tc_0']*np.power(Q, 4.)/Tc/Tc - 4.*Q/Tc + 2.*(temperature - Tc)/Tc/Tc)
        d2GdP2 = 2.*params['V_D']*params['V_D']*temperature / (params['S_D']*Tc*Tc) \
            * (15.*params['Tc_0']*temperature*np.power(Q, 4.)/Tc/Tc + temperature*(temperature - Tc)/Tc/Tc \
                   - 6.*params['Tc_0']*np.power(Q, 5.)/Tc - 2.*Q - 2.*Q*(temperature - Tc)/Tc)
        d2GdPdT = (2.*params['V_D']/Tc/Tc)*(3.*params['Tc_0']*
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



params = {
    'S_D': 5.,
    'V_D': 1.e-8,
    'Tc_0': 1200.
    }


print 'S, V, Cp, KT, alpha'
P = 1.e9
temperatures = np.linspace(300., 1300., 11)
for i, T in enumerate(temperatures):
    dT = 0.01
    dP = 100000.
    G, dGdT, dGdP, d2GdT2, d2GdP2, d2GdPdT = calc(P, T, params)
    G1, dGdT1, dGdP1, d2GdT21, d2GdP21, d2GdPdT1 = calc(P-dP, T-dT, params)
    G2, dGdT2, dGdP2, d2GdT22, d2GdP22, d2GdPdT2 = calc(P-dP, T+dT, params)
    G3, dGdT3, dGdP3, d2GdT23, d2GdP23, d2GdPdT3 = calc(P+dP, T-dT, params)
    G4, dGdT4, dGdP4, d2GdT24, d2GdP24, d2GdPdT4 = calc(P+dP, T+dT, params)
    G5, dGdT5, dGdP5, d2GdT25, d2GdP25, d2GdPdT5 = calc(P-dP, T, params)
    G6, dGdT6, dGdP6, d2GdT26, d2GdP26, d2GdPdT6 = calc(P, T-dT, params)

    print dGdT - (G2 - G1)/(2.*dT),
    print dGdP - (G3 - G1)/(2.*dP),
    print d2GdT2 - ((G2 - G5)/dT - (G5 - G1)/dT) / dT, 
    print d2GdP2 - ((G3 - G6)/dP - (G6 - G1)/dP) / dP,
    print d2GdPdT - ((G4 - G3)/(2.*dT) - (G2 - G1)/(2.*dT))/(2.*dP)


    print pressure, temperature, params
    Tc = params['Tc_0'] + params['V_D']*pressure/params['S_D']
    if temperature < Tc:
        # Wolfram input to check partial differentials
        # x = T, y = P, a = S, c = Tc0, d = V
        # D[a ((-c + x - (d y)/a) (1 - x/(c + (d y)/a))^2 + c (1 - x/(c + (d y)/(a)))^6)/3., x]
        # Note mistake in Stixrude and Lithgow Bertelloni (2011) for the Vex term (equation 32)
        Q = 1. - temperature/Tc
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
