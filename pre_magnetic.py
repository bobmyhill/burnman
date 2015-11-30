from burnman.constants import gas_constant
import numpy as np
# x = T
# y = P
# a, b = magnetic moment
# c, d = curie_temperature

# D[D[r*x*log(a + y*b + 1)*((x/(c + y*d))^-5 / 10 + (x/(c + y*d))^-15 / 315 + (x/(c + y*d))^-25 / 1500) , y], y]
def calc(pressure, temperature, params):
    
    structural_parameter=params['magnetic_structural_parameter']
    tau=temperature/(params['curie_temperature'][0] + pressure*params['curie_temperature'][1])
    magnetic_moment=params['magnetic_moment'][0] + pressure*params['magnetic_moment'][1]

    A = (518./1125.) + (11692./15975.)*((1./structural_parameter) - 1.)
    if tau < 1: 
        f=1.-(1./A)*(79./(140.*structural_parameter*tau) + (474./497.)*(1./structural_parameter - 1.)*(np.power(tau, 3.)/6. + np.power(tau, 9.)/135. + np.power(tau, 15.)/600.))

        G = gas_constant*temperature*np.log(magnetic_moment + 1.)*f
        dGdT = 0.
        dGdP = 0.
        d2GdT2 = 0.
        d2GdP2 = 0.
        d2GdPdT = 0.
    else:
        f=-(1./A)*(np.power(tau,-5)/10. + np.power(tau,-15)/315. + np.power(tau, -25)/1500.)

        G = gas_constant*temperature*np.log(magnetic_moment + 1.)*f
        dGdT = gas_constant*np.log(magnetic_moment + 1.)*-(1./A) \
            *(np.power(tau,-5)*(1./10. - 1./2.) 
              + np.power(tau,-15)*(1./315. - 1./21.)  
              + np.power(tau, -25)*(1/1500. - 1./60.))
        dGdP = 0.
        d2GdT2 = gas_constant*np.log(magnetic_moment + 1.)*-(1./A)/temperature \
            *(np.power(tau,-5)*(2.) 
              + np.power(tau,-15)*(2./3.)  
              + np.power(tau, -25)*(2./5.))
        d2GdP2 = 0.
        d2GdPdT = 0.


    return G, dGdT, dGdP, d2GdT2, d2GdP2, d2GdPdT



params = {
    'curie_temperature': [1000., 1.e-8],
    'magnetic_moment': [2.2, 1.e-10],
    'magnetic_structural_parameter': 0.4
    }


print 'S, V, Cp, KT, alpha'
P = 1.e9
temperatures = np.linspace(300., 1300., 11)
for i, T in enumerate(temperatures):
    dT = 0.01
    dP = 1.
    G, dGdT, dGdP, d2GdT2, d2GdP2, d2GdPdT = calc(P, T, params)
    G1, dGdT1, dGdP1, d2GdT21, d2GdP21, d2GdPdT1 = calc(P-dP, T-dT, params)
    G2, dGdT2, dGdP2, d2GdT22, d2GdP22, d2GdPdT2 = calc(P-dP, T+dT, params)
    G3, dGdT3, dGdP3, d2GdT23, d2GdP23, d2GdPdT3 = calc(P+dP, T-dT, params)
    G4, dGdT4, dGdP4, d2GdT24, d2GdP24, d2GdPdT4 = calc(P+dP, T+dT, params)
    G5, dGdT5, dGdP5, d2GdT25, d2GdP25, d2GdPdT5 = calc(P-dP, T, params)
    G6, dGdT6, dGdP6, d2GdT26, d2GdP26, d2GdPdT6 = calc(P, T-dT, params)

    #print dGdT, (G2 - G1)/(2.*dT)
    print dGdP,  (G3 - G1)/(2.*dP)
    #print d2GdT2, ((G2 - G5)/dT - (G5 - G1)/dT) / dT
    print d2GdP2, ((G3 - G6)/dP - (G6 - G1)/dP) / dP
    #print d2GdPdT - ((G4 - G3)/(2.*dT) - (G2 - G1)/(2.*dT))/(2.*dP)
