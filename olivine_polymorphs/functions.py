def gibbs_murnaghan(pressure, params):
    G0, V0, K0, Kprime0 = params
    exponent=(Kprime0-1.0)/Kprime0
    return G0 + V0*(K0/(Kprime0 - 1.0)*(np.power((1.+Kprime0/K0*pressure),exponent)-1.)) 

def pressure(VoverV0, dT, K0, a, b, dPdT):
    f = (np.power(VoverV0, -2./3.)-1.)/2.
    return 3.*K0*f*np.power((1.+2.*f), 2.5)*(1. + a*f +b*f*f) +dPdT*dT
