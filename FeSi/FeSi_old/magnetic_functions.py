import numpy as np
from burnman import constants

"""
kronecker delta function for integers
"""
kd = lambda x, y: 1 if x == y else 0


def _phi( molar_fractions, alpha):
    phi=np.array([alpha[i]*molar_fractions[i] for i in range(len(molar_fractions))])
    phi=np.divide(phi, np.sum(phi))
    return phi

def _magnetic_gibbs(temperature, Tc, magnetic_moment, structural_parameter):
    """
    Returns the magnetic contribution to the Gibbs free energy [J/mol]
    Expressions are those used by Chin, Hertzman and Sundman (1987)
    as reported in Sundman in the Journal of Phase Equilibria (1991)
    """
    tau=temperature/Tc

    A = (518./1125.) + (11692./15975.)*((1./structural_parameter) - 1.)
    if tau < 1: 
        f=1.-(1./A)*(79./(140.*structural_parameter*tau) + (474./497.)*(1./structural_parameter - 1.)*(np.power(tau, 3.)/6. + np.power(tau, 9.)/135. + np.power(tau, 15.)/600.))
    else:
        f=-(1./A)*(np.power(tau,-5)/10. + np.power(tau,-15)/315. + np.power(tau, -25)/1500.)
    return constants.gas_constant*temperature*np.log(magnetic_moment + 1.)*f
        

def magnetic(X, endmember_magnetic_moments, endmember_tcs, endmember_alphas, WBs, WTcs, structural_parameter, temperature):

    phi=_phi(X, endmember_alphas)

    # magnetic_moment and tc value at X
    Tc=np.dot(endmember_tcs.T, X) + np.dot(endmember_alphas.T,X)*np.dot(phi.T,np.dot(WTcs,phi))

    if Tc > 1.e-12:
        tau=temperature/Tc
        magnetic_moment=np.dot(endmember_magnetic_moments, X) + np.dot(endmember_alphas.T,X)*np.dot(phi.T,np.dot(WBs,phi))
        Gmag=_magnetic_gibbs(temperature, Tc, magnetic_moment, structural_parameter)

        A = (518./1125.) + (11692./15975.)*((1./structural_parameter) - 1.)
        if tau < 1: 
            f=1.-(1./A)*(79./(140.*structural_parameter*tau) + (474./497.)*(1./structural_parameter - 1.)*(np.power(tau, 3.)/6. + np.power(tau, 9.)/135. + np.power(tau, 15.)/600.))
        else:
            f=-(1./A)*(np.power(tau,-5)/10. + np.power(tau,-15)/315. + np.power(tau, -25)/1500.)
        b=(474./497.)*(1./structural_parameter - 1.)
        a=[-79./(140.*structural_parameter), -b/6., -b/135., -b/600., -1./10., -1./315., -1./1500.]
            
        # Now calculate local change in B, Tc with respect to X_1
        # Endmember excesses
            
        dtaudtc=-temperature/(Tc*Tc)
        if tau < 1: 
            dfdtau=(1./A)*(-a[0]/(tau*tau) + 3.*a[1]*np.power(tau, 2.) + 9.*a[2]*np.power(tau, 8.) + 15.*a[3]*np.power(tau, 14.))
        else:
            dfdtau=(1./A)*(-5.*a[4]*np.power(tau,-6) - 15.*a[5]*np.power(tau,-16) - 25.*a[6]*np.power(tau, -26))
    else:
        Gmag=dfdtau=dtaudtc=magnetic_moment=f=0.0
            
    partial_B=np.zeros(len(X))
    partial_Tc=np.zeros(len(X))
    endmember_Gmag=np.zeros(len(X))
    for l in range(len(X)):
        if endmember_tcs[l] > 1.e-12:
            endmember_Gmag[l] = _magnetic_gibbs(temperature, endmember_tcs[l], endmember_magnetic_moments[l], structural_parameter)

        q=np.array([kd(i,l)-phi[i] for i in range(len(X))])
        partial_B[l]=endmember_magnetic_moments[l]-endmember_alphas[l]*np.dot(q,np.dot(WBs,q))
        partial_Tc[l]=endmember_tcs[l]-endmember_alphas[l]*np.dot(q,np.dot(WTcs,q))

    tc_diff = partial_Tc - Tc
    magnetic_moment_diff= partial_B - magnetic_moment

    dGdXdist=constants.gas_constant*temperature*(magnetic_moment_diff*f/(magnetic_moment + 1.) + dfdtau*dtaudtc*tc_diff*np.log(magnetic_moment + 1.))

    endmember_contributions=np.dot(endmember_Gmag, X) 

    # Calculate partials
    return Gmag - endmember_contributions + dGdXdist, Gmag - endmember_contributions
