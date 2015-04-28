# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman import tools
from burnman.mineral import Mineral
from burnman.chemicalpotentials import *
from burnman import constants

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from fitting_functions import *
from scipy import optimize

R=constants.gas_constant

"""
kronecker delta function for integers
"""
kd = lambda x, y: 1 if x == y else 0

'''
# Order-disorder model for FeSi

Notes:
The bcc phase is pretty complicated, with structural changes:
A2->B2->D03

The D03 structure appears around 25 at% Si (as [Fe]0.5[Si]0.25[Si]0.25) at low P and T, probably becoming unstable at about 60 GPa. It has a fairly limited stability range above about 20 GPa, especially at temperatures > 1500 C. At ambient pressure, above the Curie temperature the B2-D03 transition is second order, but below this temperature (~700 C at 10 at.% Si) B2 and D03 structures can coexist, due to interactions between magnetic and chemical ordering.

To describe the D03 structure as well as B2 requires three sites, four endmembers and two order parameters:
Fe    (A2)  [Fe]0.5[Fe]0.25[Fe]0.25 
FeSi  (B2)  [Fe]0.5[Fe]0.25[Si]0.25 
Fe3Si (D03) [Fe]0.5[Si]0.25[Si]0.25
Si    (A2)  [Si]0.5[Si]0.25[Si]0.25

With only B2 to worry about, there are two sites and three endmembers:
Fe    (A2)  [Fe]0.5[Fe]0.5
FeSi  (B2)  [Fe]0.5[Si]0.5
Si    (A2)  [Si]0.5[Si]0.5


Another complication is the magnetism in the phase. Lacaze and Sundman (1991) proposed a quadratic compositional dependence of the Curie temperature and linear dependence on the magnetic moment (i.e. no effect of ordering).

'''
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

endmember_magnetic_moments=np.array([2.2, 1.1, 0.0])
endmember_tcs=np.array([1043., 521.5, 0.])
endmember_alphas=np.array([1.0, 1.0, 1.0])
WBs = np.array([[0.0, 0.0, 0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
WTcs = np.array([[0.0, 0.0, 0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
structural_parameter=0.4

'''

A2-B2 only...

'''


def eqm_order(Q, X, T, m, n, DeltaH, W): # Wab, Wao, Wbo

    A = DeltaH + (m+n)*W[1] - n*W[0]
    B = 2./(m+n)*(-m*m*W[1] + m*n*(W[0] - W[1] - W[2]) - n*n*W[2])
    C = m*(W[2] - W[1] - W[0]) + n*(W[2] - W[1] + W[0])

    pa = 1. - X - m/(m+n)*Q
    pb = X - n/(m+n)*Q

    Kd=(1.-pa)*(1.-pb)/(pa*pb)
    return A + B*Q + C*X + m*n*R*T*np.log(Kd)



# Test diopside jadeite
X=0.5
m=1.
n=1.
DeltaH=-6000.
W=[26000., 16000., 16000.] # convergent ordering

temperatures=np.linspace(373.15, 1373.15, 101)
order=np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    order[i]=optimize.fsolve(eqm_order, 0.999*X*2, args=(X, T, m, n, DeltaH, W))

plt.plot( temperatures, order, linewidth=1, label='order')
plt.title('FeSi ordering')
plt.xlabel("Temperature")
plt.ylabel("Order")
plt.legend(loc='upper right')
plt.show()



# Convergent ordering for B2:
m=n=0.5

X0=0.2
Tc0=1673.15 # X=0.2

'''
X=0.5
Tc=Tc0*(X*(1.-X))/(X0*(1.-X0)) # convert critical temperature
W[0]=-45.8e3
#W[0]=-43.5e3
B = -constants.gas_constant*Tc # (X=0.5)
# B = (0.5*Wab - Wao - Wbo) = (0.5*Wab - 2.0*Wao)

W[1] = (0.5*W[0] - B)/2.0
W[2] = W[1]
DeltaH=0.5*W[0] - W[1] # A+CX = 0, thus A=0

print DeltaH, W, B, Tc 
'''

Si_activity_data=[]
for line in open('data/Sakao_Elliott_1975_activity_coeff_Si_bcc.dat'):
    content=line.strip().split()
    if content[0] != '%':
        Si_activity_data.append(map(float,content))

Si_activity_data = zip(*Si_activity_data)

def activity_coefficient_Si(X, T, m, n, DeltaH, W):
    Q=optimize.fsolve(eqm_order, 0.999*X*2, args=(X, T, m, n, DeltaH, W))[0]
    po=Q
    pa=1. - X - 0.5*Q
    pb=X - 0.5*Q
    RTlngSi=pa*(1.-pb)*W[0] - pa*po*W[1] + (1.-pb)*po*W[2]

    magnetic_Si_activity_contribution=magnetic([1.-X, 0., X], endmember_magnetic_moments, endmember_tcs, endmember_alphas, WBs, WTcs, structural_parameter,T)[0][2]

    #RTlnaidealSi=0.5*constants.gas_constant*T*np.log(pb*(1.-pa))
    
    #return (RTlngSi + RTlnaidealSi)/(constants.gas_constant*T)
    return (RTlngSi+magnetic_Si_activity_contribution)/(constants.gas_constant*T)


def fit_Ws(data, Wh, Ws):
    RTlngSi=[]
    for datum in data:
        T, X = datum
        W = [Wh - T*Ws, 0., 0.] 
        DeltaH = 0.5*W[0]
        RTlngSi.append(activity_coefficient_Si(X, T, m, n, DeltaH, W))
    return np.array(RTlngSi)
        
temperatures = Si_activity_data[0]
compositions = Si_activity_data[1]

xdata= zip(*[temperatures, compositions])
ydata = np.array(Si_activity_data[2])


popt, pcov = optimize.curve_fit(fit_Ws, xdata, ydata)
Wh, Ws = popt

print 'Interaction parameters:', Wh, Ws




temperatures=np.linspace(Tc0-50., Tc0+50., 101)
order=np.empty_like(temperatures)
X=0.2
m=n=0.5
for i, T in enumerate(temperatures):
    W = [Wh - T*Ws, 0., 0.] 
    DeltaH = 0.5*W[0]
    order[i]=optimize.fsolve(eqm_order, 0.999*X*2, args=(X, T, m, n, DeltaH, W))[0]

plt.plot( temperatures, order, linewidth=1, label='order')
plt.title('FeSi ordering')
plt.xlabel("Temperature")
plt.ylabel("Order")
plt.legend(loc='upper right')
plt.show()



from burnman.solidsolution import SolidSolution
from burnman.solutionmodel import *
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass

atomic_masses=read_masses()
class bcc_Fe_Si(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='BCC Fe-Si solid solution'
        structural_parameter=0.4
        magnetic_moments=[2.22, 1.11, 0.0]
        Tcs=[1043., 521.5, 0.]
        magnetic_moment_excesses=[[0., 0.], [0.]]
        Tc_excesses=[[0., 0.], [0.]]
        magnetic_parameters=[structural_parameter, magnetic_moments, Tcs, magnetic_moment_excesses, Tc_excesses]

        endmembers = [[minerals.Myhill_calibration_iron.bcc_iron(), '[Fe]0.5[Fe]0.5'],[minerals.Fe_Si_O.FeSi_B2(), '[Fe]0.5[Si]0.5'],[minerals.Fe_Si_O.Si_bcc_A2(), '[Si]0.5[Si]0.5']]
        enthalpy_interaction=[[0.e3, -59.650e3],[0.e3]]
        entropy_interaction=[[0.e3, -9.494],[0.e3]]
        volume_interaction=[[0.e3, 0.e3],[0.e3]]
        burnman.SolidSolution.__init__(self, endmembers, \
                                           burnman.solutionmodel.SymmetricRegularSolution_w_magnetism(endmembers, magnetic_parameters, enthalpy_interaction, volume_interaction, entropy_interaction), molar_fractions)


bcc=bcc_Fe_Si()



plt.plot( Si_activity_data[1], Si_activity_data[2], marker='.', linestyle='none', label='Sakao and Elliott, 1975')

compositions=np.linspace(0.01, 0.09, 20)
lnactivity=np.empty_like(compositions)
for T in [1373.15, 1473.15, 1573.15, 1623.15]:
    for i, X in enumerate(compositions):
        bcc.set_composition([1.-X, 0.0, X])
        bcc.set_state(1.e5, T)
        lnactivity[i]=bcc.log_activity_coefficients[2]
    plt.plot( compositions, lnactivity, linewidth=1, label=str(T)+' K')


plt.title('FeSi ordering')
plt.xlabel("Composition")
plt.ylabel("log gamma_Si")
plt.legend(loc='lower right')
plt.show()



temperature=1473.15
compositions=np.linspace(0, 1, 101)
val1=np.empty_like(compositions)
val2=np.empty_like(compositions)
val3=np.empty_like(compositions)
for i, X0 in enumerate(compositions):
    X=[X0, 1.-X0, 0.0] # FeSi (ordered) to Fe
    val1[i]=np.dot(np.array(bcc.solution_model._magnetic_excess_partial_gibbs( 1.e5, temperature, X)), np.array(X))
    X=[0.5+X0/2., 0.0, 0.5-X0/2.] # FeSi (disordered) to Fe
    val2[i]=np.dot(np.array(bcc.solution_model._magnetic_excess_partial_gibbs( 1.e5, temperature, X)), np.array(X))
    X=[X0, 0.0, 1.0-X0] # Si to Fe (disordered)     
    val3[i]=np.dot(np.array(bcc.solution_model._magnetic_excess_partial_gibbs( 1.e5, temperature, X)), np.array(X))


plt.plot( compositions, val1, 'g-', linewidth=1, label='FeSi (ordered) to Fe')
plt.plot( compositions, val2, 'r-', linewidth=1, label='FeSi (disordered) to Fe')
plt.plot( compositions, val3, 'b-', linewidth=1, label='Si to Fe')
plt.title('Magnetic gibbs')
plt.xlabel("X (Fe)")
plt.ylabel("Magnetic gibbs contribution (J/mol)")
plt.legend(loc='upper right')
plt.show()



# Calculate Gibbs of the bcc phase at the composition and temperature of interest...

# 1) Calculate equilibrium order
# 2) Calculate gibbs
# 3) Calculate activities



