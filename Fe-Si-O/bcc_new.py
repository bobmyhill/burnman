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

'''

# Test diopside jadeite
X=0.5
m=1.
n=1.
DeltaH=-6000.
W=[26000., 16000., 16000.]

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
'''

'''
class dummy (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'dummy endmember',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 0 ,
            'S_0': 0 ,
            'V_0': 1.e-5 ,
            'Cp': [0, 0, 0, 0] ,
            'a_0': 0. ,
            'K_0': 100.e9 ,
            'Kprime_0': 4. ,
            'Kdprime_0': 0. ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class dummy_2 (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'dummy endmember',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -6000. ,
            'S_0': 0 ,
            'V_0': 1.e-5 ,
            'Cp': [0, 0, 0, 0] ,
            'a_0': 0. ,
            'K_0': 100.e9 ,
            'Kprime_0': 4. ,
            'Kdprime_0': 0. ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)


from burnman.solidsolution import SolidSolution
from burnman.solutionmodel import *
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass

atomic_masses=read_masses()
class bcc_Fe_Si(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='BCC Fe-Si solid solution'
        endmembers = [[dummy(), '[Fe][Fe]'],[dummy_2(), '[Fe][Si]'],[dummy(), '[Si][Si]']]
        enthalpy_interaction=[[16.e3, 26.e3],[16.e3]]
        burnman.SolidSolution.__init__(self, endmembers, \
                                           burnman.solutionmodel.SymmetricRegularSolution(endmembers, enthalpy_interaction), molar_fractions)


bcc=bcc_Fe_Si()


def eqm_order(Q, X, T):
    bcc.set_composition()
    bcc.set_state(1.e5, T)
    return bcc.gibbs


'''


def eqm_order_2(order_parameters, deltaHs, Ws, X, T):
    Q1, Q2 = order_parameters
    deltaH0, deltaH1 = deltaHs
    W=Ws

# Proportions
    p=[1.-X-0.5*Q1-0.75*Q2,Q2, Q1, X-0.5*Q1-0.25*Q2]
    
# Non ideal activities
    RTlng=[0., 0., 0., 0.]
    RTlng[0] = (1-p[0])*(-p[1]*W[0][1] - p[2]*W[0][2] - p[3]*W[0][3]) -p[1]*(-p[2]*W[1][2] - p[3]*W[1][3]) - p[2]*(-p[3]*W[2][3])
    RTlng[1] = (1-p[1])*(-p[2]*W[1][2] - p[3]*W[1][3] - p[0]*W[0][1]) -p[2]*(-p[3]*W[2][3] - p[0]*W[0][2]) - p[3]*(-p[0]*W[0][3])
    RTlng[2] = (1-p[2])*(-p[3]*W[2][3] - p[0]*W[0][2] - p[1]*W[1][2]) -p[3]*(-p[0]*W[0][3] - p[1]*W[1][3]) - p[0]*(-p[1]*W[0][1])
    RTlng[3] = (1-p[3])*(-p[0]*W[0][3] - p[1]*W[1][3] - p[2]*W[2][3]) -p[0]*(-p[1]*W[0][1] - p[2]*W[0][2]) - p[1]*(-p[2]*W[1][2])

    XAFe=1. - X + 0.5*Q1 + 0.25*Q2
    XBFe=1. - X - 0.5*Q1 + 0.25*Q2
    XCFe=1. - X - 0.5*Q1 - 0.75*Q2

    XASi=1.-XAFe
    XBSi=1.-XBFe
    XCSi=1.-XCFe

    lnKd0=(1./8.)*np.log((XAFe*XAFe*XBSi*XCSi)/(XASi*XASi*XBFe*XCFe))
    
    lnKd1=(1./16.)*np.log((XAFe*XAFe*XBFe*XCSi*XCSi*XCSi)/(XASi*XASi*XBSi*XCFe*XCFe*XCFe))

    # mu_i = G_i + RTlna
    
    # 0 = Fe0.5Si0.5 - 0.5 Fe - 0.5 Si
    # deltaH = H_reaction
    eqn0=deltaH0 - (RTlng[2] - 0.5*RTlng[0] - 0.5*RTlng[3]) + R*T*lnKd0
    eqn1=deltaH1 - (RTlng[1] - 0.75*RTlng[0] - 0.25*RTlng[3]) + R*T*lnKd1

    return [eqn0, eqn1]

# Fe, Fe3Si, FeSi, Si
Ws=[[0., 0., 16000., 26000.],[0., 0., 0., 0.],[0., 0., 0., 16000.]]
deltaHs=[-6000., -6000.]
T=2200.
X=0.3


# Test
m=0.5
n=0.5
DeltaH=-6000.
W=[26000., 16000., 16000.]

order=optimize.fsolve(eqm_order, [0.599], args=(X, T, m, n, DeltaH, W))
print eqm_order(order, X, T, m, n, DeltaH, W)
print order


order=optimize.fsolve(eqm_order_2, [0.999, 0.199], args=(deltaHs, Ws, X, T))
print eqm_order_2(order, deltaHs, Ws, X, T)
print order
