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

from burnman.solidsolution import SolidSolution
from burnman.solutionmodel import *
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()


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


print 'WARNING, STILL NEED TO SET MIXING PARAMETERS VALUES IN SOLUTION MODEL'

'''
# First, we fit the activity data to simple symmetric disordered mixing between Fe and Si
# (Q is zero at the low Si contents and high temperatures of Sakao and Elliott)
'''


dH = -81000./2.
Wh = -59670.9616111 
Ws = -9.50500004066
Whod = (Wh/2. - dH)

def eqm_order(Q, X, T, m, n, DeltaH, W): # Wab, Wao, Wbo
    A = DeltaH + (m+n)*W[1] - n*W[0]
    B = 2./(m+n)*(-m*m*W[1] + m*n*(W[0] - W[1] - W[2]) - n*n*W[2])
    C = m*(W[2] - W[1] - W[0]) + n*(W[2] - W[1] + W[0])

    pa = 1. - X - m/(m+n)*Q
    pb = X - n/(m+n)*Q

    Kd=(1.-pa)*(1.-pb)/(pa*pb)
    return A + B*Q + C*X + m*n*constants.gas_constant*T*np.log(Kd)


'''
Now we can make the bcc solid solution
It turns out that the order-disorder transition occurs at almost exactly the temperature 
predicted for an interaction parameter of 0 kJ/mol between Fe-FeSi and FeSi-Si, which is nice.
'''
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
        enthalpy_interaction=[[Whod, Wh],[Whod]]
        entropy_interaction=[[0.e3, Ws],[0.e3]]
        volume_interaction=[[0., 0.],[0.]] # [[0.e3, ((7.31 - (9.2+7.09)/2.)*1.e-6)*4],[0.e3]] # V (Fe0.5Si0.5 dis) from Brosh et al., 2009, Pearson
        burnman.SolidSolution.__init__(self, endmembers, \
                                           burnman.solutionmodel.SymmetricRegularSolution_w_magnetism(endmembers, magnetic_parameters, enthalpy_interaction, volume_interaction, entropy_interaction), molar_fractions)

bcc=bcc_Fe_Si()
FeSi_B20=minerals.Fe_Si_O.FeSi_B20()

'''
Plot equilibrium state of order for bcc Fe-Si alloy 
'''

def bcc_order(X, P, T):
    bcc.set_composition([0.5, 0.0, 0.5])
    bcc.set_state(P, T)
    Wab=Wh-T*Ws
    DeltaH = bcc.endmembers[1][0].gibbs - 0.5*(bcc.endmembers[0][0].gibbs+bcc.endmembers[2][0].gibbs)
    DeltaH -= bcc.endmembers[1][0].method._magnetic_gibbs(P, T, bcc.endmembers[1][0].params) - 0.5*(bcc.endmembers[0][0].method._magnetic_gibbs(P, T, bcc.endmembers[0][0].params))
    Wao=Wbo=n*Wab - (m+n)*DeltaH
    W = [Wab, Wao, Wbo]
    return optimize.fsolve(eqm_order, 0.999*X*2, args=(X, T, m, n, DeltaH, W))[0]

X=0.2
temperatures=np.linspace(600., 3200., 101)
order=np.empty_like(temperatures)
m=n=0.5

for i, T in enumerate(temperatures):
    order[i]=bcc_order(X, 1.e5, T)

plt.plot( temperatures, order, linewidth=1, label='order')
plt.title('FeSi ordering')
plt.xlabel("Temperature (K)")
plt.ylabel("Order")
plt.legend(loc='upper right')
plt.show()


'''
Plot activity coefficients of dilute Fe-Si alloys
'''


Si_activity_data=[]
for line in open('data/Sakao_Elliott_1975_activity_coeff_Si_bcc.dat'):
    content=line.strip().split()
    if content[0] != '%':
        Si_activity_data.append(map(float,content))
Si_activity_data = zip(*Si_activity_data)

plt.plot( Si_activity_data[1], Si_activity_data[2], marker='.', linestyle='none', label='Sakao and Elliott, 1975')

compositions=np.linspace(0.01, 0.09, 20)
lnactivity=np.empty_like(compositions)
for T in [1373.15, 1473.15, 1573.15, 1623.15]:
    for i, X in enumerate(compositions):
        Q=bcc_order(X, 1.e5, T)
        #print X, T, Q
        bcc.set_composition([1.-X-0.5*Q, Q, X-0.5*Q])
        bcc.set_state(1.e5, T)
        lnactivity[i]=bcc.log_activity_coefficients[2]
    plt.plot( compositions, lnactivity, linewidth=1, label=str(T)+' K')
    np.savetxt(str(T)+'_Si_activities.dat', zip(*[compositions, lnactivity]))

    
plt.title('FeSi ordering')
plt.xlabel("Composition")
plt.ylabel("log gamma_Si")
plt.legend(loc='lower right')
plt.show()


'''
Plot the magnetic excess gibbs free energy
'''
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



'''
Plot equilibrium compositions of bcc alloy in equilibrium with FeSi in the B20 structure
'''

def eqm_comp_bcc_w_FeSi_B20(arg, P, T):
    X_bcc = arg[0]
    Q = bcc_order(X_bcc, P, T)
    bcc.set_composition([1.-X-0.5*Q, Q, X-0.5*Q])
    bcc.set_state(P, T)
    mu_FeSi=bcc.partial_gibbs[0] + bcc.partial_gibbs[2]
    FeSi_B20.set_state(P, T)
    return [mu_FeSi - FeSi_B20.gibbs]

def eqm_T_B20_B2(arg, P):
    T=arg[0]
    X=0.5
    Q = bcc_order(X, P, T)
    bcc.set_composition([1.-X-0.5*Q, Q, X-0.5*Q])
    bcc.set_state(P, T)
    FeSi_B20.set_state(P, T)
    return [bcc.gibbs*2. - FeSi_B20.gibbs]

pressures=np.linspace(1.e5, 80.e9, 100)
temperatures=np.empty_like(pressures)
for i, P in enumerate(pressures):
    T=optimize.fsolve(eqm_T_B20_B2, [2500.], args=(P))[0]
    X=0.5
    Q = bcc_order(X, P, T)
    bcc.set_composition([1.-X-0.5*Q, Q, X-0.5*Q])
    bcc.set_state(P, T)
    FeSi_B20.set_state(P, T)
    temperatures[i]=T
    #print P/1.e9, T, Q, bcc.gibbs*2., FeSi_B20.gibbs


B20_data=[]
B20_B2_data=[]
B2_data=[]
for line in open('data/FeSi_Lord_2010.dat'):
    content=line.strip().split()
    if content[2] == 'B20':
        B20_data.append([float(content[0]), float(content[1])])
    if content[2] == 'B20_B2':
        B20_B2_data.append([float(content[0]), float(content[1])])
    if content[2] == 'B2':
        B2_data.append([float(content[0]), float(content[1])])

        
B20_data = zip(*B20_data)
B20_B2_data = zip(*B20_B2_data)
B2_data = zip(*B2_data)

plt.plot( pressures/1.e9, temperatures, linewidth=1, label='ordered')
plt.plot( B20_data[0], B20_data[1], marker='.', linestyle='None', label='B20' )
plt.plot( B20_B2_data[0], B20_B2_data[1], marker='.', linestyle='None', label='B20_B2' )
plt.plot( B2_data[0], B2_data[1], marker='.', linestyle='None', label='B2' )
plt.title('FeSi reactions')
plt.xlabel("Pressure (GPa)")
plt.ylabel("Temperature")
plt.legend(loc='upper right')
plt.show()


np.savetxt('B2_B20.dat', zip(*[pressures/1.e9, temperatures]))






T=2200. # K
V_ordered=[]
V_disordered=[]
V_Fe=[]
V_Si=[]
pressures=np.linspace(1.e5, 50.e9, 101)
for P in pressures:
    bcc.set_composition([0., 1., 0.])
    bcc.set_state(P, T)
    V_ordered.append(bcc.V)
    bcc.set_composition([0.5, 0., 0.5])
    bcc.set_state(P, T)
    V_disordered.append(bcc.V)
    V_Fe.append(bcc.endmembers[0][0].V)
    V_Si.append(bcc.endmembers[2][0].V)

plt.plot( pressures/1.e9, V_ordered, linewidth=1, label='ordered')
plt.plot( pressures/1.e9, V_disordered, linewidth=1, label='disordered')
plt.plot( pressures/1.e9, V_Fe, linewidth=1, label='Fe')
plt.plot( pressures/1.e9, V_Si, linewidth=1, label='Si')
plt.title('FeSi ordering')
plt.xlabel("Pressure (GPa)")
plt.ylabel("Volume")
plt.legend(loc='upper right')
plt.show()


'''
Now let's deal with the HCP solid solution
'''

class hcp_Fe_Si(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='HCP Fe-Si solid solution'

        endmembers = [[minerals.Myhill_calibration_iron.hcp_iron(), '[Fe]'],[minerals.Fe_Si_O.Si_hcp_A3(), '[Si]']]
        enthalpy_interaction=[[-50.e3]]
        entropy_interaction=[[0.e3]]
        volume_interaction=[[3.11e-7]]
        burnman.SolidSolution.__init__(self, endmembers, \
                                           burnman.solutionmodel.SymmetricRegularSolution(endmembers, enthalpy_interaction, volume_interaction, entropy_interaction), molar_fractions)

hcp=hcp_Fe_Si()



def bcc_hcp_eqm(data, P, T):
    Xbcc=data[0]
    Xhcp=data[1]
    Q=bcc_order(Xbcc, P, T)
    bcc.set_composition([1.-Xbcc-0.5*Q, Q, Xbcc-0.5*Q])
    bcc.set_state(P, T)
    hcp.set_composition([1.-Xhcp, Xhcp])
    hcp.set_state(P, T)
    #print [bcc.partial_gibbs[0]-hcp.partial_gibbs[0], bcc.partial_gibbs[2]-hcp.partial_gibbs[1]]
    return [bcc.partial_gibbs[0]-hcp.partial_gibbs[0], bcc.partial_gibbs[2]-hcp.partial_gibbs[1]]


P=125.e9
temperatures=np.linspace(1000., 3000., 50)
plot_temperatures=[]
plot_bcc=[]
plot_hcp=[]
for T in temperatures:
    solve=optimize.fsolve(bcc_hcp_eqm, [0.49, 0.01], args=(P, T), full_output=True)
    if solve[2]==1:
        plot_temperatures.append(T)
        plot_bcc.append(solve[0][0])
        plot_hcp.append(solve[0][1])

# 0.49, 0.31
# 0.46, 0.18
# 0.29, 0.05
# 0.11, 0.07


plt.plot( plot_bcc, plot_temperatures, linewidth=1, label='bcc')
plt.plot( plot_hcp, plot_temperatures, linewidth=1, label='hcp')
plt.title('Two phase region')
plt.xlabel("Composition")
plt.ylabel("Temperature (K)")
plt.legend(loc='upper right')
plt.show()
