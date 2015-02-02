# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman.mineral import Mineral
from burnman.processchemistry import *
from burnman.chemicalpotentials import *
from burnman import constants

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import optimize
atomic_masses=read_masses()


'''
First, let's use the description of bcc iron from Sundman, 1991
N.B.: Saxena and Dubrovinsky, 1998 (Geophysical Monograph 101) have a similar expression, but this is basically the same as Sundman's, with a few numbers tweaked badly (there is a discontinuity in Gbcc at 1811 K) and insufficient data in the paper (no reference to the magnetic contribution in FCC).
'''

def magnetic_gibbs(T, Tc, beta, p):
    A = (518./1125.) + (11692./15975.)*((1./p) - 1.)
    tau=T/Tc
    if tau < 1: 
        f=1.-(1./A)*(79./(140.*p*tau) + (474./497.)*(1./p - 1.)*(np.power(tau, 3.)/6. + np.power(tau, 9.)/135. + np.power(tau, 15.)/600.))
    else:
        f=-(1./A)*(np.power(tau,-5)/10. + np.power(tau,-15)/315. + np.power(tau, -25)/1500.)
    return constants.gas_constant*T*np.log(beta + 1.)*f

def HSERFe(T):
    if T < 1811:
        gibbs=1224.83 + 124.134*T - 23.5143*T*np.log(T) - 0.00439752*T*T - 5.89269e-8*T*T*T + 77358.3/T
    else:
        gibbs=-25384.451 + 299.31255*T - 46.*T*np.log(T) + 2.2960305e31*np.power(T,-9.)
    return gibbs 

def gibbs_bcc_1bar_SD1998(T):
    if T < 1811:
        gibbs= 1400. + 124.06*T - 23.5143*T*np.log(T) -0.00439752*T*T - 5.8927e-8*T*T*T + 77359./T 
    else:
        gibbs= - 25383.581 + 299.31255*T - 46.*T*np.log(T) + 2.29603e31*np.power(T,-9.)
    Tc=1043.
    beta=2.22
    p=0.4
    return gibbs + magnetic_gibbs(T, Tc, beta, p)

def gibbs_bcc_1bar(T):
    Tc=1043.
    beta=2.22
    p=0.4
    return HSERFe(T) + magnetic_gibbs(T, Tc, beta, p)

def gibbs_fcc_1bar(T):
    Tc=201.
    beta=2.10
    p=0.28
    if T < 1811:
        gibbs=HSERFe(T) - 1462.4 + 8.282*T - 1.15*T*np.log(T) + 0.00064*T*T
    else:
        gibbs= - 27098.266 + 300.25256*T - 46.*T*np.log(T) + 2.78854e31*np.power(T,-9.)
    return gibbs + magnetic_gibbs(T, Tc, beta, p)

print 'Gibbs of bcc at 298.15 K:', gibbs_bcc_1bar(298.15)
print 'Since bcc is the stable phase at 298.15 K (i.e. Hf=0.0), entropy at 298.15 K is', -gibbs_bcc_1bar(298.15)/298.15, 'J/K/mol'

def bcc_hcp_eqm(T):
    return gibbs_bcc_1bar(T) - gibbs_fcc_1bar(T)

print 'fcc stable between', optimize.fsolve(bcc_hcp_eqm, 1000.)[0], 'and', optimize.fsolve(bcc_hcp_eqm, 1800.)[0], 'K'

'''
Now, let's compare this to the gibbs free energy in the FCC model we'll be using to extrapolate to high pressure...
'''

# Convert from parameters in Komabayashi and Fei
# a* = H_0
# b* = a - S_0
# c* = -a
# d* = -b/2
# f* = -c/2
# i* = 4d

a_KF=16300.921
b_KF=381.47162
c_KF=-52.2754
d_KF=0.000177578
f_KF=-395355.43
i_KF=-2476.28

Cp_fcc=[-c_KF, -2.*d_KF, -2.*f_KF, i_KF/4.]
T=298.15
S_0_fcc = Cp_fcc[0] - b_KF + ( Cp_fcc[0]*np.log(T) + Cp_fcc[1]*T - Cp_fcc[2]/2./T/T - 2.*Cp_fcc[3]/np.sqrt(T) )
H_0_fcc = a_KF + ( Cp_fcc[0]*T + Cp_fcc[1]*T*T/2. - Cp_fcc[2]/T + 2*Cp_fcc[3]*np.sqrt(T) )

class fcc_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FCC iron',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': H_0_fcc ,
            'S_0': S_0_fcc ,
            'V_0': 7.09e-06 ,
            'Cp': Cp_fcc ,
            'a_0': 3.56e-05 ,
            'K_0': 1.64e+11 ,
            'Kprime_0': 5.16 ,
            'Kdprime_0': -3.1e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)
fcc=fcc_iron()

temperatures=np.linspace(300, 2000, 101)
bcc_gibbs=np.empty_like(temperatures)
bcc_gibbs_SD1998=np.empty_like(temperatures)
fcc_gibbs=np.empty_like(temperatures)
fcc_gibbs_model=np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    bcc_gibbs[i] = gibbs_bcc_1bar(T)
    bcc_gibbs_SD1998[i] = gibbs_bcc_1bar_SD1998(T)
    fcc_gibbs[i] = gibbs_fcc_1bar(T)
    fcc.set_state(1.e5, T)
    fcc_gibbs_model[i] = fcc.gibbs

plt.plot( temperatures, bcc_gibbs - fcc_gibbs, 'r-', linewidth=3., label='bcc - fcc (Sundman, 1991)')
plt.plot( temperatures, bcc_gibbs_SD1998 - fcc_gibbs_model, 'b-', linewidth=3., label='bcc - fcc (Saxena and Dubrovinsky, 1998)')

plt.title('Gibbs free energy of iron')
plt.xlabel("Temperature (K)")
plt.ylabel("G (J/mol)")
plt.legend(loc='lower right')
plt.show()

'''
Next, we find the FCC ($\gamma$-iron) thermal equation of state from the data of Tsujino et al., 2013
They used the MgO pressure standard of Matsui et al., 2000
'''

fcc_data=[]
for line in open('Tsujino_et_al_2013.dat'):
    content=line.strip().split()
    if content[0] != '%':
        fcc_data.append(map(float, content))

# Temperature [K]  Volume of y-Fe [angstrom^3] (with error)  Pressure [GPa] (with error)  V/V0 of MgO [angstrom^3] (with error) 
T, V, Verr, P, Perr, relV, relVerr = zip(*fcc_data)


Z=4.
nA=6.02214e23
voltoa=1.e30

volumes=np.array(V)*(nA/Z/voltoa)
sigma=np.array(Verr)*(nA/Z/voltoa)
pt=np.array(zip(np.array(P)*1.e9, T))

# Initial guess.
def fitV(mineral):
    def fit(data, V_0, K_0, Kprime_0, a_0):
        mineral.params['V_0'] = V_0 
        mineral.params['K_0'] = K_0
        mineral.params['Kprime_0'] = Kprime_0
        mineral.params['Kdprime_0'] = -Kprime_0/K_0
        mineral.params['a_0'] = a_0
        
        vols=[]
        for i, datum in enumerate(data):
            mineral.set_state(datum[0], datum[1])
            vols.append(mineral.V)
        return vols
    return fit

guesses=np.array([fcc.params['V_0'], fcc.params['K_0'], fcc.params['Kprime_0'], fcc.params['a_0']])
popt, pcov = optimize.curve_fit(fitV(fcc), pt, volumes, guesses, sigma)

print ''
print 'Fitted FCC parameters'
print "V0: ", popt[0], "+/-", np.sqrt(pcov[0][0]), "m^3/mol"
print "V0: ", popt[0]/(nA/Z/voltoa), "+/-", np.sqrt(pcov[0][0])/(nA/Z/voltoa), "A^3"
print "k0: ", popt[1]/1.e9, "+/-", np.sqrt(pcov[1][1])/1.e9, "GPa"
print "k0':", popt[2], "+/-", np.sqrt(pcov[2][2])
print "k0\":", -1.e9*popt[2]/popt[1], "GPa^-1"
print "a0 :", popt[3], "+/-", np.sqrt(pcov[3][3]), "K^-1"


Vdiff=np.empty_like(volumes)
for i, datum in enumerate(pt):
        fcc.set_state(datum[0], datum[1])
        Vdiff[i]=(fcc.V - volumes[i])/volumes[i]*100.

plt.plot( np.array(P), Vdiff , marker=".", linestyle="None")
plt.title('Volume fit')
plt.xlabel("Pressure (GPa)")
plt.ylabel("Percentage volume difference (m^3/mol)")
plt.show()



print

print "Covariance matrix:"
print pcov


'''
Now, we find the HCP ($\varepsilon$-iron thermal equation of state from 
They used Au as a pressure standard (equation of state from Tsuchiya, 2003)
'''

class hcp_iron (Mineral):
    def __init__(self):
        formula='Fe1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'HCP iron',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -0.0 ,
            'S_0': 27.09 ,
            'V_0': 7.09e-06 ,
            'Cp': [46.2, 0.005159, 723100.0, -556.2] ,
            'a_0': 3.56e-05 ,
            'K_0': 1.64e+11 ,
            'Kprime_0': 5.16 ,
            'Kdprime_0': -3.1e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

hcp=hcp_iron()

hcp_data=[]
for line in open('Yamazaki_et_al_2012.dat'):
    content=line.translate(None, ')(').strip().split()
    if content[0] != '%':
        hcp_data.append(map(float, content))

# T (K) V (Au) P a-axis (A) c-axis (A) V (e-Fe [HCP] angstroms^3)
T, VAu, VAuerr, P, Perr, a, aerr, c, cerr, V, Verr = zip(*hcp_data)

Z=2.
volumes=np.array(V)*(nA/Z/voltoa)
sigma=np.array(Verr)*(nA/Z/voltoa)
pt=np.array(zip(np.array(P)*1.e9, T))

# Initial guess.
guesses=np.array([hcp.params['V_0'], hcp.params['K_0'], hcp.params['Kprime_0'], hcp.params['a_0']])
popt, pcov = optimize.curve_fit(fitV(hcp), pt, volumes, guesses, sigma)

print ''
print 'Fitted HCP parameters'
print "V0: ", popt[0], "+/-", np.sqrt(pcov[0][0]), "m^3/mol"
print "V0: ", popt[0]/(nA/Z/voltoa), "+/-", np.sqrt(pcov[0][0])/(nA/Z/voltoa), "A^3"
print "k0: ", popt[1]/1.e9, "+/-", np.sqrt(pcov[1][1])/1.e9, "GPa"
print "k0':", popt[2], "+/-", np.sqrt(pcov[2][2])
print "k0\":", -1.e9*popt[2]/popt[1], "GPa^-1"
print "a0 :", popt[3], "+/-", np.sqrt(pcov[3][3]), "K^-1"


Vdiff=np.empty_like(volumes)
for i, datum in enumerate(pt):
        hcp.set_state(datum[0], datum[1])
        Vdiff[i]=(hcp.V - volumes[i])/volumes[i]*100.

plt.plot( np.array(P), Vdiff , marker=".", linestyle="None")
plt.title('Volume fit')
plt.xlabel("Pressure (GPa)")
plt.ylabel("Percentage volume difference (m^3/mol)")
plt.show()



print

print "Covariance matrix:"
print pcov

'''
H_0 and S_0 for HCP can be obtained from the reaction line of Komabayashi et al., 2009
'''






'''


def fcc_hcp_eqm(arg, T):
    P=arg[0]
    hcp.set_state(P,T)
    fcc.set_state(P,T)
    return [hcp.gibbs - fcc.gibbs]


temperatures=np.linspace(400.,2400.,101)
hcp_fcc_pressures=np.empty_like(temperatures)

for idx, T in enumerate(temperatures):
    hcp_fcc_pressures[idx]=optimize.fsolve(fcc_hcp_eqm, 1.e9, args=(T))[0]


f=open('Fe-O_boundaries_int.dat', 'r')
datastream = f.read()  # We need to re-open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
phase_boundaries=np.array(datalines, np.float32).T
phase_boundaries[0]=[1.0-x_to_y(phase_boundaries[0][i]) for i in range(len(phase_boundaries[0]))]


plt.plot( hcp_fcc_pressures/1.e9, temperatures, 'r-', linewidth=3., label='HCP-FCC')


plt.title('Iron phase diagram')
plt.ylabel("Temperature (K)")
plt.xlabel("Pressure (GPa)")
plt.legend(loc='lower right')
plt.show()

'''
