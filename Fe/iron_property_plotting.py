# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman import tools
from burnman.mineral import Mineral
from burnman.minerals import Myhill_calibration_iron
from burnman.processchemistry import *
from burnman.chemicalpotentials import *
from burnman import constants

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import optimize
atomic_masses=read_masses()

nA=6.02214e23
voltoa=1.e30
Pr=1.e5

'''
First, let's use the description of bcc and fcc iron from Sundman, 1991
N.B.: Saxena and Dubrovinsky, 1998 (Geophysical Monograph 101) have a similar expression, but this is basically the same as Sundman's, with a few numbers tweaked badly (there is a discontinuity in Gbcc at 1811 K) and insufficient data in the paper (no reference to the magnetic contribution in FCC).
'''

bcc=Myhill_calibration_iron.bcc_iron()
fcc=Myhill_calibration_iron.fcc_iron_HP()
hcp=Myhill_calibration_iron.hcp_iron_HP()



temperatures=np.linspace(300., 2000., 101)
bcc_gibbs=np.empty_like(temperatures)
fcc_gibbs=np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    bcc.set_state(Pr, T)
    fcc.set_state(Pr, T)
    fcc_gibbs[i] = fcc.gibbs
    bcc_gibbs[i] = bcc.gibbs

plt.plot( temperatures, bcc_gibbs - fcc_gibbs, 'r-', linewidth=3., label='bcc - fcc (Sundman, 1991)')
plt.title('1 bar iron model')
plt.xlabel("Temperature (K)")
plt.ylabel("Gibbs free energy difference (J/mol)")
plt.show()


def magnetic_gibbs(T, Tc, beta, p):
    A = (518./1125.) + (11692./15975.)*((1./p) - 1.)
    tau=T/Tc
    if tau < 1: 
        f=1.-(1./A)*(79./(140.*p*tau) + (474./497.)*(1./p - 1.)*(np.power(tau, 3.)/6. + np.power(tau, 9.)/135. + np.power(tau, 15.)/600.))
    else:
        f=-(1./A)*(np.power(tau,-5)/10. + np.power(tau,-15)/315. + np.power(tau, -25)/1500.)
    return constants.gas_constant*T*np.log(beta + 1.)*f


def gibbs_bcc_1bar_SD1998(T):
    if T < 1811:
        gibbs= 1400. + 124.06*T - 23.5143*T*np.log(T) -0.00439752*T*T - 5.8927e-8*T*T*T + 77359./T 
    else:
        gibbs= - 25383.581 + 299.31255*T - 46.*T*np.log(T) + 2.29603e31*np.power(T,-9.)
    Tc=1043.
    beta=2.22
    p=0.4
    return gibbs + magnetic_gibbs(T, Tc, beta, p)

def bcc_hcp_eqm(T):
    bcc.set_state(Pr, T)
    fcc.set_state(Pr, T)
    return bcc.gibbs - fcc.gibbs

print 'fcc stable between', optimize.fsolve(bcc_hcp_eqm, 1184.)[0], 'and', optimize.fsolve(bcc_hcp_eqm, 1668.)[0], 'K'


'''
Next, we find the FCC ($\gamma$-iron) thermal equation of state from the data of Tsujino et al., 2013
They used the MgO pressure standard of Matsui et al., 2000
'''

fcc_data=[]
fcc_data_Nishihara=[]
'''
for line in open('data/Tsujino_et_al_2013.dat'):
    content=line.strip().split()
    if content[0] != '%':
        fcc_data.append(map(float, content))

#Temperature K  Volume of y-Fe angstrom^3  Pressure GPa  V/V0 of MgO angstrom^3 
T, V, Verr, P, Perr, relVMgO, RelVMgOerr = zip(*fcc_data)

Z=4.
volumes=np.array(V)*nA/voltoa/Z
pt=np.array(zip(np.array(P)*1.e9, T))
'''

scaling=1.0
for line in open('data/Nishihara_et_al_2012_fcc_volumes.dat'):
    content=line.strip().split()
    if content[0] != '%':
        fcc_data.append([float(content[0]), float(content[1]), float(content[2])*scaling, float(content[3])])
        fcc_data_Nishihara.append([float(content[0]), float(content[1]), float(content[2])*scaling, float(content[3])])


for line in open('data/Basinski_et_al_1955_fcc_volumes_RP.dat'):
    content=line.strip().split()
    if content[0] != '%':
        fcc_data.append([1.e-4, float(content[0]), float(content[1])*4., 0.001])


#Temperature K  Volume of y-Fe angstrom^3  Pressure GPa  V/V0 of MgO angstrom^3 
P, T, V, Verr = zip(*fcc_data)

Z=4.
volumes=np.array(V)*nA/voltoa/Z
sigmas=np.array(Verr)*nA/voltoa/Z
pt=np.array(zip(np.array(P)*1.e9, T))


'''
for line in open('data/Boehler_fcc_volumes.dat'):
    content=line.strip().split()
    if content[0] != '%':
        fcc_data.append(map(float, content))


# Pressure (kbar), Temperature [K]  Volume of y-Fe (cm^3/mol)
P, T, V = zip(*fcc_data)

Z=4.
volumes=np.array(V)*1.e-6
pt=np.array(zip(np.array(P)*1.e8, T))
'''

# Initial guess.
def fitV(mineral):
    def fit(data, V_0, K_0, a_0):
        mineral.params['V_0'] = V_0
        mineral.params['K_0'] = K_0
        mineral.params['Kdprime_0'] = -mineral.params['Kprime_0']/mineral.params['K_0']
        mineral.params['a_0'] = a_0
        
        vols=[]
        for i, datum in enumerate(data):
            mineral.set_state(datum[0], datum[1])
            vols.append(mineral.V)
        return vols
    return fit

def fitV_wout_a0(mineral):
    def fit(data, V_0, K_0):
        mineral.params['V_0'] = V_0
        mineral.params['K_0'] = K_0
        mineral.params['Kdprime_0'] = -mineral.params['Kprime_0']/mineral.params['K_0']
        
        vols=[]
        for i, datum in enumerate(data):
            mineral.set_state(datum[0], datum[1])
            vols.append(mineral.V)
        return vols
    return fit

def fitV_w_Kprime_wout_a0(mineral):
    def fit(data, V_0, K_0, Kprime_0):
        mineral.params['V_0'] = V_0
        mineral.params['K_0'] = K_0
        mineral.params['Kprime_0'] = Kprime_0
        mineral.params['Kdprime_0'] = -mineral.params['Kprime_0']/mineral.params['K_0']
        
        vols=[]
        for i, datum in enumerate(data):
            mineral.set_state(datum[0], datum[1])
            vols.append(mineral.V)
        return vols
    return fit

def fitV_a0(mineral):
    def fit(data, V_0, a_0):
        mineral.params['V_0'] = V_0
        mineral.params['a_0'] = a_0
        mineral.params['Kdprime_0'] = -mineral.params['Kprime_0']/mineral.params['K_0']
        
        vols=[]
        for i, datum in enumerate(data):
            mineral.set_state(datum[0], datum[1])
            vols.append(mineral.V)
        return vols
    return fit

def fitV_w_Kprime(mineral):
    def fit(data, V_0, K_0, Kprime_0, a_0):
        mineral.params['V_0'] = V_0 
        mineral.params['K_0'] = K_0
        mineral.params['Kprime_0'] = Kprime_0
        mineral.params['Kdprime_0'] = -mineral.params['Kprime_0']/mineral.params['K_0']
        mineral.params['a_0'] = a_0
        
        vols=[]
        for i, datum in enumerate(data):            
            mineral.set_state(datum[0], datum[1])
            vols.append(mineral.V)
        return vols
    return fit

def fitV_T0(mineral):
    def fit(pressures, V_0, K_0, Kprime_0):
        mineral.params['V_0'] = V_0 
        mineral.params['K_0'] = K_0
        mineral.params['Kprime_0'] = Kprime_0
        mineral.params['Kdprime_0'] = -mineral.params['Kprime_0']/mineral.params['K_0']
        
        vols=[]
        for pressure in pressures:
            mineral.set_state(pressure, 298.15)
            vols.append(mineral.V)
        return vols
    return fit

'''
guesses=np.array([fcc.params['V_0'], fcc.params['K_0'], fcc.params['a_0']])

popt, pcov = optimize.curve_fit(fitV(fcc), pt, volumes, guesses, sigmas)

bcc.set_state(1.e5, 1273)
print 'BCC volume at 1273 K:', bcc.V/(nA/Z/voltoa)

print ''
print 'Fitted FCC parameters'
print "V0: ", popt[0], "+/-", np.sqrt(pcov[0][0]), "m^3/mol"

print "V0: ", popt[0]/(nA/Z/voltoa), "+/-", np.sqrt(pcov[0][0])/(nA/Z/voltoa), "A^3"
print "k0: ", popt[1]/1.e9, "+/-", np.sqrt(pcov[1][1])/1.e9, "GPa"
print "k0':", fcc.params['Kprime_0'], '[fixed]'
print "k0\":", -1.e9*fcc.params['Kprime_0']/popt[1], "GPa^-1"
print "a0 :", popt[2], "+/-", np.sqrt(pcov[2][2]), "K^-1"
'''

bcc_expt_temperatures=[]
bcc_expt_volumes=[]

'''
bcc.params['a_0'] = 3.8e-5
'''

for line in open('data/Stuart_et_al_1966_bcc_iron.dat'):
    content=line.strip().split()
    if content[0] != '%':
        bcc_expt_temperatures.append(float(content[0]))
        bcc_expt_volumes.append(float (content[1])/2.)

'''
for line in open('data/bcc_iron_volumes.dat'):
    content=line.strip().split()
    if content[0] != '%':
        bcc_expt_temperatures.append(float(content[0]))
        bcc_expt_volumes.append(float (content[1]))
'''

fcc_expt_temperatures=[]
fcc_expt_volumes=[]
for line in open('data/Basinski_et_al_1955_fcc_volumes_RP.dat'):
    content=line.strip().split()
    if content[0] != '%':
        fcc_expt_temperatures.append(float(content[0]))
        fcc_expt_volumes.append(float(content[1]))


temperatures=np.linspace(50., 2000., 101)
bcc_volume=np.empty_like(temperatures)
fcc_volume=np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    bcc.set_state(Pr, T)
    fcc.set_state(Pr, T)
    fcc_volume[i] = fcc.V
    bcc_volume[i] = bcc.V

plt.plot( fcc_expt_temperatures, fcc_expt_volumes, c='r', marker='o', linestyle='none', label='fcc volumes')
plt.plot( bcc_expt_temperatures, bcc_expt_volumes, c='b', marker='o', linestyle='none', label='bcc volumes')

plt.plot( temperatures, fcc_volume/nA*voltoa, 'r-', linewidth=1., label='fcc volume')
plt.plot( temperatures, bcc_volume/nA*voltoa, 'b-', linewidth=1., label='bcc volume')
plt.title('1 bar iron model')
plt.legend(loc='lower right')
plt.xlabel("Temperature (C)")
plt.ylabel("Volume (Angstroms^3/mol)")
plt.show()

Vdiff=np.empty_like(volumes)
for i, datum in enumerate(pt):
        fcc.set_state(datum[0], datum[1])
        Vdiff[i]=(fcc.V - volumes[i])/volumes[i]*100.

plt.plot( np.array(P), Vdiff , marker=".", linestyle="None")


P, T, V, Verr = zip(*fcc_data_Nishihara)
Z=4.
volumes=np.array(V)*nA/voltoa/Z
sigmas=np.array(Verr)*nA/voltoa/Z
pt=np.array(zip(np.array(P)*1.e9, T))

Vdiff_Nishihara=np.empty_like(volumes)
for i, datum in enumerate(pt):
        fcc.set_state(datum[0], datum[1])
        Vdiff_Nishihara[i]=(fcc.V - volumes[i])/volumes[i]*100.
plt.plot( np.array(P), Vdiff_Nishihara , marker=".", linestyle="None")
plt.title('Volume fit')
plt.xlabel("Pressure (GPa)")
plt.ylabel("Percentage volume difference (m^3/mol)")
plt.show()




'''
Now, we find the HCP ($\varepsilon$-iron thermal equation of state from 
They used Au as a pressure standard (equation of state from Tsuchiya, 2003)
'''

Z=2.

hcp_data=[]
for line in open('data/Yamazaki_et_al_2012.dat'):
    content=line.translate(None, ')(').strip().split()
    if content[0] != '%':
        hcp_data.append(map(float, content))

# T (K) V (Au) P a-axis (A) c-axis (A) V (e-Fe [HCP] angstroms^3)
T, VAu, VAuerr, P, Perr, a, aerr, c, cerr, V, Verr = zip(*hcp_data)

hcp_data=[]
for line in open('data/Dewaele_et_al_2006.dat'):
    content=line.translate(None, ')(').strip().split()
    if content[0] != '%':
        hcp_data.append([float(content[0]), float(content[1]), float(content[2])*Z, float(content[3])*Z])

# T (K) V (Au) P a-axis (A) c-axis (A) V (e-Fe [HCP] angstroms^3)
T_D, P_D, V_D, Verr_D = zip(*hcp_data)


volumes=np.array(V + V_D)*(nA/Z/voltoa)
sigma=np.array(Verr + Verr_D)*(nA/Z/voltoa)
pt=np.array(zip(np.array(P + P_D)*1.e9, np.array(T+T_D)))

'''
# Initial guess.
guesses=np.array([hcp.params['V_0'], hcp.params['K_0'], hcp.params['Kprime_0'], hcp.params['a_0']])
popt, pcov = optimize.curve_fit(fitV_w_Kprime(hcp), pt, volumes, guesses, sigma)
'''

'''
print ''
print 'Fitted HCP parameters'
print "V0: ", popt[0], "+/-", np.sqrt(pcov[0][0]), "m^3/mol"
print "V0: ", popt[0]/(nA/Z/voltoa), "+/-", np.sqrt(pcov[0][0])/(nA/Z/voltoa), "A^3"
print "k0: ", popt[1]/1.e9, "+/-", np.sqrt(pcov[1][1])/1.e9, "GPa"
print "k0':", popt[2], "+/-", np.sqrt(pcov[2][2])
print "k0\":", -1.e9*popt[2]/popt[1], "GPa^-1"
print "a0 :", popt[3], "+/-", np.sqrt(pcov[3][3]), "K^-1"
'''


volumes=np.array(V_D)*(nA/Z/voltoa)
sigma=np.array(Verr_D)*(nA/Z/voltoa)
p=np.array(P_D)*1.e9

'''
# Initial guess.
guesses=np.array([hcp.params['V_0'], hcp.params['K_0'], hcp.params['Kprime_0']])
popt, pcov = optimize.curve_fit(fitV_T0(hcp), p, volumes, guesses, sigma)

print ''
print 'Fitted HCP parameters (Dewaele et al., 2006)'
print "V0: ", popt[0], "+/-", np.sqrt(pcov[0][0]), "m^3/mol"
print "V0: ", popt[0]/(nA/Z/voltoa), "+/-", np.sqrt(pcov[0][0])/(nA/Z/voltoa), "A^3"
print "k0: ", popt[1]/1.e9, "+/-", np.sqrt(pcov[1][1])/1.e9, "GPa"
print "k0':", popt[2], "+/-", np.sqrt(pcov[2][2])
print "k0\":", -1.e9*popt[2]/popt[1], "GPa^-1"
'''


Vdiff=np.empty_like(volumes)
for i, pressure in enumerate(p):
        hcp.set_state(pressure, 298.15)
        Vdiff[i]=(hcp.V - volumes[i])/volumes[i]*100.

plt.plot( np.array(P_D), Vdiff , marker=".", linestyle="None")
plt.title('Volume fit')
plt.xlabel("Pressure (GPa)")
plt.ylabel("Percentage volume difference (m^3/mol)")
plt.show()

'''
print "Covariance matrix:"
print pcov
'''

'''
H_0 and S_0 for HCP can be obtained from the reaction line of Komabayashi et al., 2009
'''

def find_pressure(mineral):
    def pressure(arg, volume, T):
        P=arg[0]
        mineral.set_state(P, T) 
        return mineral.V - volume
    return pressure



hcp_fcc_data=[]
for line in open('data/Komabayashi_et_al_2009_HCP_FCC.dat'):
    content=line.translate(None, ')(').strip().split()
    if content[0] != '%':
        hcp_fcc_data.append([float(content[0]), float(content[1]), float(content[2]), float(content[3]), float(content[4]), float(content[5]), content[6]])

P, Perr, T, Terr, V, Verr, note = zip(*hcp_fcc_data)
transition_temperatures=np.array(T)
transition_temperature_uncertainties=np.array(Terr)
transition_volumes=np.array(V)*(nA/Z/voltoa)

def equilibrium_boundary_T(mineral1, mineral2):
    def eqm(arg, P):
        T=arg[0]
        mineral1.set_state(P,T)
        mineral2.set_state(P,T)
        return [mineral1.gibbs - mineral2.gibbs]
    return eqm

def equilibrium_boundary_P(mineral1, mineral2):
    def eqm(arg, T):
        P=arg[0]
        mineral1.set_state(P,T)
        mineral2.set_state(P,T)
        return [mineral1.gibbs - mineral2.gibbs]
    return eqm
'''

def fit_H_S(mineral1, mineral2):
    def find_H_S(data, a, K):

        mineral1.params['a_0']= a
        mineral2.params['K_0']= K
        calc_temperatures=[]
        for datum in data:
            volume=datum[0]
            temperature=datum[1]
            pressure = optimize.fsolve(find_pressure(hcp), 1.e9, args=(volume, temperature))[0]
            calc_temperatures.append(optimize.fsolve(equilibrium_boundary_T(mineral1, mineral2), 1000., args=(pressure))[0])
        return calc_temperatures
    return find_H_S

print ''
print 'FCC parameters'
print "H0: ", fcc.params['H_0'], "J/mol"
print "S0: ", fcc.params['S_0'], "J/K/mol"


hcp.params['S_0']= 30.7 # To match BCC-HCP phase boundary
guesses=np.array([hcp.params['a_0'], fcc.params['K_0']])
popt, pcov = optimize.curve_fit(fit_H_S(hcp, fcc), np.array([transition_volumes, transition_temperatures]).T, transition_temperatures, guesses, transition_temperature_uncertainties)


print ''
print 'Fitted HCP parameters'
print "H0: ", popt[0], "+/-", np.sqrt(pcov[0][0]), "J/mol"
print "S0: ", popt[1], "+/-", np.sqrt(pcov[1][1]), "J/K/mol"
print popt
'''
'''
hcp.params['V_0']=11.214*(nA/voltoa)
print hcp.params['V_0']
hcp.params['K_0']=163.4e9
hcp.params['Kprime_0']=5.38
'''

transition_pressures=np.empty_like(T)
for i, temperature in enumerate(transition_temperatures):
    transition_pressures[i] = optimize.fsolve(find_pressure(hcp), 1.e9, args=(transition_volumes[i], transition_temperatures[i]))[0]


hcp_fcc_pressures=np.linspace(10.e9, 70.e9, 101)
hcp_fcc_temperatures=np.empty_like(hcp_fcc_pressures)
for i, pressure in enumerate(hcp_fcc_pressures):
    hcp_fcc_temperatures[i]=optimize.fsolve(equilibrium_boundary_T(hcp, fcc), 1000., args=(pressure))[0]

bcc_fcc_pressures=np.linspace(1.e5, 10.e9, 21)
bcc_fcc_temperatures=np.empty_like(bcc_fcc_pressures)
for i, pressure in enumerate(bcc_fcc_pressures):
    bcc_fcc_temperatures[i]=optimize.fsolve(equilibrium_boundary_T(bcc, fcc), 1000., args=(pressure))[0]


bcc_hcp_temperatures=np.linspace(300., 870., 21)
bcc_hcp_pressures=np.empty_like(bcc_hcp_temperatures)
for i, temperature in enumerate(bcc_hcp_temperatures):
    bcc_hcp_pressures[i]=optimize.fsolve(equilibrium_boundary_P(bcc, hcp), 1000., args=(temperature))[0]



'''
Almost finished! Let's plot the modelled FCC-HCP transition along with the experimental data, to
see how well the inversion worked.
'''

in_pressures=[]
in_temperatures=[]
in_temperature_error=[]
out_pressures=[]
out_temperatures=[]
out_temperature_error=[]
for pressure, temperature, temperature_error, label in zip(transition_pressures, transition_temperatures, transition_temperature_uncertainties, note):
    if label=='field-in':
        in_pressures.append(pressure)
        in_temperatures.append(temperature)
        in_temperature_error.append(temperature_error)
    else:
        out_pressures.append(pressure)
        out_temperatures.append(temperature)
        out_temperature_error.append(temperature_error)

in_pressures=np.array(in_pressures)
in_temperatures=np.array(in_temperatures)
in_temperature_error=np.array(in_temperature_error)
out_pressures=np.array(out_pressures)
out_temperatures=np.array(out_temperatures)
out_temperature_error=np.array(out_temperature_error)

plt.plot( bcc_fcc_pressures/1.e9, bcc_fcc_temperatures, 'b-', linewidth=1, label='BCC-FCC transition')
plt.plot( bcc_hcp_pressures/1.e9, bcc_hcp_temperatures, 'b-', linewidth=1, label='BCC-HCP transition')
plt.plot( hcp_fcc_pressures/1.e9, hcp_fcc_temperatures, 'b-', linewidth=1, label='FCC-HCP transition')
plt.plot( in_pressures/1.e9, in_temperatures, marker=".", linestyle="None", label='in')
plt.errorbar( in_pressures/1.e9, in_temperatures, yerr=in_temperature_error, linestyle="None")
plt.plot( out_pressures/1.e9, out_temperatures, marker="+", linestyle="None", label='out')
plt.errorbar( out_pressures/1.e9, out_temperatures, yerr=out_temperature_error, linestyle="None")
plt.title('Iron phase diagram')
plt.ylabel("Temperature (K)")
plt.xlabel("Pressure (GPa)")
plt.ylim(300., 2700.)
plt.legend(loc='lower right')
plt.show()

'''
Finally, let's print our updated mineral classes
'''
'''
tools.print_mineral_class(fcc, 'fcc_iron')
tools.print_mineral_class(hcp, 'hcp_iron')
'''
