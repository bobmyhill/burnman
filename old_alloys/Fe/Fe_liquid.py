# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

from scipy.interpolate import interp1d, UnivariateSpline, splrep, splev
from scipy.optimize import fsolve, curve_fit
import burnman
from HP_convert import *
from listify_xy_file import *
from fitting_functions import *


# Liquid thermodynamic properties

# Melting curve
# Solid phase(s): for aK_T, S, V at melting points
# Supplementary data for S and V at other pressures?
# Heat capacity for the liquid (as a function of pressure)

# Cp = Cv + R for ideal gases
# Cp = Cv + V*T*a*a*K_T
# gamma = a*K_T*V/Cv

'''
Next, we find the FCC ($\gamma$-iron) thermal equation of state from the data of Tsujino et al., 2013
They used the MgO pressure standard of Matsui et al., 2000
'''

Fe_fcc1 = burnman.minerals.Myhill_calibration_iron.fcc_iron()
Fe_fcc = burnman.minerals.Myhill_calibration_iron.fcc_iron_SLB()
Fe_fcc.set_state(1.e5, 298.15)
Fe_fcc1.set_state(1.e5, 298.15)
print Fe_fcc.gibbs, Fe_fcc1.gibbs 


fa = burnman.minerals.SLB_2011.fayalite()

temperatures = np.linspace(100., 1809., 101)
Cps = np.empty_like(temperatures)
Cps1 = np.empty_like(temperatures)
P = 1.e5
for i, T in enumerate(temperatures):
    Fe_fcc.set_state(P, T)
    Fe_fcc1.set_state(P, T)
    fa.set_state(P, T)

    Cps[i] = Fe_fcc.C_p
    Cps1[i] = Fe_fcc1.C_p
        
plt.plot(temperatures, Cps, label=str(P/1.e9)+' GPa')
plt.plot(temperatures, Cps1, label=str(P/1.e9)+' GPa')
plt.legend(loc='lower right')
plt.show()

nA=6.02214e23
voltoa=1.e30

fcc_data=[]
fcc_data_Nishihara=[]


for line in open('data/Tsujino_et_al_2013.dat'):
    content=line.strip().split()
    if content[0] != '%':
        fcc_data.append([float(content[3]), float(content[0]), float(content[1]), float(content[2])])


for line in open('data/Nishihara_et_al_2012_fcc_volumes.dat'):
    content=line.strip().split()
    if content[0] != '%':
        fcc_data.append([float(content[0]), float(content[1]), float(content[2]), float(content[3])])
        fcc_data_Nishihara.append([float(content[0]), float(content[1]), float(content[2]), float(content[3])])


'''
for line in open('data/Basinski_et_al_1955_fcc_volumes_RP.dat'):
    content=line.strip().split()
    if content[0] != '%':
        fcc_data.append([1.e-4, float(content[0]), float(content[1])*4., 0.001])
'''

#Temperature K  Volume of y-Fe angstrom^3  Pressure GPa  V/V0 of MgO angstrom^3 
P, T, V, Verr = zip(*fcc_data)

print V
Z=4.
volumes=np.array(V)*nA/voltoa/Z
sigmas=np.array(Verr)*nA/voltoa/Z
pt=[np.array(P)*1.e9, T]


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


#guesses=[6.94e-06, 151.e9, 5.6, 2., 1.]
#popt, pcov = optimize.curve_fit(fit_EoS_data(Fe_fcc, ['V_0', 'K_0', 'Kprime_0', 'grueneisen_0', 'q_0']), pt, volumes, guesses, sigmas)
#print 'FCC parameters', popt

Fe_hcp = burnman.minerals.Myhill_calibration_iron.hcp_iron_SLB()

Z=2.
nA=6.02214e23
voltoa=1.e30

hcp_data=[]
for line in open('data/Yamazaki_just_data.dat'):
    content=line.translate(None, ')(').strip().split()
    if content[0] != '%':
        hcp_data.append(map(float, content))

# T (K) V (Au) P a-axis (A) c-axis (A) V (e-Fe [HCP] angstroms^3)
T_Y, VAu_Y, VAuerr_Y, P_Y, Perr_Y, a_Y, aerr_Y, c_Y, cerr_Y, V_Y, Verr_Y = zip(*hcp_data)


hcp_data=[]

for line in open('data/Komabayashi_et_al_2009_HCP_volumes.dat'):
    content=line.strip().split()
    if content[0] != '%':
        hcp_data.append([float(content[2]), float(content[0]), float(content[4]), float(content[5])])

T, P, V, Verr = zip(*hcp_data)

hcp_data=[]
err_scaling=1.0
for line in open('data/Dewaele_et_al_2006.dat'):
    content=line.translate(None, ')(').strip().split()
    if content[0] != '%':
        hcp_data.append([float(content[0]), float(content[1]), float(content[2])*Z, float(content[3])*Z*err_scaling])

# T (K) V (Au) P a-axis (A) c-axis (A) V (e-Fe [HCP] angstroms^3)
T_D, P_D, V_D, Verr_D = zip(*hcp_data)


volumes=np.array(V_D+V)*(nA/Z/voltoa)
sigma=np.array(Verr_D+Verr)*(nA/Z/voltoa)
pt=np.array([np.array(P_D+P)*1.e9, np.array(T_D+T)])

# Initial guess.
#guesses=[6.753e-6, 166.e9, 5., 1.9, 1.0]
#popt, pcov = optimize.curve_fit(fit_EoS_data(Fe_hcp, ['V_0', 'K_0', 'Kprime_0', 'grueneisen_0', 'q_0']), pt, volumes, guesses)
#print popt


# FIND EQUILIBRIUM PRESSURES

Fe_fcc.set_state(1.e5, 1200.)
print Fe_fcc.S
print Fe_fcc.C_p

print 'hi'
temperatures = np.linspace(800., 3000., 101)
pressures = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    pressures[i] = fsolve(eqm_pressure([Fe_fcc, Fe_hcp], [1.0, -1.0]), [1.e9], args=(T))[0]

Fe_fcc.set_state(50.e9, 2000.)
Fe_hcp.set_state(50.e9, 2000.)
print Fe_fcc.gibbs - Fe_hcp.gibbs
print Fe_fcc.V, Fe_hcp.V

plt.plot(pressures/1.e9, temperatures)
plt.show()
exit()


pressures = np.linspace(100.e9, 200.e9, 101)
volumes = np.empty_like(pressures)
for T in [1000., 3000., 8000.]:
    for i, P in enumerate(pressures):
        Fe_hcp.set_state(P, T)
        volumes[i] = Fe_hcp.V
        
    plt.plot(pressures/1.e9, volumes, label=str(T)+' K')

plt.plot(np.array(P_D), np.array(V_D)*(nA/Z/voltoa), marker='o', linestyle='None')
plt.legend(loc='upper right')
plt.show()





Fe_fcc = burnman.minerals.Myhill_calibration_iron.fcc_iron()
Fe_hcp = burnman.minerals.Myhill_calibration_iron.hcp_iron()

# Convert solid phase parameters to HP 
P_ref = 200.e9
T_ref = 1809.
HP_convert(Fe_fcc, 300., 2200., T_ref, P_ref)
HP_convert(Fe_hcp, 300., 2200., T_ref, P_ref)

temperatures = np.linspace(300., 4000., 101)
Cps = np.empty_like(temperatures)

for P in [1.e9, 100.e9]:
    for i, T in enumerate(temperatures):
        Fe_hcp.set_state(P, T)
        Cps[i] = Fe_hcp.C_p
        
    plt.plot(temperatures, Cps)

plt.ylim(0., 100.)
plt.show()

class liquid():
    def __init__(self, PT_melting, phases, Cp):
        # splrep seems to give a good tradeoff between smoothness and 
        # weird minima and maxima
        # could also use UnivariateSpline or interp1d?
        self.melting_curve = splrep(PT_melting[0], 
                                    PT_melting[1], 
                                    s=0)
        self.phases = phases
        self.Cp = Cp

    def set_state(self, pressure, temperature):
        for phase, max_pressure in self.phases:
            if pressure < max_pressure:
                self.liquidus_phase = phase
                break

        #self.Tmelt = self.melting_curve(pressure)
        self.T_melt = splev(pressure, self.melting_curve, der=0)
        dP = 100. # Pa
        dT = splev(pressure + dP/2., self.melting_curve, der=0) \
            - splev(pressure - dP/2., self.melting_curve, der=0)
        dTdP = dT/dP
        self.liquidus_phase.set_state(pressure, self.T_melt)
        aK_T = self.liquidus_phase.alpha*self.liquidus_phase.K_T
        Sfusion = burnman.constants.gas_constant*np.log(2.) / (1. - aK_T*dTdP)
        Vfusion = Sfusion*dTdP

        print self.liquidus_phase.S, Sfusion

        Smelt = self.liquidus_phase.S + Sfusion
        Vmelt = self.liquidus_phase.V + Vfusion

        #DS = \int Cp/T dT
        deltaS = self.Cp*(np.log(self.T_melt) - np.log(temperature))
        self.S = Smelt - deltaS

        # Dgibbs = \int S dT
        deltagibbs = -self.Cp*(self.T_melt*(np.log(self.T_melt) - 1.) - temperature*(np.log(temperature) - 1.))
        self.gibbs = self.liquidus_phase.gibbs - deltagibbs
        

melting_curve_data = listify_xy_file('data/Anzellini_2013_Fe_melting_curve.dat')
melting_curve_data[0] = melting_curve_data[0]*1.e9
Fe_liq = liquid(melting_curve_data, [[Fe_fcc, 98.e9], [Fe_hcp, 300.e9]], 46.024)

pressures = np.linspace(1.e9, 200.e9, 101)
temperatures = np.empty_like(pressures)
gibbs = np.empty_like(pressures)
entropy = np.empty_like(pressures)

T = 2000.
for i, P in enumerate(pressures):
    Fe_liq.set_state(P, T)
    temperatures[i] = Fe_liq.T_melt
    Fe_liq.set_state(P, Fe_liq.T_melt)
    gibbs[i] = Fe_liq.gibbs
    entropy[i] = Fe_liq.S


plt.plot(pressures, temperatures)
plt.plot(melting_curve_data[0], melting_curve_data[1], marker='o', linestyle='None')
plt.show()



plt.plot(pressures, entropy)
plt.show()
plt.plot(pressures, gibbs)
plt.show()

