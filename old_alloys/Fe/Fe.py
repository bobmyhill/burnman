import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import Komabayashi_2014, Myhill_calibration_iron, SLB_2011
import numpy as np
from scipy import optimize, integrate
import matplotlib.pyplot as plt

def invariant(data):
    P, T = data
    return [Fe_hcp.calcgibbs(P, T) -  Fe_fcc.calcgibbs(P, T),
            Fe_hcp.calcgibbs(P, T) -  Fe_liq.calcgibbs(P, T)]
        
def eqm_pressure(minerals, multiplicities):
    def eqm(P, T):
        gibbs = 0.
        for i, mineral in enumerate(minerals):
            gibbs = gibbs + mineral.calcgibbs(P[0], T)*multiplicities[i]
        return gibbs
    return eqm

def eqm_temperature(minerals, multiplicities):
    def eqm(T, P):
        gibbs = 0.
        for i, mineral in enumerate(minerals):
            gibbs = gibbs + mineral.calcgibbs(P, T[0])*multiplicities[i]
        return gibbs
    return eqm

# READ IN DATA
Fe_melting = []

f=open('data/Fe_melting_Anzellini_2003.dat', 'r')
datastream = f.read()  # We need to open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] != "%":
        Fe_melting.append([float(line[0])*1.e9, 
                           float(line[1]), 
                           float(line[2])])

P, T, Terr = zip(*Fe_melting)
P_obs = np.array(P)
T_obs = np.array(T)
PT_obs = zip(*[P, T])


Fe_fcc=Myhill_calibration_iron.fcc_iron()
Fe_hcp=Myhill_calibration_iron.hcp_iron()
Fe_liq=Myhill_calibration_iron.liquid_iron()


from HP_convert import *
HP_convert(Fe_fcc, 300., 2200., 1809., 50.e9)
HP_convert(Fe_hcp, 300., 2200., 1809., 50.e9)
HP_convert(Fe_liq, 1809., 2400., 1809., 50.e9)

print 'converted'

ass = [Fe_fcc, Fe_hcp, Fe_liq]
for mineral in ass:
    print mineral.params['name']
    mineral.set_state(1.e5, 3800.)
    print mineral.gibbs

Fe_liq.set_state(1.e5, 3000.)
K0=Fe_liq.K_T
print Fe_liq.H, Fe_liq.S, Fe_liq.V, Fe_liq.alpha, Fe_liq.K_T
Fe_liq.set_state(1.e5+1., 3000.)
K1=Fe_liq.K_T
print K1-K0

#Fe_liq.params['V_0'] = 7.2e-6
#Fe_liq.params['a_0'] = 65.e-6 # Nasch and Manghani at 1550 C (extrapolated to avoid magnetic effects)
#Fe_liq.params['K_0'] = 147.e9
# Uncorrected values from Nasch and Manghani (1998), to fit properties at 1809 K.
'''
Fe_liq.params['V_0'] = 7.068e-6  # Komabayashi = 6.88
Fe_liq.params['K_0'] = 156.e9  # Komabayashi = 148
Fe_liq.params['Kprime_0'] = 5.8 # Komabayashi = 5.8
Fe_liq.params['a_0'] = 8.2e-5 # Komabayashi = 9
Fe_liq.params['delta_0'] = 6.3 # Komabayashi = 5.1
Fe_liq.params['kappa'] = 0.56 # Komabayashi = 0.56
'''
# Nonmagnetic values from Nasch and Manghani (1998), to fit properties at 1809 K.
# Results in liquid volumes that are much too small to fit the melting curve
Fe_liq.set_state(1.e5, 1809.)
print 'H', Fe_liq.H, Fe_liq.S
'''
Fe_liq.params['V_0'] = 6.25e-6  # Komabayashi = 6.88
Fe_liq.params['K_0'] = 156.e9  # Komabayashi = 148
Fe_liq.params['Kprime_0'] = 5.8 # Komabayashi = 5.8
Fe_liq.params['a_0'] = 8.2e-5 # Komabayashi = 9
Fe_liq.params['delta_0'] = 6.3 # Komabayashi = 5.1
Fe_liq.params['kappa'] = 0.56 # Komabayashi = 0.56

Fe_liq.set_state(100.e9, 3000.)
Fe_hcp.set_state(100.e9, 3000.)
print Fe_liq.V, Fe_hcp.V
'''

Fe_liq.set_state(1.e5, 1809.)
print 'BEFORE'
print Fe_liq.S, 55.845*1.e-6/Fe_liq.V, Fe_liq.alpha, Fe_liq.K_T/1.e9, Fe_liq.gr


P_inv, T_inv = optimize.fsolve(invariant, [50.e9, 4000.])


pressures_fcc_liq = np.linspace(1.e9, P_inv, 50)
temperatures_fcc_liq = np.empty_like(pressures_fcc_liq)
for i, P in enumerate(pressures_fcc_liq):
    temperatures_fcc_liq[i] = optimize.fsolve(eqm_temperature([Fe_fcc, Fe_liq], [1., -1.]), [2000.], args=(P))[0]


pressures_hcp_liq = np.linspace(P_inv, 400.e9, 50)
temperatures_hcp_liq = np.empty_like(pressures_hcp_liq)
for i, P in enumerate(pressures_hcp_liq):
    temperatures_hcp_liq[i] = optimize.fsolve(eqm_temperature([Fe_hcp, Fe_liq], [1., -1.]), [2000.], args=(P))[0]
    

temperatures_fcc_hcp = np.linspace(1000., T_inv)
pressures_fcc_hcp = np.empty_like(temperatures_fcc_hcp)
for i, T in enumerate(temperatures_fcc_hcp):
    pressures_fcc_hcp[i] = optimize.fsolve(eqm_pressure([Fe_fcc, Fe_hcp], [1., -1.]), [50.e9], args=(T))[0]


plt.plot(P_obs/1.e9, T_obs, marker='.', linestyle='None')
plt.plot(pressures_hcp_liq/1.e9, temperatures_hcp_liq, label='hcp')
plt.plot(pressures_fcc_liq/1.e9, temperatures_fcc_liq, label='fcc')
plt.plot(pressures_fcc_hcp/1.e9, temperatures_fcc_hcp, label='fcc-hcp')

plt.legend(loc='lower left')
plt.show()

P = 100.e9

Fe_fcc.set_state(P, 3000.)
Fe_hcp.set_state(P, 3000.)

Gdiff = Fe_fcc.gibbs - Fe_hcp.gibbs
if Gdiff < 0.:
    print 'fcc'
else:
    print 'hcp'


print Fe_liq.params

'''
def fit_melting_curve(data, V_0, K_0, a_0):
    Fe_liq.params['V_0'] = V_0
    Fe_liq.params['K_0'] = K_0
    Fe_liq.params['a_0'] = a_0
    temperatures = []
    for datum in data:
        P, T = datum # temperature is just to work out whether the phase is hcp or fcc

        Gdiff = Fe_fcc.calcgibbs(P,T) - Fe_hcp.calcgibbs(P,T)
        if P < 120.e9:
            Fe_phase = Fe_fcc
        else:
            Fe_phase = Fe_hcp

        temperatures.append(optimize.fsolve(eqm_temperature([Fe_phase, Fe_liq], [1., -1.]), [T], args=(P))[0])

    return temperatures

guesses = [Fe_liq.params['V_0'], Fe_liq.params['K_0'], Fe_liq.params['a_0']]
popt, pcov = optimize.curve_fit(fit_melting_curve, PT_obs, T_obs, guesses, Terr)
print popt, pcov



'''
# PLOT AGAIN

pressures = np.linspace(1.e9, 500.e9, 50)
temperatures_fcc = np.empty_like(pressures)
temperatures_hcp = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures_hcp[i] = optimize.fsolve(eqm_temperature([Fe_hcp, Fe_liq], [1., -1.]), [2000.], args=(P))[0]
    temperatures_fcc[i] = optimize.fsolve(eqm_temperature([Fe_fcc, Fe_liq], [1., -1.]), [2000.], args=(P))[0]

temperatures_fcc_hcp = np.linspace(1000., 4000.)
pressures_fcc_hcp = np.empty_like(temperatures_fcc_hcp)
for i, T in enumerate(temperatures_fcc_hcp):
    pressures_fcc_hcp[i] = optimize.fsolve(eqm_pressure([Fe_fcc, Fe_hcp], [1., -1.]), [50.e9], args=(T))[0]


pressures = np.linspace(1.e9, 500.e9, 50)
for i, P in enumerate(pressures):
    Fe_liq.set_state(P, temperatures_hcp[i])
    Fe_hcp.set_state(P, temperatures_hcp[i])
    print P/1.e9, temperatures_hcp[i], 'liq K:', Fe_liq.K_T, 'hcp K:', Fe_hcp.K_T

'''
plt.plot(P_obs/1.e9, T_obs, marker='.', linestyle='None')
plt.plot(pressures/1.e9, temperatures_hcp, label='hcp')
plt.plot(pressures/1.e9, temperatures_fcc, label='fcc')
plt.plot(pressures_fcc_hcp/1.e9, temperatures_fcc_hcp, label='fcc-hcp')

plt.legend(loc='lower left')
plt.show()


print 'AFTER'

Fe_liq.set_state(1.e5, 1809.)
print Fe_liq.S, 55.845*1.e-6/Fe_liq.V, Fe_liq.alpha, Fe_liq.K_T/1.e9, Fe_liq.gr
'''

