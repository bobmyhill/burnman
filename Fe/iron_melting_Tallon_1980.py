# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

from scipy.interpolate import interp1d
from scipy.optimize import fsolve, curve_fit
import burnman
from HP_convert import *
from listify_xy_file import *
from fitting_functions import *

# Tallon (1980) suggested that melting of simple substances was associated with an entropy change of
# Sfusion = burnman.constants.gas_constant*np.log(2.) + a*K_T*Vfusion
# Realising also that dT/dP = Vfusion/Sfusion, we can express the entropy 
# and volume of fusion in terms of the melting curve:
# Sfusion = burnman.constants.gas_constant*np.log(2.) / (1. - a*K_T*dTdP)
# Vfusion = Sfusion*dT/dP

Fe_liq_Kom = burnman.minerals.Komabayashi_2014.liquid_iron()
Fe_fcc_Kom = burnman.minerals.Komabayashi_2014.fcc_iron()
Fe_hcp_Kom = burnman.minerals.Komabayashi_2014.hcp_iron()


Fe_liq = burnman.minerals.Myhill_calibration_iron.liquid_iron()
Fe_fcc = burnman.minerals.Myhill_calibration_iron.fcc_iron()
Fe_hcp = burnman.minerals.Myhill_calibration_iron.hcp_iron()

melting_curve_data = listify_xy_file('data/Anzellini_2013_Fe_melting_curve.dat')
melting_curve_data = listify_xy_file('data/Fe_melting_Komabayashi.dat')
melting_temperature = interp1d(melting_curve_data[0]*1.e9, 
                               melting_curve_data[1], 
                               kind='cubic')
'''
# Plot the melting curve we're trying to fit
data = listify_xy_file('data/Fe_melting_Anzellini_2003.dat')
plt.plot(data[0]*1.e9, data[1], linestyle='None', marker='o')
plt.plot(melting_curve_data[0]*1.e9, melting_curve_data[1])
pressures = np.linspace(1.e5, 225.e9, 101)
plt.plot(pressures, melting_temperature(pressures))
plt.show()
'''

# Convert solid phase parameters to HP 
P_ref = 200.e9
T_ref = 1809.
HP_convert(Fe_fcc, 300., 2200., T_ref, P_ref)
HP_convert(Fe_hcp, 300., 2200., T_ref, P_ref)

Fe_fcc.params['Cp'] = [52.2754, -0.000355156, 790710.86, -619.07]
Fe_hcp.params['Cp'] = [52.2754, -0.000355156, 790710.86, -619.07]

'''
# First, let's compare the Komabayashi and Myhill EoSs
temperatures = np.linspace(300., 4000., 101)
volumes_Kom = np.empty_like(temperatures)
volumes = np.empty_like(temperatures)
entropies_Kom = np.empty_like(temperatures)
entropies = np.empty_like(temperatures)
Cps_Kom = np.empty_like(temperatures)
Cps = np.empty_like(temperatures)
Cps_fcc_Kom = np.empty_like(temperatures)
Cps_fcc = np.empty_like(temperatures)
P = 100.e9
for i, T in enumerate(temperatures):
    Fe_hcp_Kom.set_state(P, T)
    Fe_hcp.set_state(P, T)
    Fe_fcc_Kom.set_state(1.e5, T)
    Fe_fcc.set_state(1.e5, T)
    volumes_Kom[i] = Fe_hcp_Kom.V
    volumes[i] = Fe_hcp.V
    entropies_Kom[i] = Fe_hcp_Kom.S
    entropies[i] = Fe_hcp.S
    Cps_Kom[i] = Fe_hcp_Kom.C_p
    Cps[i] =  Fe_hcp.C_p
    Cps_fcc_Kom[i] = Fe_fcc_Kom.C_p
    Cps_fcc[i] =  Fe_fcc.C_p


plt.plot(temperatures, volumes_Kom, label='Komabayashi')
plt.plot(temperatures, volumes)
plt.legend(loc="lower right")
plt.show()

plt.plot(temperatures, entropies_Kom, label='Komabayashi')
plt.plot(temperatures, entropies)
plt.legend(loc="lower right")
plt.show()

plt.plot(temperatures, Cps_Kom, label='Komabayashi')
plt.plot(temperatures, Cps)
plt.legend(loc="lower right")
plt.show()

plt.plot(temperatures, Cps_fcc_Kom, label='Komabayashi')
plt.plot(temperatures, Cps_fcc)
plt.legend(loc="lower right")
plt.show()
'''
'''
# Convert liquid phase parameters to HP
P_ref = 97.e9
T_ref = melting_temperature(P_ref)

# Make sure HCP=FCC
Fe_hcp.params['S_0'] = Fe_hcp.params['S_0'] + 1.2
Fe_hcp.set_state(P_ref, T_ref)
Fe_fcc.set_state(P_ref, T_ref)
Fe_hcp.params['H_0'] = Fe_hcp.params['H_0'] + Fe_fcc.gibbs - Fe_hcp.gibbs

Fe_liq.params['T_einstein'] = Fe_hcp.params['T_einstein']
Fe_liq.params['a_0'] = 78.e-6
HP_convert(Fe_liq, 1809., 2400., T_ref, P_ref)
Fe_liq.params['Cp'] = [38., 0., 0., 0.]

Fe_liq.set_state(1.e5, 1809.)
print Fe_liq.C_p
Fe_liq.set_state(1.e5, 3000.)
print Fe_liq.C_p

# Here we find the properties of the liquid at the HCP-FCC-LIQ triple point
dP = 100. # Pa
dT = melting_temperature(P_ref + dP/2.) - melting_temperature(P_ref - dP/2.)
dTdP = dT/dP
Fe_fcc.set_state(P_ref, T_ref)
aK_T = Fe_fcc.alpha*Fe_fcc.K_T
Sfusion = burnman.constants.gas_constant*np.log(2.) / (1. - aK_T*dTdP)
Vfusion = Sfusion*dTdP

Fe_liq.set_state(P_ref, T_ref)
Fe_fcc.set_state(P_ref, T_ref)
Fe_liq.params['V_0'] = Fe_fcc.V + Vfusion
Fe_liq.params['S_0'] = Fe_fcc.S + Sfusion

Fe_liq.set_state(P_ref, T_ref+0.00001)
Fe_liq.params['H_0'] = Fe_fcc.gibbs + T_ref*Fe_liq.params['S_0']

print 'hcp'
print Fe_hcp.params
print 'fcc'
print Fe_fcc.params
print 'liq'
print Fe_liq.params


#### FITTING FINISHED

print 'Check properties at 1 bar'
Fe_liq.set_state(1.e9, 1809.)
print 'T  Cp  alpha'
print 1809., Fe_liq.C_p, Fe_liq.alpha
Fe_liq.set_state(1.e9, 2409.)
print 2409., Fe_liq.C_p, Fe_liq.alpha


# Now we plot the entropy and volume of the liquid phase along the melting curve
pressures = np.linspace(1.e5, 225.e9, 31)
Sfusion = np.empty_like(pressures)
Vfusion = np.empty_like(pressures)
Smelt = np.empty_like(pressures)
Smelt2 = np.empty_like(pressures)
Smelt3 = np.empty_like(pressures)
Smelt4 = np.empty_like(pressures)
Smelt5 = np.empty_like(pressures)
Vmelt = np.empty_like(pressures)
Vmelt2 = np.empty_like(pressures)
Vmelt3 = np.empty_like(pressures)
alpha_melt = np.empty_like(pressures)
alpha_melt2 = np.empty_like(pressures)

for i, P in enumerate(pressures):
    if P > 97.e9:
        Fe_phase = Fe_hcp
    else:
        Fe_phase = Fe_fcc

    dP = 100. # Pa
    dT = melting_temperature(P + dP/2.) - melting_temperature(P - dP/2.)
    dTdP = dT/dP
    T = melting_temperature(P)
    Fe_phase.set_state(P, T)
    aK_T = Fe_phase.alpha*Fe_phase.K_T
    Sfusion[i] = burnman.constants.gas_constant*np.log(2.) / (1. - aK_T*dTdP)
    Vfusion[i] = Sfusion[i]*dTdP

    Smelt[i] = Fe_phase.S + Sfusion[i]
    Vmelt[i] = Fe_phase.V + Vfusion[i]

    print P/1.e9, Fe_liq.K_T, Fe_phase.K_T, Fe_liq.S, Fe_phase.S


    Fe_liq.set_state(P, T)
    Smelt2[i] = Fe_liq.S
    Vmelt2[i] = Fe_liq.V

    Fe_hcp_Kom.set_state(P, T)
    Fe_liq_Kom.set_state(P, T)


    alpha_melt[i] = Fe_liq.alpha
    alpha_melt2[i] = Fe_liq_Kom.alpha

    Smelt3[i] = Fe_liq_Kom.S
    Vmelt3[i] = Fe_liq_Kom.V


    Smelt4[i] = Fe_phase.S
    Smelt5[i] = Fe_hcp_Kom.S

plt.plot(pressures, Smelt)
plt.plot(pressures, Smelt2, marker='o')
plt.plot(pressures, Smelt3, marker='o')
plt.plot(pressures, Smelt4, marker='o')
plt.plot(pressures, Smelt5, marker='o')
plt.show()


plt.plot(pressures, Vmelt)
plt.plot(pressures, Vmelt2, marker='o')
plt.plot(pressures, Vmelt3, marker='o')

plt.show()

plt.plot(pressures, alpha_melt)
plt.plot(pressures, alpha_melt2, marker='o')

plt.show()

pressures = [1.e5, 50.e9]
temperatures = np.linspace(300., 3000., 101)

def ak_infty(phase):
    u_0 = phase.params['T_einstein']/phase.params['T_0']
    ksi_0 = u_0*u_0*np.exp(u_0)/(np.power((np.exp(u_0) - 1.), 2.))
    return phase.params['a_0']*phase.params['K_0']/ksi_0

print ak_infty(Fe_hcp)
print ak_infty(Fe_fcc)

ak = np.empty_like(temperatures)
for P in pressures:
    for i, T in enumerate(temperatures):
        Fe_hcp.set_state(P, T)
        ak[i] = Fe_hcp.alpha*Fe_hcp.K_T
    plt.plot(temperatures, ak)

plt.show()


pressures = np.linspace(5.2e9, 330.e9, 101)
temperatures = np.empty_like(pressures)
temperatures_2 = np.empty_like(pressures)
for i, P in enumerate(pressures):
    if P > 97.e9:
        temperatures[i] = fsolve(eqm_temperature([Fe_hcp, Fe_liq], [1.0,-1.0]), [2000.], args=(P))[0]
        temperatures_2[i] = temperatures[i]
    else:
        temperatures[i] = fsolve(eqm_temperature([Fe_fcc, Fe_liq], [1.0,-1.0]), [2000.], args=(P))[0]
        temperatures_2[i] = fsolve(eqm_temperature([Fe_fcc, Fe_hcp], [1.0,-1.0]), [2000.], args=(P))[0]
plt.plot(pressures, temperatures)
plt.plot(pressures, temperatures_2)
'''

data = listify_xy_file('data/Fe_melting_Anzellini_2003.dat')
plt.plot(data[0]*1.e9, data[1], linestyle='None', marker='o')
data = listify_xy_file('data/Fe_melting_Komabayashi.dat')
plt.plot(data[0]*1.e9, data[1], 'r--')

plt.show()

