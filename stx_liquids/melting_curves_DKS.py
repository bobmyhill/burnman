import numpy as np
from scipy.optimize import fsolve, brentq, root
import matplotlib.pyplot as plt


import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    DKS_2013_liquids, \
    DKS_2013_solids, \
    SLB_2011, \
    HP_2011_ds62, HHPH_2013
from burnman import constants
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from silicate_models_DKS_raw import *


pv_DKS = DKS_2013_solids.perovskite()
pv_SLB = SLB_2011.mg_perovskite()
pv_HP = HP_2011_ds62.mpv()
pv_HHPH = HHPH_2013.mpv()

#pv_DKS.property_modifiers = [['linear', {'delta_E': 15000., 'delta_S': -14.5, 'delta_V': 0.}]]

per_DKS = DKS_2013_solids.periclase()
per_SLB = SLB_2011.periclase()
#per_SLB.params['grueneisen_0'] = 1.6
#per_SLB.params['q_0'] = 0.15
per_HP = HP_2011_ds62.per()
per_HHPH = HHPH_2013.per()

stv_DKS = DKS_2013_solids.stishovite()
stv_SLB = SLB_2011.stishovite()
stv_HP = HP_2011_ds62.stv()
stv_HHPH = HHPH_2013.stv()

#stv_SLB.property_modifiers = [['linear', {'delta_E': 0., 'delta_S': 0., 'delta_V': 0.}]] # remove landau for comparison

mmm = [[per_DKS, pv_DKS, stv_DKS],
       [per_SLB, pv_SLB, stv_SLB],
       [per_HP, pv_HP, stv_HP],
       [per_HHPH, pv_HHPH, stv_HHPH]]

Simon_Glatzel = lambda Tr, Pr, A, C, P: Tr*np.power(1. + (P - Pr)/A, 1./C)
pressures = np.linspace(20.e9, 140.e9, 7)
per_pv_temperatures = Simon_Glatzel(2705., 24., 19.156, 3.7796, pressures/1.e9)
pv_stv_temperatures = Simon_Glatzel(2605., 24., 29.892, 3.677, pressures/1.e9)

temperatures = 0.5*(per_pv_temperatures + pv_stv_temperatures)
#temperatures = np.linspace(300., 3000., 101)
#pressures = 100.e9 + temperatures*0.
#temperatures = pressures*0. + 3000.
print 'P, T, DKS, SLB, HP'
Garr = []
Sarr = []
Varr = []
Cparr = []
for (P, T) in zip(*[pressures, temperatures]):
    print P/1.e9, T
    Gs = []
    Ss = []
    Vs = []
    Cps = []
    for mm in mmm:
        for m in mm:
            m.set_state(P, T)
        #print mm[1].gibbs - (mm[0].gibbs + mm[2].gibbs),
        #print 'S', mm[1].S - (mm[0].S + mm[2].S),
        #print 'V', (mm[1].V - (mm[0].V + mm[2].V))*1.e6,
        Gs.append(mm[1].gibbs - (mm[0].gibbs + mm[2].gibbs))
        Ss.append(mm[1].S - (mm[0].S + mm[2].S))
        Vs.append((mm[1].V - (mm[0].V + mm[2].V))*1.e6)
        Cps.append((mm[1].heat_capacity_p - (mm[0].heat_capacity_p + mm[2].heat_capacity_p))/mm[1].heat_capacity_p )
        print mm[1].gibbs - (mm[0].gibbs + mm[2].gibbs)
    print
    Garr.append(Gs)
    Sarr.append(Ss)
    Varr.append(Vs)
    Cparr.append(Cps)

Garr = np.array(Garr).T
Sarr = np.array(Sarr).T
Varr = np.array(Varr).T
Cparr = np.array(Cparr).T
#plt.plot(pressures/1.e9, Garr[2] - Garr[1], label='HP - SLB')
plt.plot(pressures/1.e9, Garr[1] - Garr[0], label='SLB - DKS')
plt.plot(pressures/1.e9, Garr[2] - Garr[0], label='HP - DKS')
#plt.plot(pressures/1.e9, Garr[3] - Garr[0], label='HHPH - DKS')
plt.legend(loc='lower right')
plt.show()
plt.plot(temperatures, Sarr[0], label='DKS')
plt.plot(temperatures, Sarr[1], label='SLB')
plt.plot(temperatures, Sarr[2], label='HP')
#plt.plot(temperatures, Sarr[3], label='HHPH')
plt.legend(loc='lower right')
plt.show()
plt.plot(pressures/1.e9, Varr[1] - Varr[0], label='SLB - DKS')
plt.plot(pressures/1.e9, Varr[2] - Varr[0], label='HP - DKS')
#plt.plot(pressures/1.e9, Varr[3] - Varr[0], label='HHPH - DKS')
plt.legend(loc='lower right')
plt.show()
plt.plot(temperatures, Cparr[1], label='SLB')
plt.plot(temperatures, Cparr[2], label='HP')
#plt.plot(temperatures, Cparr[3], label='HHPH')
plt.legend(loc='lower right')
plt.show()
exit()



def MS_liquidus_temperature(temperature, pressure, solid, n_cations_solid, SiO2_fraction_solid, SiO2_fraction):
    # Find liquidus curves from stoichiometric phases
    
    solid.set_state(pressure, temperature)
    FMS.molar_fractions = np.array([0.001, (1. - SiO2_fraction),  SiO2_fraction - 0.001])
    FMS.set_state(pressure, temperature)
    mu_phase_liq = ( FMS.partial_gibbs[1]*( 1. - SiO2_fraction_solid ) +
                     FMS.partial_gibbs[2]*SiO2_fraction_solid ) * n_cations_solid
    return solid.gibbs - mu_phase_liq


def MS_eutectic_temperature(args, pressure, solids, n_cations_solids, SiO2_fraction_solids):
    # Find liquidus curves from stoichiometric phases
    temperature, SiO2_fraction = args
    FMS.molar_fractions = np.array([0.001, (1. - SiO2_fraction),  SiO2_fraction - 0.001])
    FMS.set_state(pressure, temperature)

    out = []
    for i, solid in enumerate(solids):
        solid.set_state(pressure, temperature)
        
        mu_phase_liq = ( FMS.partial_gibbs[1]*( 1. - SiO2_fraction_solids[i] ) +
                         FMS.partial_gibbs[2]*SiO2_fraction_solids[i] ) * n_cations_solids[i]
        out.append(solid.gibbs - mu_phase_liq)
    return out

pressures = np.linspace(24.e9, 136.e9, 21)
temperatures = np.empty_like(pressures)
compositions = np.empty_like(pressures)
temperatures2 = np.empty_like(pressures)
compositions2 = np.empty_like(pressures)
Tguess = 4000.
Xguess = 0.3
Tguess2 = 4000.
Xguess2 = 0.9
for i, P in enumerate(pressures):
    sol = fsolve(MS_eutectic_temperature, [Tguess, Xguess], args=(P, [per, mpv], [1., 2.], [0., 1./2.]), full_output=True)
    if sol[2] == 1:
        print P/1.e9, sol[0],
        temperatures[i] = sol[0][0]
        compositions[i] = sol[0][1]
        Tguess = sol[0][0]
        Xguess = sol[0][1]
    sol = fsolve(MS_eutectic_temperature, [Tguess2, Xguess2], args=(P, [mpv, stv], [2., 1.], [1./2., 1.]), full_output=True)
    if sol[2] == 1:
        print sol[0]
        temperatures2[i] = sol[0][0]
        compositions2[i] = sol[0][1]
        Tguess2 = sol[0][0]
        Xguess2 = sol[0][1]

plt.plot(pressures/1.e9, temperatures, color='red', label='model, MgO-MgSiO3')
plt.plot(pressures/1.e9, temperatures2, color='blue', label='model, MgSiO3-SiO2')

Simon_Glatzel = lambda T24, A, C, P: T24*np.power(1. + (P - 24.)/A, 1./C)
plt.plot(pressures/1.e9, Simon_Glatzel(2705., 19.156, 3.7796, pressures/1.e9), color='red', linestyle='--', label='expt, MgO-MgSiO3')
plt.plot(pressures/1.e9, Simon_Glatzel(2605., 29.892, 3.677, pressures/1.e9), color='blue', linestyle='--', label='expt, MgSiO3-SiO2')
plt.legend(loc='upper left')
plt.show()

exit()




curves = [[per, 1., 0., 0.001, 0.40, 21],
          [mpv, 2., 1./2., 0.35, 0.8, 21],
          [stv, 1., 1., 0.70, 0.999, 21]]


for P in [24.e9]:
    for curve in curves:
        solid, nc_solid, c_solid, X_min, X_max, n = curve
        X_SiO2 = np.linspace(X_min, X_max, n)
    
        temperatures = np.empty_like(X_SiO2)
        
    
        Tmin = 2000.
        Tmax = 6000.
                
        for i, X in enumerate(X_SiO2):
            sol = brentq(MS_liquidus_temperature, Tmin, Tmax, args=(P, solid, nc_solid, c_solid, X), full_output=True)
            if sol[1].converged == True:
                temperatures[i] = sol[0]
                Tmin = temperatures[i] - 300.
                Tmax = temperatures[i] + 300.
                print X, temperatures[i]
            else:
                temperatures[i] = 2000.
        plt.plot(X_SiO2, temperatures)
plt.ylabel('Temperature (K)')
plt.xlabel('X SiO2')
plt.show()
