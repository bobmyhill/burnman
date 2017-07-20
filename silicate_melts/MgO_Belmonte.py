import os, sys, numpy as np, matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
plt.style.use('ggplot')
rcParams['figure.figsize'] = 15, 10
#plt.rcParams['axes.facecolor'] = 'white'
#plt.rcParams['axes.edgecolor'] = 'black'

if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import tools
from burnman.processchemistry import formula_mass, dictionarize_formula
from burnman import minerals
from scipy.optimize import brentq, fsolve



def gr(x, gamma_0, gamma_inf, q_0):
    lmda = q_0/np.log(gamma_0/gamma_inf)
    gamma = gamma_0 * np.exp( q_0/lmda * ( np.power(x, lmda) - 1. ) )
    return gamma


per_B = minerals.BOZA_2017.periclase()
MgO_liq_B = minerals.BOZA_2017.MgO_liquid()

tools.check_eos_consistency(per_B, tol=1.e-2, verbose=True)

per_DKS = minerals.DKS_2013_solids.periclase()
MgO_liq_DKS = minerals.DKS_2013_liquids.MgO_liquid()

pressures = np.linspace(1.e5, 100.e9, 101)
for T in [2000., 3000., 4000., 5000.]:
    temperatures = [T]*len(pressures)
    
    Vl, Sl, Cvl, grl = MgO_liq_DKS.evaluate(['V', 'S', 'heat_capacity_v', 'grueneisen_parameter'], pressures, temperatures)
    
    plt.plot(Vl, grl, label='{0} K'.format(T))

Kprime_inf = 3.2
gamma_inf = 0.5*Kprime_inf - 1./6.

volumes = np.linspace(0.0000001, Vl[0], 101)
plt.plot(volumes, gr((volumes/Vl[0]), 0.65, gamma_inf, -1.9))


plt.legend(loc='lower right')    
plt.show()


for (per, MgO_liq) in [(per_B, MgO_liq_B), (per_DKS, MgO_liq_DKS)]:
    pressures = np.linspace(1.e5, 39.e9, 101)
    temperatures = np.empty_like(pressures)
    
    T_guess = 3000.
    
    for i, P in enumerate(pressures):
        temperatures[i] = tools.equilibrium_temperature([per, MgO_liq], [1.0, -1.0], P, temperature_initial_guess = T_guess)
        
    Vl, Sl, KSl, Cvl = MgO_liq.evaluate(['V', 'S', 'K_S', 'heat_capacity_v'], pressures, temperatures)
    Vs, Ss, KSs, Cvs = per.evaluate(['V', 'S', 'K_S', 'heat_capacity_v'], pressures, temperatures)
    
    fig = plt.figure()
    ax = [fig.add_subplot(2, 3, i) for i in range(1, 7)]
    
    ax[0].plot(pressures/1.e9, temperatures)
    ax[1].plot(pressures/1.e9, Vl, label='liquid')
    ax[1].plot(pressures/1.e9, Vs, label='solid')
    
    ax[2].plot(pressures/1.e9, KSl/1.e9, label='liquid')
    ax[2].plot(pressures/1.e9, KSs/1.e9, label='solid')

    Kpl = np.gradient(KSl, pressures, edge_order=2)
    Kps = np.gradient(KSs, pressures, edge_order=2)
    
    ax[3].plot(pressures/1.e9, Kpl, label='liquid')
    ax[3].plot(pressures/1.e9, Kps, label='solid')
    
    ax[4].plot(pressures/1.e9, Sl, label='liquid')
    ax[4].plot(pressures/1.e9, Ss, label='solid')
    ax[5].plot(pressures/1.e9, Cvl, label='liquid')
    ax[5].plot(pressures/1.e9, Cvs, label='solid')

    for i in range(1, 6):
        ax[i].set_xlabel('Pressure (GPa)')
        ax[i].legend(loc='upper right')

    
    ax[0].set_ylabel('Temperature (K)')
    ax[1].set_ylabel('Volume (m$^3$/mol)')
    
    ax[2].set_ylabel('$K_S$ (GPa)')
    ax[3].set_ylabel('$K\'_S$')
    
    ax[4].set_ylabel('Entropy (J/K/mol)')
    ax[5].set_ylabel('$C_V$ (J/K/mol)')
        
    plt.show()
