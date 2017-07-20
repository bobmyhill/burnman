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




per_DKS = minerals.DKS_2013_solids.periclase()
per_SLB = minerals.SLB_2011.periclase()
MgO_liq = minerals.current_melts.MgO_liquid()



fig = plt.figure()
ax = [fig.add_subplot(2, 2, i) for i in range(1, 4)]

fig1 = mpimg.imread('figures/Belonoshko_2010_MgO_melting.png')

ax[0].imshow(fig1, extent=[0., 300., 2000., 10000.], aspect='auto')

pressures = np.linspace(20.e9, 200.e9, 101)
temperatures = np.zeros_like(pressures)
T_guess = 3000.
for i, P in enumerate(pressures):
    temperatures[i] = tools.equilibrium_temperature([MgO_liq, per_DKS], [1., -1.],
                                                    P, temperature_initial_guess = T_guess)

ax[0].scatter(pressures/1.e9, temperatures, label='{0}-{1}'.format(per_DKS.name, MgO_liq.name))


V_SLB, S_SLB = per_SLB.evaluate(['V', 'S'], pressures, temperatures)
V_DKS, S_DKS = per_DKS.evaluate(['V', 'S'], pressures, temperatures)
ax[1].plot(pressures/1.e9, V_SLB, label='SLB')
ax[1].plot(pressures/1.e9, V_DKS, label='DKS')
ax[2].plot(pressures/1.e9, S_SLB, label='SLB')
ax[2].plot(pressures/1.e9, S_DKS, label='DKS')

for i in range(0, 3):
    ax[i].set_xlabel('Pressure (GPa)')
    ax[i].legend(loc='upper left')
    
ax[0].set_ylabel('Temperature ($^{\circ}$C)')
ax[1].set_ylabel('Volume (m$^3$/mol)')
ax[2].set_ylabel('Entropy (J/K/mol)')


fig.savefig("MgO_melt_first_guess.pdf", bbox_inches='tight', dpi=100)

plt.show()
