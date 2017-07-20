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



crst = minerals.HP_2011_ds62.crst()
qtz_HP = minerals.HP_2011_ds62.q()
qtz = minerals.SLB_2011.quartz()
coe = minerals.SLB_2011.coesite()
stv = minerals.SLB_2011.stishovite()
SiO2_liq = minerals.current_melts.SiO2_liquid()

qtz_HP.set_state(1.e5, 300.)
qtz.set_state(1.e5, 300.)

crst.params['H_0'] = crst.params['H_0'] - qtz_HP.gibbs + qtz.gibbs


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
fig1 = mpimg.imread('figures/Shen_Lazor_1995_SiO2_melt.png')
fig2 = mpimg.imread('figures/Zhang_et_al_1993_SiO2_melt.png')

ax.imshow(fig1, extent=[0., 80., 1500.-273.15, 5000.-273.15], aspect='auto')
ax.imshow(fig2, extent=[0., 15., 1400., 3000.], aspect='auto')


inv = []
for (m1, m2, P_guess, T_guess) in [(qtz, coe, 4.e9, 2500.),
                                   (coe, stv, 14.e9, 3000.)]:
    pt = tools.invariant_point([SiO2_liq, m1], [1., -1.],
                               [SiO2_liq, m2], [1., -1.],
                               pressure_temperature_initial_guess=[P_guess, T_guess])
    inv.append(pt)
    temperatures = np.linspace(1200., pt[1], 101)
    pressures = np.zeros_like(temperatures)
    for i, T in enumerate(temperatures):
        pressures[i] = tools.equilibrium_pressure([m1, m2], [1., -1.],
                                                  T, pressure_initial_guess = P_guess)

    ax.plot(pressures/1.e9, temperatures-273.15, label='{0}-{1}'.format(m1.name, m2.name))
    

all_P = []
all_T = []
for (m, pressures, T_guess) in [(qtz, np.linspace(1.e5, inv[0][0], 101), inv[0][1]),
                                (coe, np.linspace(inv[0][0], inv[1][0], 101), inv[1][1]),
                                (stv, np.linspace(inv[1][0], 150.e9, 101), inv[1][1])]:
    temperatures = np.zeros_like(pressures)
    for i, P in enumerate(pressures):
        temperatures[i] = tools.equilibrium_temperature([SiO2_liq, m], [1., -1.],
                                                        P, temperature_initial_guess = T_guess)

    ax.plot(pressures/1.e9, temperatures-273.15, label='{0}-{1}'.format(m.name, SiO2_liq.name))
    all_P.extend(list(pressures))
    all_T.extend(list(temperatures))
    



PTs = [[1.e5, 2000.93408591],
       [1.e9, 2191.60813947],
       [2.e9, 2382.27861806],
       [5.e9, 2757.18107273],
       [10.e9, 3316.26545404],
       [20.e9, 4027.25345035],
       [25.e9, 4340.73094649]]

ax.scatter(np.array(zip(*PTs)[0])/1.e9, np.array(zip(*PTs)[1])-273.15)
ax.set_xlabel('Pressure (GPa)')
ax.set_ylabel('Temperature ($^{\circ}$C)')
ax.legend(loc='upper left')

fig.savefig("SiO2_melt_first_guess.pdf", bbox_inches='tight', dpi=100)

plt.show()
