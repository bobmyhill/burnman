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
qtz = minerals.HP_2011_ds62.q()
coe = minerals.HP_2011_ds62.coe()
stv = minerals.HP_2011_ds62.stv()

seif = minerals.SLB_2011.seifertite() # SLB seifertite


qtz_SLB = minerals.SLB_2011.quartz()
coe_SLB = minerals.SLB_2011.coesite()
stv_SLB = minerals.SLB_2011.stishovite()
seif_SLB = minerals.SLB_2011.seifertite()

temperatures = np.linspace(300., 5000., 501.)

for (m, P) in [(crst, 0.5e9),
               (coe, 5.e9),
               (stv, 50.e9)]:
    Cv = m.evaluate(['heat_capacity_v'], [P] * len(temperatures), temperatures)[0]
    plt.plot(temperatures, Cv, label=m.name)


temperatures = np.linspace(1., 5000., 501.)

for (m, P) in [(coe_SLB, 5.e9),
               (stv_SLB, 50.e9)]:
    Cv = m.evaluate(['heat_capacity_v'], [P] * len(temperatures), temperatures)[0]
    plt.plot(temperatures, Cv, label=m.name)
plt.legend(loc='lower right')
plt.show()


stv.set_state(80.e9, 300.)
stv_SLB.set_state(80.e9, 300.)
seif.params['F_0'] = seif.params['F_0'] - stv_SLB.gibbs + stv.gibbs

SiO2_liq = minerals.current_melts.SiO2_liquid()



fig = plt.figure()
ax = [fig.add_subplot(2, 3, i) for i in range(1, 6)]

fig1 = mpimg.imread('figures/Shen_Lazor_1995_SiO2_melt.png')
fig2 = mpimg.imread('figures/Zhang_et_al_1993_SiO2_melt.png')

ax[0].imshow(fig1, extent=[0., 80., 1500.-273.15, 5000.-273.15], aspect='auto')
ax[0].imshow(fig2, extent=[0., 15., 1400., 3000.], aspect='auto')


inv = []
for (m1, m2, P_guess, T_guess) in [(crst, qtz, 0.4e9, 1990.),
                                   (qtz, coe, 4.e9, 2500.),
                                   (coe, stv, 14.e9, 3000.),
                                   (stv, seif, 80.e9, 5000.)]:
    pt = tools.invariant_point([SiO2_liq, m1], [1., -1.],
                               [SiO2_liq, m2], [1., -1.],
                               pressure_temperature_initial_guess=[P_guess, T_guess])
    inv.append(pt)
    temperatures = np.linspace(1200., pt[1], 101)
    pressures = np.zeros_like(temperatures)
    for i, T in enumerate(temperatures):
        pressures[i] = tools.equilibrium_pressure([m1, m2], [1., -1.],
                                                  T, pressure_initial_guess = P_guess)

    ax[0].plot(pressures/1.e9, temperatures-273.15, label='{0}-{1}'.format(m1.name, m2.name))
    

all_P = []
all_T = []
gr = []
Cv = []
V = []
S = []
gr2 = []
Cv2 = []
V2 = []
S2 = []
for (m, m_SLB, pressures, T_guess) in [(crst, qtz_SLB, np.linspace(1.e5, inv[0][0], 101), inv[0][1]),
                                       (qtz, qtz_SLB, np.linspace(inv[0][0], inv[1][0], 101), inv[1][1]),
                                       (coe, coe_SLB, np.linspace(inv[1][0], inv[2][0], 101), inv[2][1]),
                                       (stv, stv_SLB, np.linspace(inv[2][0], inv[3][0], 101), inv[3][1]),
                                       (seif, seif_SLB, np.linspace(inv[3][0], 150.e9, 101), inv[3][1])]:
    temperatures = np.zeros_like(pressures)
    for i, P in enumerate(pressures):
        temperatures[i] = tools.equilibrium_temperature([SiO2_liq, m], [1., -1.],
                                                        P, temperature_initial_guess = T_guess)
        gr.append(m.grueneisen_parameter)
        Cv.append(m.heat_capacity_v)
        V.append(m.V)
        S.append(m.S)

        m_SLB.set_state(P, temperatures[i])
        gr2.append(m_SLB.grueneisen_parameter)
        Cv2.append(m_SLB.heat_capacity_v)
        V2.append(m_SLB.V)
        S2.append(m_SLB.S)
        

    ax[0].plot(pressures/1.e9, temperatures-273.15, label='{0}-{1}'.format(m.name, SiO2_liq.name))
    all_P.extend(list(pressures))
    all_T.extend(list(temperatures))
    



PTs = [[1.e5, 2000.93408591],
       [1.e9, 2191.60813947],
       [2.e9, 2382.27861806],
       [5.e9, 2757.18107273],
       [10.e9, 3316.26545404],
       [20.e9, 4027.25345035],
       [25.e9, 4340.73094649]]

ax[0].scatter(np.array(zip(*PTs)[0])/1.e9,  np.array(zip(*PTs)[1])-273.15)


pressures = np.array(all_P)
temperatures = np.array(all_T)
ax[1].plot(pressures/1.e9, gr, label='HP')
ax[2].plot(pressures/1.e9, Cv, label='HP')
ax[3].plot(pressures/1.e9, S, label='HP')
ax[4].plot(pressures/1.e9, V, label='HP')

ax[1].plot(pressures/1.e9, gr2, label='SLB')
ax[2].plot(pressures/1.e9, Cv2, label='SLB')
ax[3].plot(pressures/1.e9, S2, label='SLB')
ax[4].plot(pressures/1.e9, V2, label='SLB')


ax[0].set_ylabel('Temperature ($^{\circ}$C)')
ax[1].set_ylabel('$\gamma$')
ax[2].set_ylabel('$C_V$')
ax[3].set_ylabel('$S$')
ax[4].set_ylabel('$V$')

for i in range(0, 5):
    ax[i].set_xlabel('Pressure (GPa)')
    ax[i].legend(loc='upper left')

fig.savefig("SiO2_melt_first_guess.pdf", bbox_inches='tight', dpi=100)

plt.show()
