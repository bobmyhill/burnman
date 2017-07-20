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

SiO2_liq_DKS = minerals.DKS_2013_liquids.SiO2_liquid()
SiO2_liq = minerals.current_melts.SiO2_liquid()
SiO2_liq_B = minerals.BOZA_2017.SiO2_liquid()

SiO2_liq_DKS.name='DKS'
SiO2_liq.name='This study'
SiO2_liq_B.name='BOZA'


temperatures = np.linspace(1200., 2400., 101)
pressures = np.array([1.e9] * len(temperatures))

fig = plt.figure()
ax = [fig.add_subplot(2, 1, i) for i in range(1, 3)]

for liq in [SiO2_liq, SiO2_liq_DKS, SiO2_liq_B]:
    V, C_v = liq.evaluate(['V', 'heat_capacity_v'], pressures, temperatures)
    ax[0].plot(temperatures, V, label=liq.name)
    ax[1].plot(temperatures, C_v, label=liq.name)

ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
plt.show()


pressures = np.linspace(1.e5, 25.e9, 101)
temperatures = np.array([2000.] * len(pressures))

fig = plt.figure()
ax = [fig.add_subplot(2, 1, i) for i in range(1, 3)]

for liq in [SiO2_liq, SiO2_liq_DKS, SiO2_liq_B]:
    V, C_v = liq.evaluate(['V', 'heat_capacity_v'], pressures, temperatures)
    ax[0].plot(pressures/1.e9, V, label=liq.name)
    ax[1].plot(pressures/1.e9, C_v, label=liq.name)

ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
plt.show()




crst_B = minerals.BOZA_2017.beta_cristobalite()
qtz_B = minerals.BOZA_2017.beta_quartz()
coe_B = minerals.BOZA_2017.coesite()
stv_B = minerals.BOZA_2017.stishovite()

temperatures = np.linspace(1700., 2300., 101)
pressures = [1.e5] * len(temperatures)

fig = plt.figure()
ax_V = fig.add_subplot(4, 1, 1)
ax_G = fig.add_subplot(4, 1, 2)
ax_S = fig.add_subplot(4, 1, 3)
ax_C = fig.add_subplot(4, 1, 4)
ax_V.plot(temperatures, SiO2_liq_B.evaluate(['V'], pressures, temperatures)[0])
ax_V.plot(temperatures, crst_B.evaluate(['V'], pressures, temperatures)[0])


ax_G.plot(temperatures, (SiO2_liq_B.evaluate(['gibbs'], pressures, temperatures)[0] -
                         crst_B.evaluate(['gibbs'], pressures, temperatures)[0]))

ax_S.plot(temperatures, (SiO2_liq_B.evaluate(['S'], pressures, temperatures)[0] -
                         crst_B.evaluate(['S'], pressures, temperatures)[0]))


ax_C.plot(temperatures, SiO2_liq_B.evaluate(['heat_capacity_p'], pressures, temperatures)[0]) 
ax_C.plot(temperatures, crst_B.evaluate(['heat_capacity_p'], pressures, temperatures)[0]) 
plt.show()




fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
fig1 = mpimg.imread('figures/Shen_Lazor_1995_SiO2_melt.png')
fig2 = mpimg.imread('figures/Zhang_et_al_1993_SiO2_melt.png')

ax.imshow(fig1, extent=[0., 80., 1500.-273.15, 5000.-273.15], aspect='auto')
ax.imshow(fig2, extent=[0., 15., 1400., 3000.], aspect='auto')


inv = []
for (m1, m2, P_guess, T_guess) in [(crst_B, qtz_B, 0.4e9, 1990.),
                                   (qtz_B, coe_B, 4.e9, 2500.),
                                   (coe_B, stv_B, 14.e9, 3000.)]:
    pt = tools.invariant_point([SiO2_liq_B, m1], [1., -1.],
                               [SiO2_liq_B, m2], [1., -1.],
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
for (m, pressures, T_guess) in [(crst_B, np.linspace(1.e5, inv[0][0], 101), inv[0][1]),
                                (qtz_B, np.linspace(inv[0][0], inv[1][0], 101), inv[1][1]),
                                (coe_B, np.linspace(inv[1][0], inv[2][0], 101), inv[2][1]),
                                (stv_B, np.linspace(inv[2][0], 80.e9, 101), inv[2][1])]:
    temperatures = np.zeros_like(pressures)
    for i, P in enumerate(pressures):
        temperatures[i] = tools.equilibrium_temperature([SiO2_liq_B, m], [1., -1.],
                                                        P, temperature_initial_guess = T_guess)

    ax.plot(pressures/1.e9, temperatures-273.15, label='{0}-{1}'.format(m.name, SiO2_liq_B.name))
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

plt.show()


pressures = np.array(all_P)
temperatures = np.array(all_T)
    
Vl, Sl, Cvl = SiO2_liq_B.evaluate(['V', 'S', 'heat_capacity_v'], pressures, temperatures)
Vs, Ss, Cvs = stv_B.evaluate(['V', 'S', 'heat_capacity_v'], pressures, temperatures)

fig = plt.figure()
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

ax[0].imshow(fig1, extent=[0., 80., 1500., 5000.], aspect='auto')
ax[0].imshow(fig2, extent=[0., 15., 1400.+273.15, 3000.+273.15], aspect='auto')

ax[0].plot(pressures/1.e9, temperatures)
ax[1].plot(pressures/1.e9, Vl, label='liquid')
ax[1].plot(pressures/1.e9, Vs, label='solid')
ax[2].plot(pressures/1.e9, Sl, label='liquid')
ax[2].plot(pressures/1.e9, Ss, label='solid')
ax[3].plot(pressures/1.e9, Cvl, label='liquid')
ax[3].plot(pressures/1.e9, Cvs, label='solid')



for T in [2000., 3000., 4000., 5000.]:
    print(T)
    temperatures = np.array([T] * len(pressures))
    Vl, Sl, Cvl = SiO2_liq_B.evaluate(['V', 'S', 'heat_capacity_v'], pressures, temperatures)
    Vs, Ss, Cvs = stv_B.evaluate(['V', 'S', 'heat_capacity_v'], pressures, temperatures)

    #ax[3].plot(pressures/1.e9, Cvl, label='Liquid ({0:.0f} K)'.format(T))
    #ax[3].plot(pressures/1.e9, Cvs, label='Solid ({0:.0f} K)'.format(T))


    
stv_SLB = minerals.SLB_2011.stishovite()
stv_HP = minerals.HP_2011_ds62.stv()
pressures = np.linspace(20.e9, 80.e9, 101)
for T in [2000., 3000., 4000., 5000.]:
    print(T)
    temperatures = np.array([T] * len(pressures))   
    VsS, SsS, CvsS = stv_SLB.evaluate(['V', 'S', 'heat_capacity_v'], pressures, temperatures)
    VsH, SsH, CvsH = stv_HP.evaluate(['V', 'S', 'heat_capacity_v'], pressures, temperatures)
    ax[3].plot(pressures/1.e9, CvsS, label='SLB ({0:.0f} K)'.format(T))
    ax[3].plot(pressures/1.e9, CvsH, label='HP ({0:.0f} K)'.format(T))



    
ax[1].legend(loc='upper right')
ax[2].legend(loc='upper right')
ax[3].legend(loc='upper right')
plt.show()
