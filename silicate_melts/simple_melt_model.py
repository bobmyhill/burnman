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

from burnman.combinedmineral import CombinedMineral

'''
per = minerals.HP_2011_ds62.per()
MgO_liq = minerals.current_melts.MgO_liquid()

pressures = np.linspace(1.e5, 100.e9, 101)
temperatures = np.zeros_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = tools.equilibrium_temperature([MgO_liq, per], [1., -1.],
                                                    P, temperature_initial_guess = 3098.)

plt.plot(pressures/1.e9, temperatures)
plt.show()
exit()
'''


pren = minerals.HP_2011_ds62.pren()
en = minerals.HP_2011_ds62.en()
hen= minerals.HP_2011_ds62.hen()
maj = CombinedMineral([minerals.HP_2011_ds62.maj()], [0.5], [0., 0., 0.]) # Mg4Si4O12/2.
Mg2Si2O6_liq = minerals.current_melts.Mg2Si2O6_liquid()


temperatures = np.linspace(1500., 2100., 101)
pressures = np.array([1.e5] * len(temperatures))
Cps = Mg2Si2O6_liq.evaluate(['heat_capacity_p'], pressures, temperatures)[0]
plt.plot(temperatures, Cps)
plt.plot(temperatures, [146.44*2.]*len(temperatures)) # JANAF
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

fig1 = mpimg.imread('figures/Presnall_Gasparik_1990_en_melting.png')
ax.imshow(fig1, extent=[0., 18., 1400., 2400.], aspect='auto') # in C

pressures = np.linspace(1.e5, 20.e9, 101)
def deltaS(T, P, S, m):
    m.set_state(P, T[0])
    return S - m.S

temperatures = np.array([fsolve(deltaS, [1800.], args=(P, Mg2Si2O6_liq.params['S_0'], Mg2Si2O6_liq))[0] for P in pressures])
ax.plot(pressures/1.e9, temperatures-273.15, label='Principal isentrope')

inv = []
for (m1, m2, P_guess, T_guess) in [(pren, en, 4.e9, 1800.),
                                   (en, hen, 4.e9, 1800.),
                                   (hen, maj, 14.e9, 2200.)]:
    pt = tools.invariant_point([Mg2Si2O6_liq, m1], [1., -1.],
                               [Mg2Si2O6_liq, m2], [1., -1.],
                               pressure_temperature_initial_guess=[P_guess, T_guess])
    inv.append(pt)
    print ('{0}-{1}-liq invariant: {2:.2f} GPa, {3:.0f} K'.format(m1.name, m2.name, pt[0]/1.e9, pt[1]))
    temperatures = np.linspace(1673., pt[1], 31)
    pressures = np.zeros_like(temperatures)
    for i, T in enumerate(temperatures):
        pressures[i] = tools.equilibrium_pressure([m1, m2], [1., -1.],
                                                  T, pressure_initial_guess = P_guess)

    ax.plot(pressures/1.e9, temperatures-273.15, label='{0}-{1}'.format(m1.name, m2.name))

ax.legend(loc='upper left')
plt.show()
for (m, pressures, T_guess) in [(en, np.linspace(inv[0][0], inv[1][0], 101), inv[1][1]-300.),
                                (hen, np.linspace(inv[1][0], inv[2][0], 101), inv[2][1]-300.),
                                (maj, np.linspace(inv[2][0], 20.e9, 101), inv[2][1]-300.)]:
    print('Plotting {0} liquidus'.format(m.name))
    temperatures = np.zeros_like(pressures)
    for i, P in enumerate(pressures):
        print('{0} GPa'.format(P/1.e9))
        temperatures[i] = tools.equilibrium_temperature([Mg2Si2O6_liq, m], [1., -1.],
                                                        P, temperature_initial_guess = T_guess)

    ax.plot(pressures/1.e9, temperatures-273.15, label='{0}-{1}'.format(m.name, Mg2Si2O6_liq.name))

ax.legend(loc='upper left')
plt.show()
               
ax.set_xlabel('Pressure (GPa)')
ax.set_ylabel('Temperature ($^{\circ}$C)')


fig.savefig("MgSiO3_melt_first_guess.pdf", bbox_inches='tight', dpi=100)


plt.show()











crst = minerals.HP_2011_ds62.crst()
qtz = minerals.HP_2011_ds62.q()
coe = minerals.HP_2011_ds62.coe()
stv = minerals.HP_2011_ds62.stv()
SiO2_liq = minerals.current_melts.SiO2_liquid()

SiO2_liq.set_state(1.e5, 1773.)
print(SiO2_liq.bulk_sound_velocity) # c.f. Table 6 of Kress et al., 1988 (should be 2500 +/- 10 m/s)
print(SiO2_liq.isothermal_compressibility) # c.f. Table 7 (derived property, should be 7.05e-11)

SiO2_liq.set_state(1.e5, 1673.)
print(SiO2_liq.bulk_sound_velocity) # c.f. Table 4 of Ai and Lange, 2008 (derived property, should be 7.15 +/- 0.042 e-11 )
print(SiO2_liq.isothermal_compressibility) # c.f. Table 4 of Ai and Lange, 2008 (derived property, should be 7.15 +/- 0.042 e-11 )

exit()


crst_B = minerals.BOZA_2017.beta_cristobalite()
qtz_B = minerals.BOZA_2017.beta_quartz()
coe_B = minerals.BOZA_2017.coesite()
stv_B = minerals.BOZA_2017.stishovite()
SiO2_liq_B = minerals.BOZA_2017.SiO2_liquid()

temperatures = np.linspace(1700., 2300., 101)
pressures = [1.e5] * len(temperatures)

fig = plt.figure()
ax_V = fig.add_subplot(4, 1, 1)
ax_G = fig.add_subplot(4, 1, 2)
ax_S = fig.add_subplot(4, 1, 3)
ax_C = fig.add_subplot(4, 1, 4)
ax_V.plot(temperatures, SiO2_liq_B.evaluate(['V'], pressures, temperatures)[0])
ax_V.plot(temperatures, crst_B.evaluate(['V'], pressures, temperatures)[0])
ax_V.plot(temperatures, crst.evaluate(['V'], pressures, temperatures)[0])


ax_G.plot(temperatures, (SiO2_liq_B.evaluate(['gibbs'], pressures, temperatures)[0] -
                         crst_B.evaluate(['gibbs'], pressures, temperatures)[0]))

ax_S.plot(temperatures, (SiO2_liq_B.evaluate(['S'], pressures, temperatures)[0] -
                         crst_B.evaluate(['S'], pressures, temperatures)[0]))


ax_C.plot(temperatures, SiO2_liq_B.evaluate(['heat_capacity_p'], pressures, temperatures)[0]) 
ax_C.plot(temperatures, crst_B.evaluate(['heat_capacity_p'], pressures, temperatures)[0]) 
ax_C.plot(temperatures, crst.evaluate(['heat_capacity_p'], pressures, temperatures)[0]) 
plt.show()





fig = plt.figure()
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
pressures = np.linspace(1.e5, 100.e9, 101)
for T in [2000., 3000., 4000., 5000.]:
    temperatures = np.array([T]*len(pressures))
    Cv, V = SiO2_liq.evaluate(['heat_capacity_v', 'V'], pressures, temperatures)
    ax[0].plot(pressures/1.e9, Cv, label='{0} K'.format(T))
    ax[1].plot(pressures/1.e9, V, label='{0} K'.format(T))

ax[0].set_xlabel('Pressure (GPa)')
ax[1].set_xlabel('Pressure (GPa)')
ax[0].set_ylabel('$C_V$ (J/K/mol)')
ax[1].set_ylabel('$V$ (m$^3$/mol)')
ax[0].legend(loc='lower left')
ax[1].legend(loc='upper right')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
fig1 = mpimg.imread('figures/Shen_Lazor_1995_SiO2_melt.png')
ax.imshow(fig1, extent=[0., 80., 1500.-273.15, 5000.-273.15], aspect='auto')

fig1 = mpimg.imread('figures/Zhang_et_al_1993_SiO2_melt.png')
ax.imshow(fig1, extent=[0., 15., 1400., 3000.], aspect='auto')


inv = []
for (m1, m2, P_guess, T_guess) in [(crst, qtz, 0.4e9, 1990.),
                                   (qtz, coe, 4.e9, 2500.),
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
    

for (m, pressures, T_guess) in [(crst, np.linspace(1.e5, inv[0][0], 101), inv[0][1]),
                                (qtz, np.linspace(inv[0][0], inv[1][0], 101), inv[1][1]),
                                (coe, np.linspace(inv[1][0], inv[2][0], 101), inv[2][1]),
                                (stv, np.linspace(inv[2][0], 80.e9, 101), inv[2][1])]:
    temperatures = np.zeros_like(pressures)
    for i, P in enumerate(pressures):
        temperatures[i] = tools.equilibrium_temperature([SiO2_liq, m], [1., -1.],
                                                        P, temperature_initial_guess = T_guess)

    ax.plot(pressures/1.e9, temperatures-273.15, label='{0}-{1}'.format(m.name, SiO2_liq.name))




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


fig.savefig("SiO2_melt_BOZA.pdf", bbox_inches='tight', dpi=100)


plt.show()




exit()

'''
Klim = liq.params['K_0']*np.power((1. - liq.params['Kprime_inf']/liq.params['Kprime_0']),
                                  liq.params['Kprime_0']/liq.params['Kprime_inf'])
Plim = Klim/(liq.params['Kprime_inf'] - liq.params['Kprime_0'])

Vlim = ( liq.params['V_0'] *
         np.power ( liq.params['Kprime_0'] /
                    (liq.params['Kprime_0'] - liq.params['Kprime_inf']),
                    liq.params['Kprime_0'] /
                    liq.params['Kprime_inf'] /
                    liq.params['Kprime_inf'] ) *
         np.exp(-1./liq.params['Kprime_inf']) )

print('{0} GPa'.format(Klim/1.e9))
'''


liq.set_state(100.e9, 2000.)
#tools.check_eos_consistency(liq, verbose=False)

fig = plt.figure()
ax_V = fig.add_subplot(3, 1, 1)
ax_C = fig.add_subplot(3, 1, 2)
ax_Kp = fig.add_subplot(3, 1, 3)
for (pressures, T) in [(np.linspace(1.e5, 10.e9, 101), 2000.),
                       (np.linspace(1.e9, 20.e9, 101), 3000.),
                       (np.linspace(2.e9, 40.e9, 101), 4000.)]:
                
    temperatures = [T] * len(pressures)
    volumes, C_p, K_T = liq.evaluate(['V', 'heat_capacity_p', 'isothermal_bulk_modulus'], pressures, temperatures)
    ax_V.plot(pressures/1.e9, volumes*1.e6, label='{0:.0f} K'.format(T))
    ax_C.plot(pressures/1.e9, C_p, label='{0:.0f} K'.format(T))
    ax_Kp.plot(pressures/1.e9, np.gradient(K_T, pressures), label='{0:.0f} K'.format(T))

ax_V.set_xlabel('P (GPa)')
ax_C.set_xlabel('P (GPa)')
ax_Kp.set_xlabel('P (GPa)')
ax_V.set_ylabel('Volume')
ax_C.set_ylabel('$C_p$')
ax_Kp.set_ylabel('K\'')
ax_V.legend(loc='upper right')
ax_C.legend(loc='upper right')
ax_Kp.legend(loc='upper right')
plt.show()
