import burnman
import numpy as np
import matplotlib.pyplot as plt

from slb_qtz import qtz_alpha, qtz_beta


# Bachheimer and Dolino, 1975
eta1 = 0.3388
b = -166.8
c = 726.6
T0 = 564.37+273.15
T1 = 573.94+273.15


qtz_HP = burnman.minerals.HP_2011_ds62.q()

pressures = np.linspace(1.e5, 10.e9, 101)
temperatures = 300. + pressures*0.
V0, Cp0, S0 = qtz_HP.evaluate(['V', 'C_p', 'S'], pressures, temperatures)

plt.plot(pressures, V0)
plt.show()

def F(T, eta):
    return 0.5*(T - T0)*eta*eta + b*np.power(eta, 4.)/4. + c*np.power(eta, 6.)/6.


def eta_eqm(T):
    etasqr = T*0.
    idx = np.argwhere(T < T1)
    etasqr[idx] = eta1*eta1*(1. + np.sqrt(1. - (T[idx] - T0)/(T1 - T0)))
    return np.sqrt(etasqr)


etas = np.linspace(0., 1.5, 10001)
temperatures = np.array([100., 300., 500., 700., 900.])
for T in temperatures:
    plt.plot(etas, F(T, etas))

plt.scatter(eta_eqm(temperatures), F(temperatures, eta_eqm(temperatures)))
plt.show()
exit()
temperatures = np.linspace(300., 1300., 10001)
plt.plot(temperatures, eta_eqm(temperatures))
plt.scatter(temperatures, eta_eqm(temperatures))
plt.show()
exit()
qtz_HP = burnman.minerals.HP_2011_ds62.q()
qtz_HP2 = burnman.minerals.HP_2011_ds62.q()
qtz_HP2.property_modifiers = []
qtz_SLB = burnman.minerals.SLB_2011.quartz()
qtz_SLB2 = burnman.minerals.SLB_2011.quartz()
qtz_SLB2.property_modifiers = []
qtz_SLB3 = burnman.minerals.SLB_2011.quartz()
qtz_SLB3.property_modifiers = []
qtz_SLB3.params['Debye_0'] = 760.



fig = plt.figure(figsize=(12, 4))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]


temperatures = np.linspace(300., 1300., 101)
pressures = temperatures * 0. + 1.e5
V0, Cp0, S0 = qtz_HP.evaluate(['V', 'C_p', 'S'], pressures, temperatures)
V1, Cp1, S1 = qtz_HP2.evaluate(['V', 'C_p', 'S'], pressures, temperatures)



ax[0].plot(temperatures, V0, label='HP')
ax[0].plot(temperatures, V1, label='trigHP')

ax[1].plot(temperatures, S0, label='HP')
ax[1].plot(temperatures, S1, label='trigHP')

temperatures = np.linspace(10., 1300., 101)
pressures = temperatures * 0. + 1.e5
G2, V2, Cp2, S2 = qtz_SLB.evaluate(['gibbs', 'V', 'C_p', 'S'], pressures, temperatures)
G3, V3, Cp3, S3 = qtz_alpha.evaluate(['gibbs', 'V', 'C_p', 'S'], pressures, temperatures)
G4, V4, Cp4, S4 = qtz_beta.evaluate(['gibbs', 'V', 'C_p', 'S'], pressures, temperatures)


ax[0].plot(temperatures, V2, label='SLB')
ax[0].plot(temperatures, V3, label='trigSLB')
ax[0].plot(temperatures, V4, label='hexSLB')

ax[1].plot(temperatures, S2, label='SLB')
ax[1].plot(temperatures, S3, label='alphaSLB')
ax[1].plot(temperatures, S4, label='betaSLB')


ax[2].plot(temperatures, G3, label='alphaSLB')
ax[2].plot(temperatures, G4, label='betaSLB')
ax[0].legend()
ax[2].legend()
plt.show()