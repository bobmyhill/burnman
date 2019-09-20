# This equation of state uses an Einstein function for K'
import numpy as np
from scipy.special import spence
from scipy.integrate import quad, cumtrapz
import matplotlib.pyplot as plt

def Kprime(pressure, K_0, Kprime_0, Kprime_inf, Kdprime_0):
    d = -2.*Kdprime_0/Kprime_0
    return (Kprime_0 - Kprime_inf)*(d*pressure/(np.exp(d*pressure) - 1.)) + Kprime_inf

def bulk_modulus(pressure, K_0, Kprime_0, Kprime_inf, Kdprime_0):
    d = -2.*Kdprime_0/Kprime_0
    p = np.where(np.abs(d*pressure) < 1.e-15, 1.e-15, d*pressure) # replace zeros with small number. easier than subbing in pi^2/6. via a masked array
    cplx = np.real(spence(1. + 0j - np.exp(p))
                   + p*np.log(1. + 0j - np.exp(p)))
    return K_0 + (Kprime_0 - Kprime_inf)*((cplx - np.pi*np.pi/6.)/d - d*pressure*pressure/2.) + Kprime_inf*pressure

def inv_K(pressure, K_0, Kprime_0, Kprime_inf, Kdprime_0):
    inv_K = 1./bulk_modulus(pressure, K_0, Kprime_0, Kprime_inf, Kdprime_0)
    return inv_K

def volume(pressure, K_0, Kprime_0, Kprime_inf, Kdprime_0):
    return np.exp(-quad(inv_K, 0., pressure, args=(K_0, Kprime_0, Kprime_inf, Kdprime_0))[0])


#print(bulk_modulus(0., 4., 2.))
pressures = np.linspace(-22, 300, 1001)

fig = plt.figure(figsize=(18, 12))
ax = [fig.add_subplot(2, 3, i) for i in range(1, 7)]

Kdprime_0 = -5./100.
Kp = Kprime(pressures, 100., 4., 3., Kdprime_0)
K = bulk_modulus(pressures, 100., 4., 3., Kdprime_0)
volumes = [volume(P, 100., 4., 3., Kdprime_0) for P in pressures]


ax[0].plot(pressures, volumes)
ax[1].plot(pressures, K)


ax[2].plot(volumes, Kp)

ax[3].plot(pressures, np.gradient(Kp, pressures))
#ax[1].plot(pressures, Kp)

# energy
ax[4].plot(np.power(volumes, 1./3.), -cumtrapz(pressures, volumes, initial=0))


ax[5].plot(pressures/K, 1./Kp)
ax[5].plot([0, 1], [0, 1], linestyle=':')
ax[5].plot([0, 0], [0, 1], linestyle='--')

for i in range(4):
    ax[i].set_xlabel('P (GPa)')
    ax[i].set_xlim(-25, 200)

#ax[1].set_ylabel("$K'$")
ax[0].set_ylabel("$V/V_0$")
ax[1].set_ylabel("$K_T$ (GPa)")

ax[2].set_xlabel('$V/V_0$')
ax[2].set_ylabel('$K\'$')
ax[2].set_xlim(0.6, 1.)
ax[2].set_ylim(2.5, 4.5)

ax[3].set_ylabel("$K''$ (1/GPa)")

ax[4].set_ylabel("$E$")
ax[4].set_xlabel('$(V/V_0)^{1/3}$')

ax[5].set_xlim(-0.05, 1.)
ax[5].set_ylabel("$1/K\'$")
ax[5].set_xlabel('$P/K_T$')


fig.savefig('myhill_test_eos.pdf')
plt.show()

"""
# Fake Einstein
def val(P, Pl, Pu, a, b, c):
    if P < Pl:
        return a + b*P
    elif P > Pu:
        return c
    else:
        # Cubic interpolation
        yl = a+b*Pl
        yu = c
        kl = b
        ku = 0
        t = (P - Pl)/(Pu - Pl)
        a = kl*(Pu - Pl) - (yu - yl)
        b = -ku*(Pu - Pl) + (yu - yl)

        return (1 - t)*yl + t*yu + t*(1 - t)*((1 - t)*a + t*b)

plt.plot(pressures, Kp)
K_primes = np.array([val(P, -10, 100, 4., -1/32, 3.1) for P in pressures])
plt.plot(pressures, K_primes, linestyle=':')
plt.plot(pressures, 100*np.gradient(K_primes, pressures))



#Kdprimes = np.array([val(P, -6, 8, -2, 0, 0) for P in pressures])

#plt.plot(pressures, cumtrapz(Kdprimes, pressures, initial=0) + K_primes[0])
plt.show()
"""
