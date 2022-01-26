import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt


def critical_thickness(Racr, kappa,
                       mu, rho, g,
                       alpha,
                       deltaT):
    return np.cbrt((Racr * kappa * mu)
                   / (rho*g*alpha*deltaT))
    

fig = plt.figure(figsize=(4, 3))
ax = [fig.add_subplot(1, 1, 1)]


ax[0].set_xscale('log')
ax[0].set_yscale('log')

mus = np.logspace(2, 10, 201)

Racr = 1600.
g = 9.81
rho = 2648.
Cp = 887.
kappa = 7.7/(rho*Cp)
alpha = 5.e-5 # Lesher and Spera 2015

for deltaT in [1., 10., 100.]:
    Ds = critical_thickness(Racr, kappa,
                            mus, rho, g,
                            alpha,
                            deltaT)
    ax[0].plot(mus, Ds, label=f'$\\Delta T = ${deltaT:.0f} K')

ax[0].set_xlabel('Viscosity (Pas)')
ax[0].set_ylabel('Critical thickness (m)')
ax[0].legend()
fig.set_tight_layout(True)
fig.savefig('critical_thickness.pdf')
plt.show()
