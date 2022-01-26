import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

T0 = 100.
DeltaT = 1100 - T0
Cp = 887
rho = 2648.
k = 7.7
a = 750./2.
kappa = k/(rho*Cp)


def yr_to_sec(yr):
    return yr*365.25*24.*60*60


def T(x, t):
    return T0 + DeltaT/2.*(erf((a - x)/np.sqrt(4.*kappa*t))
                           + erf((a + x)/np.sqrt(4.*kappa*t)))


ts = np.linspace(1.e-10, 1000., 6)
xs = np.linspace(0., 1600., 201)


fig = plt.figure(figsize=(5, 3))
ax = [fig.add_subplot(1, 1, 1)]

for t in ts:
    Ts = T(xs, yr_to_sec(t))
    ln, = ax[0].plot(xs, Ts, label=f'{t:.0f} yr')

i = 0
too_hot = True
while too_hot:
    i += 1
    if Ts[i] < 400.:
        too_hot = False

ax[0].text(xs[i], 450., f'aureole limit at {t:.0f} yr',
           rotation=90., size=8.,
           c=ln.get_color())

ax[0].legend()
ax[0].set_xlabel('Distance from center of intrusion (m)')
ax[0].set_ylabel('Temperature ($^{\circ}$C)')
ax[0].set_xlim(0., 1600.)
ax[0].set_ylim(0.,)
ax[0].plot([0., 1600.], [400., 400.], label=':',
           c='k')
ax[0].text(800., 420., 'biotite-in')
fig.set_tight_layout(True)
fig.savefig('sill_cooling.pdf')
plt.show()
