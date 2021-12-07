import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

fig1 = mpimg.imread('figures/sio2_phase_diagram_cropped.png')

Sa = 41.34
Ha = -910648.
Va = 22.69e-6
Sb = 42.77
Hb = -909435.
Vb = 23.06e-6

DS = Sb - Sa
DH = Hb - Ha
DV = Vb - Va


def T(P):
    return (DH + P*DV)/DS

fig = plt.figure(figsize=(4, 4))
ax = [fig.add_subplot(1, 1, 1)]

ax[0].imshow(fig1, extent=[400., 3200., 0., 16.], aspect='auto')

pressures = np.linspace(0., 4.e9, 11)
temperatures = T(pressures)

ax[0].plot(temperatures-273.15, pressures/1.e9, color='red',
           label='linearised based on STP properties')
ax[0].set_xlabel('Temperature ($^{\circ}C$)')
ax[0].set_ylabel('Pressure (GPa)')
ax[0].legend()
fig.savefig('figures/SiO_phase_diagram.pdf')
plt.show()

fig = plt.figure(figsize=(4, 4))
ax = [fig.add_subplot(1, 1, 1)]

pressures = np.linspace(0., 4.e9, 11)
temperatures = T(pressures)

ax[0].plot(temperatures, pressures/1.e9, color='red')
ax[0].set_xlabel('Temperature (K)')
ax[0].set_ylabel('Pressure (GPa)')
ax[0].text(1000., 2., '$\\alpha$-qtz')
ax[0].text(1500., 1., '$\\beta$-qtz')
fig.savefig('figures/alpha_beta_qtz_transition.pdf')
plt.show()
