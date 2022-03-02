import numpy as np
import matplotlib.pyplot as plt
from burnman.minerals.HP_2011_ds62 import q
from scipy.integrate import cumtrapz



def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )

qtz = q()


fig = plt.figure(figsize=(6, 3))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

T = np.linspace(300., 800., 101)
Cp = qtz.evaluate(['C_p'], T*0. + 1.e5, T)[0]

ax[0].fill_between(T, T*0., Cp, alpha=0.3)
ax[1].fill_between(T, T*0., Cp/T, alpha=0.3)
ax[0].plot(T, Cp)
ax[1].plot(T, Cp/T)


ax[0].set_xlabel('Temperature (K)')
ax[0].set_ylabel('Heat capacity (J/K/mol)')
ax[1].set_xlabel('Temperature (K)')
ax[1].set_ylabel('Heat capacity / T (J/K$^2$/mol)')

for i in range(2):
    ax[i].set_ylim(0, )

fig.set_tight_layout(True)

fig.savefig('qtz_Cp_integrals.pdf')
plt.show()


fig = plt.figure(figsize=(5, 3))
ax = [fig.add_subplot(1, 1, i) for i in range(1, 2)]

P = np.linspace(1.e5, 2.e9, 101)
V = qtz.evaluate(['V'], P, P*0. + 800.)[0]

#ax[0].fill_between(T, T*0., Cp, alpha=0.3)
#ax[1].fill_between(T, T*0., Cp/T, alpha=0.3)
ax[0].plot(P/1.e9, V*1.e6)
#ax[1].plot(T, Cp/T)


ax[0].set_xlabel('Pressure (GPa)')
ax[0].set_ylabel('Volume (cm$^3$/mol)')
#ax[1].set_xlabel('Temperature (K)')
#ax[1].set_ylabel('Heat capacity / T (J/K$^2$/mol)')


fig.set_tight_layout(True)

fig.savefig('qtz_volume_800K.pdf')
plt.show()



fig = plt.figure(figsize=(5, 3))
ax = [fig.add_subplot(1, 1, i) for i in range(1, 2)]


ax[0].set_ylabel('Pressure (GPa)')
ax[0].set_xlabel('Temperature (K)')

#ax[0].set_xlim(0., 1000.)
#ax[0].set_ylim(0., 5.)

Pf = 4.e9
Pi = 1.e5
Tf = 900.
Ti = 298.15

ln, = ax[0].plot([Ti, Tf, Tf], [Pi/1.e9, Pi/1.e9, Pf/1.e9], label='PT path')
add_arrow(ln)

qtz.set_state(Pf, Tf)
V = qtz.V

temperatures = np.linspace(Ti, Tf, 101)
pressures = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    qtz.set_state_with_volume(V, T)
    pressures[i] = qtz.pressure

T = [298.15]
P = [1.e5]

T.extend(list(temperatures))
P.extend(list(pressures))

pressures = np.array(P)
temperatures = np.array(T)
ln,  = ax[0].plot(temperatures, pressures/1.e9, linestyle=':', label='TV path')
add_arrow(ln)

fig.set_tight_layout(True)

ax[0].legend()
fig.savefig('integration_paths.pdf')
plt.show()