# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
Ternary where bulk moduli are 
"""
from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import minerals
from bridgmanite_endmembers import *
m1 = fesio3
m2 = alalo3
m3 = fealo3

from scipy.optimize import brentq



plt.rcParams['figure.figsize'] = 12, 4.5 # inches
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'


def meshgrid2(*arrs):
    arrs = tuple(reversed(arrs))
    lens = map(len, arrs)
    dim = len(arrs)
    sz = 1
    for s in lens:
        sz *= s
        ans = []
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)
    return tuple(ans)


compositions = []
res = 31
for i, p1 in enumerate(np.linspace(0., 1., res)):
    for p2 in np.linspace(0., 1. - p1, res - i):
        p3 = 1. - p1 - p2
        compositions.append([p1, p2, p3])

compositions = np.array(compositions)


cluster_size = 1
g = meshgrid2(*[range(cluster_size)]*len(compositions[0]))
cluster_compositions = [list(cluster) for cluster in np.vstack(map(np.ravel, g)).T
                        if np.sum(cluster) == cluster_size]

cluster_probabilities = []
for cluster_composition in cluster_compositions:
    cluster_probabilities.append(np.math.factorial(np.sum(cluster_composition)) /
                                 (np.math.factorial(cluster_composition[0]) *
                                  np.math.factorial(cluster_composition[1]) *
                                  np.math.factorial(cluster_composition[2])) *
                                 np.power(compositions.T[0], cluster_composition[0]) *
                                 np.power(compositions.T[1], cluster_composition[1]) *
                                 np.power(compositions.T[2], cluster_composition[2]))
    # the above is ok because np.power(0., 0) is evaluated as 1.

'''
for cluster_p in cluster_probabilities:
    plt.tripcolor(compositions.T[0], compositions.T[1], cluster_p)
    plt.show()
exit()
'''

def _deltaV(P, volume, temperature, m):
    m.set_state(P, temperature)
    return volume - m.V


def _deltaP(volume, pressure, temperature, composition, endmembers):
    Pi = []
    for e in endmembers:
        Pi.append(brentq(_deltaV,
                         e.method.pressure(temperature, volume, e.params) - 1.e9,
                         e.method.pressure(temperature, volume, e.params) + 1.e9,
                         args=(volume, temperature, e)))
        #Pi.append(e.method.pressure(temperature, volume, e.params))
    Pi = np.array(Pi)
    return pressure - composition.dot(Pi)


pressure = 50.e9
temperature = 300.

endmembers = [m1, m2, m3]
for e in endmembers:
    e.set_state(pressure, temperature)

endmember_volumes = np.array([m1.V, m2.V, m3.V])
endmember_helmholtz = np.array([m1.helmholtz, m2.helmholtz, m3.helmholtz])
    
ideal_volumes = compositions.dot(endmember_volumes)
ideal_helmholtz = compositions.dot(endmember_helmholtz)


cluster_helmholtz = []
for n_atoms in cluster_compositions:
    composition = np.array(map(float, n_atoms))/cluster_size
    brentq(_deltaP, 20.e-6, 27.62e-6, args=(pressure, temperature, composition, endmembers))
    endmember_F = np.array([m1.helmholtz, m2.helmholtz, m3.helmholtz])
    cluster_helmholtz.append((endmember_F - endmember_helmholtz).dot(composition))
    

cluster_helmholtz = np.array(cluster_probabilities).T.dot(np.array(cluster_helmholtz))


volumes = []
helmholtz = []
for composition in compositions:
    volumes.append(brentq(_deltaP, 20.e-6, 27.62e-6, args=(pressure, temperature, composition, endmembers))) 
    endmember_F = np.array([m1.helmholtz, m2.helmholtz, m3.helmholtz])
    helmholtz.append(endmember_F.dot(composition))
    
volumes = np.array(volumes)
helmholtz = np.array(helmholtz)

c = compositions.T

# For m1_params = {'V_0': 11.99e-5,
#             'K_0': 165.e9}
#m2_params = {'V_0': 12.e-5,
#             'K_0': 165.e9}
#m3_params = {'V_0': 12.01e-5,
#             'K_0': 165.e9}
# W = [[0., 4.*0.575, 4.*2.29], [0., 0., 4.*0.575], [0., 0., 0.]]
W = [[0., 4.*0.575, 4.*2.29], [0., 0., 4.*0.575], [0., 0., 0.]]
helmholtz_HP = (c[0]*c[1]*W[0][1] +
                c[0]*c[2]*W[0][2] +
                c[1]*c[2]*W[1][2]) 


V0s = np.array([e.params['V_0'] for e in endmembers])
K0s = np.array([e.params['K_0'] for e in endmembers])
Kprime0s = np.array([e.params['Kprime_0'] for e in endmembers])


fig = plt.figure()
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

for i in range(0, 2):
    ax[i].axis('off')

    for x in np.linspace(0.2, 0.8, 4):
        ax[i].plot([x/2., 1 - x/2.], [x, x], color='0.85', linestyle=':')
        ax[i].plot([x, x/2.], [0., x], color='0.85', linestyle=':')
        ax[i].plot([1. - x, 1. - x/2.], [0., x], color='0.85', linestyle=':')

CS0 = ax[0].tricontour(compositions.T[2] + 0.5*compositions.T[0],
                    compositions.T[0],
                    (volumes - ideal_volumes)*1.e6, colors='k')
ax[0].clabel(CS0, fontsize=10, inline=1, fmt='%1.3f')


CS1 = ax[1].tricontour(compositions.T[2] + 0.5*compositions.T[0],
                       compositions.T[0],
                       (helmholtz - cluster_helmholtz - ideal_helmholtz)/1000., colors='k')
ax[1].clabel(CS1, fontsize=10, inline=1, fmt='%1.1f')

for i in range(0, 2):
    ax[i].plot([0., 0.5, 1., 0.], [0., 1., 0., 0.], color='black')
    ax[i].text(0., 0., m2.name, horizontalalignment='right', verticalalignment='top')
    ax[i].text(0.5, 1., m1.name, horizontalalignment='center', verticalalignment='bottom')
    ax[i].text(1., 0., m3.name, horizontalalignment='left', verticalalignment='top')
    ax[i].set_xlim(-0.05, 1.05)

ax[0].text(0.65, 0.8, '$V_{excess}$ (cm$^3$/mol)',
           horizontalalignment='left', verticalalignment='bottom', size=12.)
ax[1].text(0.65, 0.8, '$\mathcal{F}_{excess}$ (kJ/mol)',
           horizontalalignment='left', verticalalignment='bottom', size=12.)

for j in range(0, 3):
    mid_F = [F for i, F in enumerate((helmholtz - cluster_helmholtz - ideal_helmholtz)/1000.) if compositions.T[j][i] == 0][15]
    mid_V = [V for i, V in enumerate((volumes - ideal_volumes)*1.e6) if compositions.T[j][i] == 0][15]
    print('{0}-{1}: {2:.2f} kJ/mol, {3:.4f} cm^3/mol'.format(endmembers[j-1].name, endmembers[j-2].name, 4.*mid_F, 4.*mid_V))
    
fig.savefig("FASO_bridgmanite_excesses.pdf", bbox_inches='tight', dpi=100)

plt.show()




