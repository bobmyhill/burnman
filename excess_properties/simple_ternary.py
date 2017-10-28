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

from scipy.optimize import brentq


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

# Periclase
m1_params = {'name':'MgO',
             'V_0': 11.24e-6,
             'K_0': 160.2e9}
m2_params = {'name': 'FeO (HS)',
             'V_0': 12.26e-6,
             'K_0': 160.0e9}
m3_params = {'name': 'FeO (LS)',
             'V_0': 11.24e-6 - 0.309e-6,
             'K_0': 160.2e9 + 25.144e9}

for params in [m1_params, m2_params, m3_params]:
    params['equation_of_state'] = 'bm3'
    params['Kprime_0'] = 4.
    params['P_0'] = 1.e5
    params['T_0'] = 300.

m1 = burnman.Mineral(params = m1_params)
m2 = burnman.Mineral(params = m2_params)
m3 = burnman.Mineral(params = m3_params)


compositions = []
for i, p1 in enumerate(np.linspace(0., 1., 31)):
    for p2 in np.linspace(0., 1. - p1, 31 - i):
        p3 = 1. - p1 - p2
        compositions.append([p1, p2, p3])

compositions = np.array(compositions)


cluster_size = 2
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


pressure = 100.e9
temperature = 300.

endmembers = [m1, m2, m3]
for e in endmembers:
    e.set_state(pressure, temperature)

endmember_volumes = np.array([m1.V, m2.V, m3.V])
endmember_names = np.array([m1.name, m2.name, m3.name])
print(endmember_names)
print(endmember_volumes)
    
ideal_volumes = compositions.dot(endmember_volumes)


cluster_helmholtz = []
for n_atoms in cluster_compositions:
    composition = np.array(map(float, n_atoms))/cluster_size
    brentq(_deltaP, 6.e-6, 15.e-6, args=(pressure, temperature, composition, endmembers))
    endmember_F = np.array([m1.helmholtz, m2.helmholtz, m3.helmholtz])
    cluster_helmholtz.append(endmember_F.dot(composition))
    

cluster_helmholtz = np.array(cluster_probabilities).T.dot(np.array(cluster_helmholtz))


volumes = []
helmholtz = []
for composition in compositions:
    volumes.append(brentq(_deltaP, 6.e-6, 15.e-6, args=(pressure, temperature, composition, endmembers))) 
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

volumes_approx = np.array([np.power(np.sum(x*K0s) /
                                    np.sum(x*K0s*np.power(V0s, np.sum(x*Kprime0s))),
                                    np.array(-1./np.sum(x*Kprime0s))) for x in compositions])

energy_approx = ( np.array([np.sum(x*K0s /
                                   (Kprime0s*(Kprime0s - 1.)) *
                                   (volumes_approx[i] * np.power(volumes_approx[i]/V0s, -Kprime0s) - V0s
                                    + (volumes_approx[i] - V0s)*(Kprime0s - 1.)))
                            for i, x in enumerate(compositions)]) )

plt.tricontour(compositions.T[0], compositions.T[1],
               volumes - ideal_volumes)
plt.tricontour(compositions.T[0], compositions.T[1],
               volumes_approx - ideal_volumes, linestyles='dotted')
plt.colorbar(label='$V_{excess}$ m$^3$/mol')
plt.show()

plt.tricontour(compositions.T[0], compositions.T[1], helmholtz - cluster_helmholtz)
plt.tricontour(compositions.T[0], compositions.T[1], energy_approx - cluster_helmholtz)
#plt.tricontour(compositions.T[0], compositions.T[1], helmholtz_HP, linestyles='dotted')
plt.colorbar(label='$\mathcal{F}_{excess}$ J/mol')
plt.show()

