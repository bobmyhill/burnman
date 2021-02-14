from __future__ import absolute_import
from __future__ import print_function
import os.path
import sys
sys.path.insert(1, os.path.abspath('../..'))

import burnman
from burnman.minerals import \
    DKS_2013_liquids, \
    DKS_2013_solids, \
    SLB_2011
from burnman import constants
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.optimize import fsolve

import matplotlib.pyplot as plt

fo = SLB_2011.forsterite()
wad = SLB_2011.mg_wadsleyite()
ring = SLB_2011.mg_ringwoodite()
foL = DKS_2013_liquids.Mg2SiO4_liquid()
lm = burnman.Composite([SLB_2011.mg_perovskite(), SLB_2011.periclase()], [0.5, 0.5], name='bdg+per')

dS = 7.5 # de Koker uses 0 (obvs) to fit fo melting point at 1 bar. Something between 0 and 15 is ok.
dV = 0 # must be > -2.e-7, because otherwise melt becomes denser than fo at the fo-wad-melt invariant
foL.property_modifiers = [['linear', {'delta_E': 0,
                                      'delta_S': dS, 'delta_V': dV}]]

fo.set_state(16.7e9, 2315+273.15) # Presnall and Walter
foL.set_state(16.7e9, 2315+273.15)

foL.property_modifiers = [['linear', {'delta_E': fo.gibbs - foL.gibbs,
                                      'delta_S': dS, 'delta_V': dV}]]
burnman.Mineral.__init__(foL)

def eqm_T(T, mins, fs=[1., 1.]):
    def delta_G(P):
        mins[0].set_state(P[0], T)
        mins[1].set_state(P[0], T)
        return mins[0].gibbs*fs[0] - mins[1].gibbs*fs[1]
    return fsolve(delta_G, [20.e9])[0]

def eqm(P, mins, fs=[1., 1.]):
    def delta_G(T):
        mins[0].set_state(P, T[0])
        mins[1].set_state(P, T[0])
        return mins[0].gibbs*fs[0] - mins[1].gibbs*fs[1]
    return fsolve(delta_G, [2000.])[0]

def inv(mins, fs=[1., 1., 1]):
    def delta_G(args):
        P, T = args
        for m in mins:
            m.set_state(P, T)
        return [mins[0].gibbs*fs[0] - mins[1].gibbs*fs[1],
                mins[0].gibbs*fs[0] - mins[2].gibbs*fs[2]]
    return fsolve(delta_G, [22.4e9, 2500.])

fo_wad_melt = inv([fo, wad, foL])
wad_ring_lm = inv([wad, ring, lm], [1., 1., 2.])
wad_lm_melt = inv([wad, lm, foL], [1., 2., 1.])


fig = plt.figure(figsize=(12,4))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

temperatures = np.linspace(1400, fo_wad_melt[1], 21)
pressures = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    pressures[i] = eqm_T(T, [fo, wad], [1., 1.])

ax[0].plot(pressures/1.e9, temperatures, color='black')

temperatures = np.linspace(wad_ring_lm[1], wad_lm_melt[1], 21)
pressures = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    pressures[i] = eqm_T(T, [wad, lm], [1., 2.])

ax[0].plot(pressures/1.e9, temperatures, color='black')

temperatures = np.linspace(1400., wad_ring_lm[1], 21)
pressures = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    pressures[i] = eqm_T(T, [wad, ring], [1., 1.])

ax[0].plot(pressures/1.e9, temperatures, color='black')


temperatures = np.linspace(1400., wad_ring_lm[1], 21)
pressures = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    pressures[i] = eqm_T(T, [ring, lm], [1., 2.])

ax[0].plot(pressures/1.e9, temperatures, color='black')


pressures = np.linspace(0.e9, fo_wad_melt[0], 21)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = eqm(P, [fo, foL])

ax[0].plot(pressures/1.e9, temperatures, color='black')

pressures = np.linspace(fo_wad_melt[0], wad_lm_melt[0], 21)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = eqm(P, [wad, foL])

ax[0].plot(pressures/1.e9, temperatures, color='black')


pressures = np.linspace(wad_lm_melt[0], 26.e9, 21)
temperatures = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures[i] = eqm(P, [lm, foL], [2., 1.])

ax[0].plot(pressures/1.e9, temperatures, color='black')

ax[0].text(7., 1700.+273.15, 'fo')
ax[0].text(18., 1900.+273.15, 'wad')
ax[0].text(21., 1500.+273.15, 'ring')
ax[0].text(23., 2000.+273.15, 'bdg+per')
ax[0].text(7., 2500.+273.15, 'melt')

ax[0].set_ylim(1673, 2973)
ax[0].set_xlim(0, 26)

ax[0].set_xlabel('Pressure (GPa)')
ax[0].set_ylabel('Temperature (K)')

mins = [fo, wad, ring, lm, foL]
pressures = np.linspace(1.e5, 30.e9, 101)
T = 2073
temperatures = pressures*0. + T
for m in mins:
    rhos = m.evaluate(['density'], pressures, temperatures)[0]
    ax[1].plot(pressures/1.e9, rhos, label=f'{m.name}, {T} K'.replace('_', ' '))

ax[1].set_xlabel('Pressure (GPa)')
ax[1].set_ylabel('Density (kg/m$^3$)')
ax[1].legend()

fig.tight_layout()
fig.savefig('output_figures/Mg2SiO4_melting.pdf')
plt.show()
