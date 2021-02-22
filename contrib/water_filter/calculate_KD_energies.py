from __future__ import absolute_import
from __future__ import print_function
import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.abspath('../..'))

import burnman
from burnman.minerals import SLB_2011
from burnman import equilibrate

def calculate_reaction_energies(m1, m2, P, T):
    m1.set_state(13.5e9, 1700.)
    m1.set_composition([0.9, 0.1])

    m2.set_state(13.5e9, 1700.)
    m2.set_composition([0.9, 0.1])

    G1 = m1.partial_gibbs/2.
    S1 = m1.partial_entropies/2.
    V1 = m1.partial_volumes/2.
    E1 = G1 + T*S1 - P*V1

    G2 = m2.partial_gibbs/2.
    S2 = m2.partial_entropies/2.
    V2 = m2.partial_volumes/2.
    E2 = G2 + T*S2 - P*V2


    dG12 = G2[1] + G1[0] - G2[0] - G1[1]
    dE12 = E2[1] + E1[0] - E2[0] - E1[1]
    dS12 = S2[1] + S1[0] - S2[0] - S1[1]
    dV12 = V2[1] + V1[0] - V2[0] - V1[1]

    return np.array([dE12, dS12, dV12])


ol = SLB_2011.mg_fe_olivine()
wad = SLB_2011.mg_fe_wadsleyite()
ring = SLB_2011.mg_fe_ringwoodite()

P = 23.5e9
T = 1800.
bdg = SLB_2011.mg_fe_bridgmanite()
per = SLB_2011.ferropericlase()
bdg.guess = np.array([0.9, 0.1])
per.guess = np.array([0.9, 0.1])
composition = {'Fe': 0.2, 'Mg': 1.8, 'Si': 1., 'O':4.0}
assemblage = burnman.Composite([bdg, per])
equality_constraints = [('P', P), ('T', T)]
sol, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False)

ring.set_state(P, T)
ring.set_composition([0.9, 0.1])

m0 = bdg
m1 = per
m2 = ring

G0 = m0.partial_gibbs/2.
S0 = m0.partial_entropies/2.
V0 = m0.partial_volumes/2.
E0 = G0 + T*S0 - P*V0

G1 = m1.partial_gibbs/2.
S1 = m1.partial_entropies/2.
V1 = m1.partial_volumes/2.
E1 = G1 + T*S1 - P*V1

G2 = m2.partial_gibbs/2.
S2 = m2.partial_entropies/2.
V2 = m2.partial_volumes/2.
E2 = G2 + T*S2 - P*V2


dG = -(G2[1] + (G0[0] + G1[0]) - G2[0] - (G0[1] + G1[1]))
dE = -(E2[1] + (E0[0] + E1[0]) - E2[0] - (E0[1] + E1[1]))
dS = -(S2[1] + (S0[0] + S1[0]) - S2[0] - (S0[1] + S1[1]))
dV = -(V2[1] + (V0[0] + V1[0]) - V2[0] - (V0[1] + V1[1]))




# These are all one-mole Fe-Mg exchanges
wad_ol = calculate_reaction_energies(ol, wad, 13.5e9, 1700.)  # fe_wad + mg_ol - fe_ol - mg_wad
ring_wad = calculate_reaction_energies(wad, ring, 17.e9, 1700.)
lm_ring = np.array([dE, dS, dV])


# For the olivine field melt, we use the pressure dependence
# suggested by the experiments of Mibe.

# For extrapolation, however, we remove the volumetric component
# (which arises from a change in NBO/T) and use the equivalent energy
# at 13.e5 GPa

f = 0.6
ol_melt = np.array([-8.31446*2273*np.log(0.30) -5.5e3*f,
                    0., 5.e-7*f]) # fe_ol + mg_melt - fe_melt - mg_ol

wad_melt = wad_ol + ol_melt
ring_melt = ring_wad + wad_melt
lm_melt = lm_ring + ring_melt

print(ol_melt)
print(wad_melt)
print(ring_melt)
print(lm_melt)

label=['ol-melt',
       'wad-melt',
       'ring-melt',
       'lm-melt']


Ps = np.linspace(5.e9, 25.e9, 101)
T =2273.
R = 8.31446
for i, (E, S, V) in enumerate([ol_melt, wad_melt, ring_melt, lm_melt]):

    KD = np.exp(-(E - T*S + Ps*V)/(R*T))

    plt.plot(Ps, KD, label=label[i])
plt.legend()
plt.show()
