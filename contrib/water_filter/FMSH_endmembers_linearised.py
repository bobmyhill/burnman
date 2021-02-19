from __future__ import absolute_import
from __future__ import print_function
import os.path
import sys
sys.path.insert(1, os.path.abspath('../..'))

import burnman
from burnman.minerals import SLB_2011, DKS_2013_liquids
from burnman.minerals.Pitzer_Sterner_1994 import H2O_Pitzer_Sterner

# Elemental entropies taken from PerpleX's hp633ver.dat
S_el_MgO = 135.2550
S_el_SiO2 = 223.9600
S_el_H2O = 233.2550


# Master endmembers
H2OL = H2O_Pitzer_Sterner()

maj = burnman.CombinedMineral([SLB_2011.mg_majorite(),
                               SLB_2011.almandine(),
                               SLB_2011.pyrope()],
                              [1./4., 0.1/3., -0.1/3.], name='maj')

wad = burnman.CombinedMineral([SLB_2011.mg_wadsleyite(),
                               SLB_2011.fe_wadsleyite()],
                              [0.9, 0.1], name='wadsleyite')

# Make other endmembers:
make_new = False
if make_new:
    fo = burnman.CombinedMineral([SLB_2011.forsterite(),
                                  SLB_2011.fayalite()],
                                 [0.9, 0.1], name='olivine')

    ring = burnman.CombinedMineral([SLB_2011.mg_ringwoodite(),
                                    SLB_2011.fe_ringwoodite()],
                                   [0.9, 0.1], name='ringwoodite')

    lm = burnman.Composite([SLB_2011.mg_perovskite(), SLB_2011.fe_perovskite(),
                            SLB_2011.periclase(), SLB_2011.wuestite()],
                           [0.47, 0.03, 0.43, 0.07], name='bdg+per')

    # We still compare the liquid to pure Mg-wadsleyite
    # Ohtani (1979) suggest that the melting temperature in the ringwoodite field
    # are pretty similar for both Mg2SiO4 and Fe2SiO4.
    mg_wad = SLB_2011.mg_wadsleyite()

    Mg2SiO4L = DKS_2013_liquids.Mg2SiO4_liquid()

    dS = 7.5  # de Koker uses 0 (obvs) to fit fo melting point at 1 bar. Something between 0 and 15 is ok.
    dV = 0  # must be > -2.e-7, because otherwise melt becomes denser than fo at the fo-wad-melt invariant
    Mg2SiO4L.property_modifiers = [['linear', {'delta_E': 0,
                                              'delta_S': dS, 'delta_V': dV}]]

    mg_wad.set_state(16.7e9, 2315+273.15)  # Presnall and Walter
    Mg2SiO4L.set_state(16.7e9, 2315+273.15)

    Mg2SiO4L.property_modifiers = [['linear', {'delta_E': mg_wad.gibbs - Mg2SiO4L.gibbs,
                                               'delta_S': dS, 'delta_V': dV}]]

    P_ref = 16.7e9
    T_ref = 2315+273.15
    wad.set_state(P_ref, T_ref)
    for m in [fo, Mg2SiO4L]:
        m.set_state(P_ref, T_ref)
        print(f'{m.name}: {m.molar_internal_energy - wad.molar_internal_energy}, {m.S - wad.S}, {m.V - wad.V}')

    P_ref = 22.68e9
    T_ref = 2210.  # near wad-ring-lm triple point
    wad.set_state(P_ref, T_ref)
    for m in [ring]:
        m.set_state(P_ref, T_ref)
        print(f'{m.name}: {m.molar_internal_energy - wad.molar_internal_energy}, {m.S - wad.S}, {m.V - wad.V}')

    for m in [lm]:
        m.set_state(P_ref, T_ref)
        print(f'{m.name}: {m.molar_internal_energy*2. - wad.molar_internal_energy}, {m.S*2. - wad.S}, {m.V*2. - wad.V}')

    exit()

fo = burnman.CombinedMineral([wad], [1.], [-16982., 4.6, 1.89e-6], name='forsterite')
ring = burnman.CombinedMineral([wad], [1.], [7060., -4.8, -8.12e-7], name='ring')
lm = burnman.CombinedMineral([wad], [1.], [77858., -0.645, -3.53e-6], name='bdg+per')


Mg2SiO4L = burnman.CombinedMineral([wad], [1.], [0., 0., 0.], name='liquid')

Mg2SiO4L.property_modifiers = [['linlog', {'delta_E': 175553., 'delta_S': 100.3, 'delta_V': -3.277e-6,
                                           'a': 2.60339693e-06, 'b': 2.64753089e-11, 'c': 1.18703511e+00}]]

H2MgSiO4fo = burnman.CombinedMineral([H2OL, maj], [1., 1.],
                                     [27500. + 1500.*15. + 12.e9*2e-6, 15., -2e-6])

# No clear pressure trend in Demouchy wad data
H2MgSiO4wad = burnman.CombinedMineral([H2OL, maj], [1., 1.],
                                      [12000. + 1500.*15., 15., 0.])

H2MgSiO4ring = burnman.CombinedMineral([H2OL, maj], [1., 1.],
                                       [10000. + 1500*15, 15., 0.])
