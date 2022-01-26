# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
an_di_melting
-------------
"""

from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
import ternary

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

data = np.loadtxt('data/melts_morb_frac_xtl.dat', unpack=True)

TC, P, mass = data[:3]
SiO2, TiO2, Al2O3, Fe2O3, Cr2O3 = data[3:8]
FeO, MgO, CaO, Na2O, K2O, P2O5, H2O = data[8:]


T = (FeO + Fe2O3 + MgO + Na2O + K2O) / 100.
AFM = np.array([MgO/T, (FeO + Fe2O3)/T, (Na2O + K2O)/T]).T

print(AFM[0:3])
figure = plt.figure()
ax = [figure.add_subplot(1, 1, 1)]
tax = ternary.TernaryAxesSubplot(ax=ax[0], scale=100)

tax.boundary()
fontsize = 14
offset = 0.30
tax.top_corner_label("FeO$^*$", fontsize=fontsize, offset=0.2)
tax.left_corner_label("A", fontsize=fontsize, offset=offset)
tax.right_corner_label("MgO", fontsize=fontsize, offset=offset)

import matplotlib.cm as cm

Tscale = (TC - TC[-1])/(TC[0] - TC[-1])

tax.scatter(AFM, c=Tscale, cmap=cm.viridis)


phases_in = [['ol', 1218.],
             ['aug', 1199.],
             ['sp', 1195.],
             ['ilm', 1102.],
             ['mag', 899.],
             ['fa', 895.]]

for ph_name, T_in in phases_in:
    d = tax.scatter([AFM[find_nearest(TC, T_in)]], c='k')
    x, y = d.get_offsets().data[0]
    ax[0].text(x+2., y, ph_name + '-in')

#leg = tax.legend(bbox_to_anchor=(1., 1.), bbox_transform=ax[0].transAxes, prop={'size': 8})
figure.set_tight_layout(True)
tax.get_axes().axis('off')
tax.clear_matplotlib_ticks()
tax.get_axes().set_aspect(1)
tax._redraw_labels()
figure.savefig('figures/morb_fractionation.pdf')
tax.show()


exit()
# Plot the data
# for i, line in enumerate(lines[2:5]):
#    tax.plot(line, linewidth=1.0,
#             color='grey', label=f"eutectic coexistence at {eut_temperatures[i+2]:.0f} K")

for i in range(len(T_contours_di)):
    tax.plot(X_liq_contours_di[i], linewidth=1.0,
             color='purple')
for i in range(len(T_contours_plag)):
    tax.plot(X_liq_contours_plag[i], linewidth=1.0,
             color='purple')
    
tax.plot(lines_batch[0], linewidth=1.0,
         color='grey', label=f"eutectic coexistence at {T_eut:.0f} K")
tax.plot(lines_batch[-1], linewidth=1.0,
         color='grey', label=f"eutectic coexistence at {T_sol:.0f} K")
tax.plot(lines_frac[0], linewidth=1.0,
         color='grey', label=f"eutectic coexistence at {T_eut_frac:.0f} K")

tax.scatter([[X_liq[1], X_liq[0], X_liq[2]]], color='red',
            label=f'bulk (liquidus at {T_liq:.0f} K)')
tax.plot([X_first_plag, liq_compositions1[0]],
         color='orange', label=f"plag-liq coexistence at {T_liq:.0f} K)")

tax.plot(liq_compositions1, linewidth=2.0,
         color='blue', label="liquid line of descent (batch)")
tax.plot(liq_compositions2, linewidth=2.0,
         color='blue')
tax.plot(liq_compositions_frac, linewidth=2.0,
         color='red', label="liquid line of descent (frac.)")

tax.ticks(axis='lbr', multiple=0.2, linewidth=1,
          offset=0.025, tick_formats="%.1f")

tax.scatter([[X_eut[1], X_eut[0], X_eut[2]]], marker='+', color='blue',
            label=f'eutectic liquid (batch) at {T_eut:.0f} K')
tax.scatter([[X_eut_frac[1], X_eut_frac[0], X_eut_frac[2]]], marker='+', color='red',
            label=f'eutectic liquid (frac.) at {T_eut_frac:.0f} K')
tax.scatter([[X_sol[1], X_sol[0], X_sol[2]]], marker='*', color='blue',
            label=f'last liquid (batch) at {T_sol:.0f} K')
tax.scatter([X_sol_frac], marker='*', color='red',
            label=f'last liquid (frac.) at {T_sol_frac:.0f} K')

leg = tax.legend(bbox_to_anchor=(1., 1.), bbox_transform=ax[0].transAxes, prop={'size': 8})
figure.set_tight_layout(True)
tax.get_axes().axis('off')
tax.clear_matplotlib_ticks()
tax.get_axes().set_aspect(1)
tax._redraw_labels()
figure.savefig('figures/di_ab_an_melting.pdf')
tax.show()
