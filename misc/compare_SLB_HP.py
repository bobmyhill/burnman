# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


""" Generates a text table with mineral properties. Run 'python table.py latex' to write a tex version of the table to mytable.tex """
from __future__ import absolute_import
from __future__ import print_function


import os
import sys
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import matplotlib.pyplot as plt
import burnman
import numpy as np

from burnman.minerals import SLB_2011, HP_2011_ds62, HHPH_2013

meq = [[SLB_2011.anorthite(), HP_2011_ds62.an()],
       [SLB_2011.albite(), HP_2011_ds62.ab()],
       [SLB_2011.spinel(), HP_2011_ds62.sp()],
       [SLB_2011.hercynite(), HP_2011_ds62.herc()],
       [SLB_2011.forsterite(), HP_2011_ds62.fo()],
       [SLB_2011.fayalite(), HP_2011_ds62.fa()],
       [SLB_2011.mg_wadsleyite(), HP_2011_ds62.mwd()],
       [SLB_2011.fe_wadsleyite(), HP_2011_ds62.fwd()],
       [SLB_2011.mg_ringwoodite(), HP_2011_ds62.mrw()],
       [SLB_2011.fe_ringwoodite(), HP_2011_ds62.frw()],
       [SLB_2011.enstatite(), HP_2011_ds62.en()],
       [SLB_2011.ferrosilite(), HP_2011_ds62.fs()],
       [SLB_2011.mg_tschermaks(), HP_2011_ds62.mgts()],
       [SLB_2011.ortho_diopside(), HP_2011_ds62.di()],
       [SLB_2011.diopside(), HP_2011_ds62.di()],
       [SLB_2011.hedenbergite(), HP_2011_ds62.hed()],
       [SLB_2011.clinoenstatite(), HP_2011_ds62.cen()],
       [SLB_2011.ca_tschermaks(), HP_2011_ds62.cats()],
       [SLB_2011.jadeite(), HP_2011_ds62.jd()],
       [SLB_2011.hp_clinoenstatite(), HP_2011_ds62.hen()],
       [SLB_2011.hp_clinoferrosilite(), HP_2011_ds62.fs()],
       [SLB_2011.ca_perovskite(), HP_2011_ds62.cpv()],
       [SLB_2011.mg_akimotoite(), HP_2011_ds62.mak()],
       [SLB_2011.fe_akimotoite(), HP_2011_ds62.fak()],
       [SLB_2011.corundum(), HP_2011_ds62.cor()],
       [SLB_2011.pyrope(), HP_2011_ds62.py()],
       [SLB_2011.almandine(), HP_2011_ds62.alm()],
       [SLB_2011.grossular(), HP_2011_ds62.gr()],
       [SLB_2011.mg_majorite(), HP_2011_ds62.maj()],
       #[SLB_2011.jd_majorite(), HP_2011_ds62.()],
       [SLB_2011.quartz(), HP_2011_ds62.q()],
       [SLB_2011.coesite(), HP_2011_ds62.coe()],
       [SLB_2011.stishovite(), HP_2011_ds62.stv()],
       #[SLB_2011.seifertite(), HP_2011_ds62.seif()],
       [SLB_2011.mg_perovskite(), HP_2011_ds62.mpv()],
       [SLB_2011.fe_perovskite(), HP_2011_ds62.fpv()],
       #[SLB_2011.mg_perovskite(), HHPH_2013.mpv()],
       #[SLB_2011.fe_perovskite(), HHPH_2013.fpv()],
       [SLB_2011.al_perovskite(), HHPH_2013.apv()],
       #[SLB_2011.mg_post_perovskite(), HP_2011_ds62.()],
       #[SLB_2011.fe_post_perovskite(), HP_2011_ds62.()],
       #[SLB_2011.al_post_perovskite(), HP_2011_ds62.()],
       [SLB_2011.periclase(), HP_2011_ds62.per()],
       [SLB_2011.wuestite(), HP_2011_ds62.fper()],
       #[SLB_2011.mg_ca_ferrite(), HP_2011_ds62.()],
       #[SLB_2011.fe_ca_ferrite(), HP_2011_ds62.()],
       #[SLB_2011.na_ca_ferrite(), HP_2011_ds62.()],
       [SLB_2011.kyanite(), HP_2011_ds62.ky()]]

#[SLB_2011.nepheline(), HP_2011_ds62.neph()],

plt.plot([300., 800.], [300., 800.])
for m_SLB, m_HP in meq:
    plt.scatter(m_HP.params['T_einstein'], np.cbrt(np.pi/6.)*m_SLB.params['Debye_0'])

plt.xlabel('HP Einstein temperature')
plt.ylabel('SLB equivalent Einstein temperature (converted from Debye)')
plt.show()




S_el_std = {'Na':  51.30, # 51.3
            'Mg':  32.68, # 32.7
            'Al':  28.35, # 28.3
            'Si':  18.81, # 18.8
            'K':   64.68, # 64.7
            'Ca':  41.63, # 41.6
            'Ti':  30.72, # 30.7
            'Mn':  32.22, # 32.0
            'Fe':  27.28, # 27.3
            'Ni':  29.80, # 29.9
            'Zr':  39.18, # 39.0
            'H':   65.34, # 65.35 for 0.5 H2 # atomic is 114.7
            'C':    5.74, # 5.7 for graphite
            'Cu':  33.15, # 33.2
            'Cr':  25.543, # 23.8 !!!!! quite different from PerpleX
            'Cl': 111.54, # 111.55 for 0.5Cl2
            'O':  102.575, # 102.6 for 0.5O2
            'S':   32.07} # 32.1 for crystalline rhombic sulfur

# values from PerpleX hp633ver.dat
# commented values from http://www.update.uu.se/~jolkkonen/pdf/CRC_TD.pdf


fig = plt.figure()
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

P = 20.e9
T = 1000.
dVT = []
dST = []
dGT = []
n = []
for m_SLB, m_HP in meq:
    m_SLB.set_state(P, T)
    m_HP.set_state(P, T)

    Sel = sum(m_HP.params['formula'][element] * S_el_std[element] for element in m_SLB.params['formula'])

    dVT.append(m_SLB.V - (m_SLB.params['n']/m_HP.params['n'])*(m_HP.V))
    dST.append(m_SLB.S - (m_SLB.params['n']/m_HP.params['n'])*(m_HP.S))
    dGT.append(m_SLB.gibbs - (m_SLB.params['n']/m_HP.params['n'])*(m_HP.gibbs + Sel*300.))
    n.append(m_SLB.params['n'])
    #plt.scatter(m_SLB.params['n'], dG0)
    if np.abs(dGT[-1]/n[-1]) > 750.:
        print(m_SLB.name, m_HP.name, dST[-1], dGT[-1], 'Teq = {0}'.format(dGT[-1]/dST[-1] + T))

    ax[0].annotate(m_HP.name, (dST[-1]/n[-1], dGT[-1]/n[-1]))
    ax[1].annotate(m_HP.name, (dVT[-1]*1.e6/n[-1], dGT[-1]/n[-1]))

dVT = np.array(dVT)
dST = np.array(dST)
dGT = np.array(dGT)
n= np.array(n)

ax[0].scatter(dST/n, dGT/n, c=n)
#ax[0].colorbar(label='# atoms (SLB)')
ax[0].plot([-6, 6], [6000, -6000], label='$\Delta\mathcal{{G}} = 0$ @ {0} K'.format(-1000.+T))
ax[0].plot([-6, 6], [3000, -3000], label='$\Delta\mathcal{{G}} = 0$ @ {0} K'.format(-500+T))
ax[0].plot([-6, 6], [0, 0], label='$\Delta\mathcal{{G}} = 0$ @ {0} K'.format(T))
ax[0].plot([-6, 6], [-3000, 3000], label='$\Delta\mathcal{{G}} = 0$ @ {0} K'.format(500+T))
ax[0].plot([-6, 6], [-6000, 6000], label='$\Delta\mathcal{{G}} = 0$ @ {0} K'.format(1000.+T))
ax[0].set_xlabel('$S_{SLB} - S_{HP}$ (J/K/mol-atom)')
ax[0].set_ylabel('$\mathcal{G}_{SLB} - \mathcal{G}_{HP}$ (J/mol-atom)')
ax[0].legend()


ax[1].scatter(dVT/n*1.e6, dGT/n, c=n)
#ax[1].colorbar(label='# atoms (SLB)')
ax[1].plot([-0.3, 0.3], [-6000, 6000], label='$\Delta\mathcal{{G}} = 0$ @ {0} GPa'.format(-20.+P/1.e9))
ax[1].plot([-0.3, 0.3], [-3000, 3000], label='$\Delta\mathcal{{G}} = 0$ @ {0} GPa'.format(-10.+P/1.e9))
ax[1].plot([-0.3, 0.3], [0, 0], label='$\Delta\mathcal{{G}} = 0$ @ {0} GPa'.format(P/1.e9))
ax[1].plot([-0.3, 0.3], [3000, -3000], label='$\Delta\mathcal{{G}} = 0$ @ {0} GPa'.format(10.+P/1.e9))
ax[1].plot([-0.3, 0.3], [6000, -6000], label='$\Delta\mathcal{{G}} = 0$ @ {0} GPa'.format(20.+P/1.e9))
ax[1].set_xlabel('$V_{SLB} - V_{HP}$ (cm^3/mol-atom)')
ax[1].set_ylabel('$\mathcal{G}_{SLB} - \mathcal{G}_{HP}$ (J/mol-atom)')
ax[1].legend()


ax[0].set_title('{0} GPa, {1} K'.format(P/1.e9, T))
ax[1].set_title('{0} GPa, {1} K'.format(P/1.e9, T))

plt.show()
