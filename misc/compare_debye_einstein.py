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

from burnman.minerals import SLB_2011, HP_2011_ds62

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
       [SLB_2011.hp_clinoenstatite(), HP_2011_ds62.en()],
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
       [SLB_2011.al_perovskite(), HP_2011_ds62.ak()],
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
