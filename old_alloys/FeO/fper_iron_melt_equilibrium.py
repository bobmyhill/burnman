import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    Komabayashi_2014,\
    Myhill_calibration_iron,\
    Fe_Si_O
from Fe_Si_O_liquid_models import *
from burnman.solidsolution import SolidSolution
import numpy as np
from scipy.optimize import fsolve

Fe_fcc = Myhill_calibration_iron.fcc_iron()
Fe_hcp = Myhill_calibration_iron.hcp_iron()
Fe_liq = Myhill_calibration_iron.liquid_iron()


T_ref = 1809.
P_ref = 50.e9
HP_convert(Fe_fcc, 300., 2200., T_ref, P_ref)
HP_convert(Fe_hcp, 300., 2200., T_ref, P_ref)
HP_convert(Fe_liq, 1809., 2400., T_ref, P_ref)

class ferropericlase(SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='ferropericlase'
        self.endmembers = [[burnman.minerals.HP_2011_ds62.per(), '[Mg]O'],[burnman.minerals.HP_2011_ds62.fper(), '[Fe]O']]
        self.type='symmetric'
        self.enthalpy_interaction=[[13.e3]]

        SolidSolution.__init__(self, molar_fractions)


Ozawa_data = []
f = open('data/Ozawa_et_al_2008_fper_iron.dat', 'r')
datalines = [ line.strip().split() for idx, line in enumerate(f.read().split('\n')) if line.strip() and idx>0 ]
for content in datalines:
    if content[0] != '%':
        Ozawa_data.append([float(content[1])*1.e9, float(content[2]), float(content[2])/10.,
                           float(content[6]), float(content[7]),
                           float(content[8]), float(content[9])]) 

#P, T, Terr, O_melt, O_melt_err, XFeO, XFeO_err = zip(*Ozawa_data)
#O_melt = np.array(O_melt)
#XFeO_melt = O_melt / (100.-O_melt)

#plt.errorbar(O_melt, XFeO, xerr = O_melt_err, yerr = XFeO_err, linestyle='none')
#plt.show()

fper = ferropericlase()
FeSiO_melt = metallic_Fe_Si_O_liquid()

def fper_melt_eqm(XFeO_melt, P, T, XFeO_per):
    fper.set_composition([1.-XFeO_per, XFeO_per])
    fper.set_state(P, T)

    FeSiO_melt.set_composition([1. - XFeO_melt[0], 0., XFeO_melt[0]])
    FeSiO_melt.set_state(P, T)

    #print FeSiO_melt.partial_gibbs, XFeO_melt[0], fper.partial_gibbs[1] - FeSiO_melt.partial_gibbs[2]
    return fper.partial_gibbs[1] - FeSiO_melt.partial_gibbs[2]

for datum in Ozawa_data:
    P, T, Terr, O_melt, O_melt_err, XFeO, XFeO_err = datum
    XFeO_melt = O_melt / (100.-O_melt)
    print P/1.e9, T, XFeO, XFeO_melt, fsolve(fper_melt_eqm, [XFeO_melt/10.], args=(P, T, XFeO))[0]

XFeO_mw = 0.2
temperatures = np.linspace(2873., 3173., 4)
pressures = np.linspace(20.e9, 140.e9, 61)
for T in temperatures:
    print T
    lnKd = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        XFeO_melt = fsolve(fper_melt_eqm, [0.01], args=(P, T, XFeO_mw))[0]
        lnKd[i] = np.log(XFeO_melt/XFeO_mw)
    plt.plot(pressures, lnKd, label=str(T)+'K')

plt.title('FeO partitioning between periclase and metallic melt')
plt.xlabel('Pressure (GPa)')
plt.ylabel('ln KD')
plt.legend(loc='upper right')
plt.show()
