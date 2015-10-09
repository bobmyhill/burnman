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
from fitting_functions import *
from burnman.solidsolution import SolidSolution
import numpy as np
from scipy.optimize import fsolve

Fe_fcc = Myhill_calibration_iron.fcc_iron_HP()
Fe_hcp = Myhill_calibration_iron.hcp_iron_HP()
Fe_liq = Myhill_calibration_iron.liquid_iron_HP()

# N.B. HHPH2013 have a positive interaction parameter (12 kJ/mol) 
# but also a negative DQF for fpv (-8 kJ/mol) ... these more or less 
# cancel out for up to 50 mol% Fe. SLB2011 favour ideal mixing. 
class mg_fe_bridgmanite(SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='ferropericlase'
        self.endmembers = [[burnman.minerals.HHPH_2013.mpv(), '[Mg]SiO3'],[burnman.minerals.HHPH_2013.fpv(), '[Fe]SiO3']]
        self.type='symmetric'
        self.enthalpy_interaction=[[0.e3]]

        SolidSolution.__init__(self, molar_fractions)


Ozawa_data = []
f = open('data/Ozawa_et_al_2009_pv_melt_compositions.dat', 'r')
datalines = [ line.strip().split() for idx, line in enumerate(f.read().split('\n')) if line.strip() and idx>0 ]
for content in datalines:
    if content[0] != '%':
        # P, T, O (mol), O err, Si (mol), Si err, XFeSiO3, err
        Ozawa_data.append([float(content[2])*1.e9, float(content[3]),
                           float(content[9]), float(content[10]),
                           float(content[11]), float(content[12]),
                           float(content[13]), float(content[14])]) 



bdg = mg_fe_bridgmanite()
FeSiO_melt = metallic_Fe_Si_O_liquid()

# Check FeSi melt

B20=minerals.Fe_Si_O.FeSi_B20()
B2=minerals.Fe_Si_O.FeSi_B2()

'''
FeSiO_melt.set_composition([0.5, 0.5, 0.0])

pressures = np.linspace(1.e9, 200.e9, 101)
temperatures_B2 = np.empty_like(pressures)
temperatures_B20 = np.empty_like(pressures)
for i, P in enumerate(pressures):
    temperatures_B2[i] = fsolve(eqm_temperature([B2, FeSiO_melt], [1., -1.]), [2000.], args=(P))[0]
    temperatures_B20[i] = fsolve(eqm_temperature([B20, FeSiO_melt], [1., -2.]), [2000.], args=(P))[0]

plt.plot(pressures/1.e9, temperatures_B2, label='B2')
plt.plot(pressures/1.e9, temperatures_B20, label='B20')
plt.legend(loc='lower right')
plt.show()


T = 3000.
excess_gibbs = np.empty_like(pressures)
for i, P in enumerate(pressures):
    FeSiO_melt.set_state(P, T)
    excess_gibbs[i] = FeSiO_melt.excess_gibbs

plt.plot(pressures/1.e9, excess_gibbs)
plt.show()
'''
P = 1.e5
T = 1600. + 273.15
FeSiO_melt.set_composition([0.5, 0.5, 0.0])
FeSiO_melt.set_state(P, T)
print FeSiO_melt.solution_model.Wg
T = 1500. + 273.15
FeSiO_melt.set_state(P, T)
print FeSiO_melt.solution_model.Wg


##################
##################
##################

def bdg_melt_eqm(XFeSiO3_bdg, P, T, XFeO_melt, XSi_melt):
    bdg.set_composition([1.-XFeSiO3_bdg[0], XFeSiO3_bdg[0]])
    bdg.set_state(P, T)

    FeSiO_melt.set_composition([1. - XFeO_melt - XSi_melt, 
                                XSi_melt, XFeO_melt])
    FeSiO_melt.set_state(P, T)

    mu_melt = FeSiO_melt.partial_gibbs

    mu_FeSiO3 = 3.*mu_melt[2] + mu_melt[1] - 2.*mu_melt[0]
    mu_FeSiO3 = burnman.chemicalpotentials.chemical_potentials([FeSiO_melt], [bdg.endmembers[1][0].params['formula']])[0]
    return bdg.partial_gibbs[1] - mu_FeSiO3

for datum in Ozawa_data:
    # P, T, O (mol), O err, Si (mol), Si err, XFeSiO3, err
    P, T, O_melt, O_melt_err, Si_melt, Si_melt_err, XFeSiO3, XFeSiO3_err = datum
    XFeO_melt = O_melt / (1.-O_melt)
    XSi_melt = Si_melt / (1.-O_melt)
    XFe_melt = 1. - XFeO_melt - XSi_melt
    #print P/1.e9, T, XFe_melt, XSi_melt, XFeO_melt, XFeSiO3
    print P/1.e9, T, XFeO_melt, XSi_melt, XFeSiO3, fsolve(bdg_melt_eqm, [0.99], args=(P, T, XFeO_melt, XSi_melt))[0]


P = 10.e9
T = 300.
FeSiO_melt.set_composition([0., 0.5, 0.5])
FeSiO_melt.set_state(P, T)
print FeSiO_melt.excess_volume


def melt_bdg_eqm(XFeO_melt, P, T, FeSiO3_bdg, XSi_melt):
    bdg.set_composition([1.-XFeSiO3_bdg, XFeSiO3_bdg])
    bdg.set_state(P, T)

    FeSiO_melt.set_composition([1. - XFeO_melt[0] - XSi_melt, 
                                XSi_melt, XFeO_melt[0]])
    FeSiO_melt.set_state(P, T)

    mu_melt = FeSiO_melt.partial_gibbs

    mu_FeSiO3 = 3.*mu_melt[2] + mu_melt[1] - 2.*mu_melt[0]
    mu_FeSiO3 = burnman.chemicalpotentials.chemical_potentials([FeSiO_melt], [bdg.endmembers[1][0].params['formula']])[0]
    return bdg.partial_gibbs[1] - mu_FeSiO3

'''
P = 25.e9
XFeSiO3_bdg = 0.2
temperatures = np.linspace(2773., 4273., 4)
X_Sis = np.linspace(0.002, 0.2, 101)
for T in temperatures:
    print T
    X_Os = np.empty_like(X_Sis)
    for i, XSi_melt in enumerate(X_Sis):
        XFeO_melt = fsolve(melt_bdg_eqm, [0.01], args=(P, T, XFeSiO3_bdg, XSi_melt))[0]

        X_Os[i] = XFeO_melt / (1. + XFeO_melt)
    plt.plot(X_Os, X_Sis, label=str(T)+'K')

plt.xlabel('X O')
plt.ylabel('X Si')
plt.legend(loc='upper right')
plt.show()
'''


P = 25.e9
XFeSiO3_bdg = 0.2
temperatures = np.linspace(2773., 4273., 4)
X_Sis = np.linspace(0.002, 0.2, 101)

for T in temperatures:
    print T
    X_Os = np.empty_like(X_Sis)
    X_Sis_wt = np.empty_like(X_Sis)
    X_Os_wt = np.empty_like(X_Sis)
    for i, XSi_melt in enumerate(X_Sis):
        XFeO_melt = fsolve(melt_bdg_eqm, [0.01], args=(P, T, XFeSiO3_bdg, XSi_melt))[0]

        X_Os[i] = XFeO_melt / (1. + XFeO_melt)

        wt_total = 55.845*(1. - X_Sis[i] - X_Os[i]) + 28.0855*X_Sis[i] + 15.9994*X_Os[i]
        X_Sis_wt[i] = 28.0855*X_Sis[i] / wt_total * 100.
        X_Os_wt[i] = 15.9994*X_Os[i] / wt_total * 100.

    plt.plot(X_Os_wt, X_Sis_wt, label=str(T)+'K')

plt.xlabel('XO (wt %)')
plt.ylabel('XSi (wt %)')
plt.xlim(0.0, 10.0)
plt.ylim(0.0, 10.0)
plt.legend(loc='upper right')
plt.show()
