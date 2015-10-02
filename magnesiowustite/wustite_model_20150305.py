# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))


# Benchmarks for the solid solution class
import burnman
from burnman import minerals
from burnman import tools
from burnman.mineral import Mineral
from burnman.minerals import Myhill_calibration_iron
from burnman.processchemistry import *
from burnman.chemicalpotentials import *
from burnman import constants

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scipy.optimize as optimize
from scipy import interpolate

atomic_masses=read_masses()


### MANUAL VARIABLES ###
S_0_FeO=58.0


'''
Wustite composition conversion functions
'''
def y_to_x(y):
    return 1/(y+1)

def x_to_y(x):
    return (1./x) - 1.

def x_to_f(x):
    return 6. - (3./x)

def f_to_x(f):
    return 3./(6.-f)

def f_to_y(f):
    return x_to_y(f_to_x(f))

'''
Wustite model
'''
class fper (Mineral): # Holland and Powell, ds62
    def __init__(self):
        formula='Fe1.0O1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'fper',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -259870.0 ,
            'S_0': 58.6 ,
            'V_0': 1.206e-05 ,
            'Cp': [44.4, 0.00828, -1214200.0, 185.2] , # HP value
            'a_0': 3.22e-05 ,
            'K_0': 1.52e+11 ,
            'Kprime_0': 4.9 ,
            'Kdprime_0': -3.2e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

class wustite (Mineral): # Holland and Powell, ds62
    def __init__(self):
        formula='Fe1.0O1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'fper',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -259870.0 ,
            'S_0': 58.6 ,
            'V_0': 1.206e-05 ,
            #'Cp': [44.4, 0.00828, -1214200.0, 185.2] , # HP value
            # 'Cp': [53.334316, 0.00779203541, -325553.876, -75.023374], # old value
            #'Cp': [50.92936041, 0.0056561, -225829.9289, 38.79636971], # 0.9379
            'Cp': [42.63803582, 0.008971021, -260780.8155, 196.5978421], # By linear extrapolation from Fe0.9254O and hematite/3..
            'a_0': 3.22e-05 ,
            'K_0': 1.52e+11 ,
            'Kprime_0': 4.9 ,
            'Kdprime_0': -3.2e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)


class defect_wustite (Mineral):
    def __init__(self):
        self.params = {
            'S_0': 2.89560534e+01,
            'Gprime_0': float('nan'),
            'a_0': 3.22e-05,
            'K_0': 1.52e+11,
            'G_0': float('nan'),
            'Kprime_0': 4.9,
            'Kdprime_0': -3.2e-11,
            'V_0': 1.10419547534e-05,
            'name': 'fper',
            'H_0': -2.69050221e+05, #+500.,
            'molar_mass': 0.0532294,
            'equation_of_state': 'hp_tmt',
            'n': 2.0,
            'formula': {'Fe': 0.6666666666666666, 'O': 1.0},
            'Cp': [-3.64959181, 0.0129193873, -1079881.27, 1112.41795],
        }
        Mineral.__init__(self)

class fe23o (Mineral): # starting guess is hem/3.
    def __init__(self):
        formula='Fe2/3O'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'hem',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -825610.0/3. ,
            'S_0': 87.4/3.,
            'V_0': 1.10701e-5 ,
            'Cp': [163.9/3., 0.0, -2257200.0/3., -657.6/3.] ,
            'a_0': 2.79e-05 ,
            'K_0': 1.52e+11 ,
            'Kprime_0': 4.9 ,
            'Kdprime_0': -3.2e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)
'''
class fe23o (Mineral): # starting guess is hem/3.
    def __init__(self):
        formula='Fe2/3O'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'hem',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -825610.0/3. ,
            'S_0': 87.4/3. ,
            'V_0': 3.027e-05/3. ,
            'Cp': [163.9/3., 0.0, -2257200.0/3., -657.6/3.] ,
            'a_0': 2.79e-05 ,
            'K_0': 2.23e+11 ,
            'Kprime_0': 4.04 ,
            'Kdprime_0': -1.8e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses),
            'landau_Tc': 955.0 ,
            'landau_Smax': 15.6/3. ,
            'landau_Vmax': 0.0 }
        Mineral.__init__(self)
'''
class fe34o (Mineral): # starting guess is mt/4.
    def __init__(self):
        formula='Fe0.75O'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'mt-composition defect wustite',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -1114500.0/4. ,
            'S_0': 146.9/4. ,
            'V_0': 4.452e-05/4. ,
            'Cp': [262.5/4., -0.007205/4., -1926200.0/4., -1655.7/4.] ,
            'a_0': 3.71e-05 ,
            'K_0': 1.857e+11 ,
            'Kprime_0': 4.05 ,
            'Kdprime_0': -2.2e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses),
            'landau_Tc': 848.0 ,
            'landau_Smax': 35.0/4. ,
            'landau_Vmax': 0.0 }
        Mineral.__init__(self)

# Configurational entropy
class wuestite_ss(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        # Name
        self.name='non-stoichiometric wuestite, ferric and ferrous iron treated as separate atoms in Sconf'
        self.endmembers = [[minerals.HP_2011_ds62.per(), '[Mg]O'],[wustite(), '[Fe]O'],[fe23o(), '[Fef2/3Vc1/3]O']]
        self.type='subregular'
        # Interaction parameters
        self.enthalpy_interaction=[[[0., 0.],[0.,0.]],[[-39613., -4236.]]]
        self.volume_interaction=[[[0., 0.],[0.,0.]],[[0., 0.]]]
        self.entropy_interaction=[[[0., 0.],[0.,0.]],[[-4.048, -4.048]]]

        burnman.SolidSolution.__init__(self, molar_fractions)

wus=wuestite_ss()
print wus.solution_model.Wh
print wus.solution_model.Ws

        
'''
Sundman (1991) oxygen and wustite models
'''

def GO2(T):
    if T < 1000.:
        return -6961.74451-76729.7484/T - 51.0057202*T - 22.2710136*T*np.log(T) - 0.0101977469*T*T + 1.32369208e-6*T*T*T 
    elif T<3300.:
        return -13137.5203 + 525809.556/T + 25.3200332*T - 33.627603*T*np.log(T) - 0.00119159274*T*T + 1.35611111e-8*T*T*T

def G_wustite(y, T):
    y2=1.-(3.*y)
    y3=2.*y
    yv=1.*y

    L0_23=-12324.4 # Note missing decimal point in Sundman 1991
    L1_23=20070.0 # Note sign error and missing decimal point in Sundman 1991
    
    Gwustite=-279318. + 252.848*T - 46.12826*T*np.log(T) - 0.0057402984*T*T
    Awustite=-55384. + 27.888*T
    
    HSERO=0.
    HSERFe=0.
    G_2O=Gwustite #+ HSERFe + HSERO
    G_3O=1.25*(Gwustite + Awustite) #+ HSERFe + HSERO
    G_VO=0.#+ HSERO
    DeltamixGex=y2*y3*(L0_23 + (y2-y3)*L1_23)
    DeltamixG=y2*G_2O + y3*G_3O + yv*G_VO + constants.gas_constant*T*(y2*np.log(y2) + y3*np.log(y3) + yv*np.log(yv)) + DeltamixGex 
    return DeltamixGex, DeltamixG

######################
# Fit to fO2
######################


'''
Volume parameters
'''
Z=4.
nA=6.02214e23
voltoa=1.e30

'''
Pressure parameters
'''
Pr=1.e5 # 1 atm.
P=1.e5 # 1 atm.

fcc=Myhill_calibration_iron.fcc_iron()
bcc=Myhill_calibration_iron.bcc_iron()
wus=wuestite_ss()
mt=minerals.HP_2011_ds62.mt()
iron=minerals.HP_2011_ds62.iron()
hem=minerals.HP_2011_ds62.hem()
oxygen=minerals.HP_2011_fluids.O2()

temperatures=np.linspace(300.,1700.,101)
gibbs_diff=np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    fcc.set_state(Pr,T)
    bcc.set_state(Pr,T)
    gibbs_diff[i]=fcc.gibbs-bcc.gibbs

plt.plot(temperatures, gibbs_diff, 'b-', linewidth=1., label='fcc-bcc')
plt.legend(loc='lower left')
plt.ylabel("Difference in gibbs free energy (J/mol)")
plt.xlabel("Temperature (K)")
plt.show()

'''
Fe+Fe3O4 (ONeill)
'''

mt.params['H_0'] = -1115.40e3 # O'Neill et al., 1988


def mu_O2_fcc_mt(T):
    return -607673 + 1060.994*T - 132.3909*T*np.log(T) + 0.06657*T*T # 750 < T < 833


T_mt_Sundman=[298.15, 350., 400., 450., 500., 550., 600., 650., 700., 750., 800., 848., 850., 900., 950., 1000., 1050., 1100., 1150., 1200., 1250., 1300., 1350., 1400., 1450., 1500., 1550., 1600., 1650., 1700., 1750., 1800., 1850., 1900.]
G_mt_Sundman=[-1159210., -1167570., -1176680., -1186870., -1198070., -1210220., -1223270., -1237200., -1251960., -1267550., -1283960., -1300510., -1301210., -1319210., -1337820., -1356990., -1376690., -1396870., -1417520., -1438620., -1460140., -1482070., -1504390., -1527090., -1550150., -1573570., -1597320., -1621410., -1645820., -1670540., -1695570., -1720890., -1746500., -1772400.]

def gibbs_mt_Sundman(T):
    s = interpolate.InterpolatedUnivariateSpline(T_mt_Sundman, G_mt_Sundman)
    return s(T) - 2.*T
    
def gibbs_mt_ONeill(T):
    oxygen.set_state(Pr, T) 
    if T<1184.733:
        bcc.set_state(Pr, T) 
        G0=3.*bcc.gibbs + 2.*oxygen.gibbs
    else:
        fcc.set_state(Pr, T) 
        G0=3.*fcc.gibbs + 2.*oxygen.gibbs
    if T<600.:
        raise Exception('T too low')
        return 0.
    elif T<848:
        return -1215345. + 2121.987*T - 264.7818*T*np.log(T) + 0.133140*T*T +G0
    elif T<1042:
        return -1354650. + 4245.621*T - 577.5478*T*np.log(T) + 0.309515*T*T +G0
    elif T<1184:
        return -1079617. + 220.694*T + 10.0979*T*np.log(T) +G0 # corrected sign error
    elif T<2000: # should be 1600
        return -1116959. + 465.536*T - 20.0438*T*np.log(T) +G0
    else:
        raise Exception('T too high')
        return 0.

'''
Functions to find the chemical potentials of Fe and Fe3O4 for different compositions of wuestite at each pressure. 
'''

# Function to return the equilibrium composition of wustite with another mineral at P and T
wus.endmembers[2][0].params['formula'].pop("Vc", None)

def wus_eqm(mineral):
    def mineral_equilibrium(c, P, T, XMg):
        wus.set_composition([XMg, 1.0-XMg-c[0],c[0]])
        wus.set_state(P, T)
        mineral.set_state(P, T)
        mu_mineral=chemical_potentials([wus], [mineral.params['formula']])[0]   
        return  mu_mineral-mineral.gibbs
    return mineral_equilibrium

def wus_2mineral_eqm(mineral1, mineral2):
    def mineral_equilibrium(arg, P):
        c=arg[0]
        T=arg[1]
        wus.set_composition([0.0, 1.0-c,c])
        wus.set_state(P, T)
        mineral1.set_state(P, T)
        mu_mineral1=chemical_potentials([wus], [mineral1.params['formula']])[0]  
        mineral2.set_state(P, T)
        mu_mineral2=chemical_potentials([wus], [mineral2.params['formula']])[0]   
        return  [mu_mineral1-mineral1.gibbs, mu_mineral2-mineral2.gibbs]
    return mineral_equilibrium

def fO2_wus_eqm_Fe(data, Wij, Wji, Ws, H_0_FeO, H_0_Fe34O, S_0_Fe34O): # last fO2 should be zero

    wus.endmembers[1][0].params['H_0']=H_0_FeO
    wus.endmembers[1][0].params['S_0']=S_0_FeO
    wus.endmembers[2][0].params['H_0']=H_0_Fe34O
    wus.endmembers[2][0].params['S_0']=S_0_Fe34O


    # Interaction parameters
    wus.enthalpy_interaction=[[[0., 0.],[0.,0.]],[[Wij, Wji]]]
    wus.volume_interaction=[[[0., 0.],[0.,0.]],[[0., 0.]]]
    wus.entropy_interaction=[[[0., 0.],[0.,0.]],[[Ws, Ws]]]
    
    burnman.SolidSolution.__init__(wus)

    fO2=[]
    for datum in data:
        if datum[0]=='ONeill': # ONeill data (Fe eqm), x is just phase in eqm and temperature, y is muO2
            code, phase, T = datum
            f=optimize.fsolve(wus_eqm(phase), 0.16, args=(Pr, T, 0.))[0]
            oxygen.set_state(Pr, T)
            fO2.append(np.log10(fugacity(oxygen, [wus])))
        else:
            phase, T, y = datum
            wus.set_composition([0., 1-3.*y, 3.*y])
            wus.set_state(Pr, T)
            oxygen.set_state(Pr, T) 
            fO2_wus=np.log10(fugacity(oxygen, [wus]))
            
            phase.set_state(Pr, T)
            if phase == wus:
                fO2.append(fO2_wus)
            elif phase == mt:
                mu_O2=2./(4.-(3./(1.-y)))*(mt.gibbs - 3./(1.-y)*wus.gibbs) 
                fO2_eqm_with_mt=np.log10(np.exp((mu_O2-oxygen.gibbs)/(constants.gas_constant*T)))
                fO2.append(fO2_eqm_with_mt-fO2_wus)
            elif phase==bcc or phase==fcc:
                gibbs_iron=phase.gibbs
                mu_O2=2.*(wus.gibbs - (1.-y)*gibbs_iron) 
                fO2_eqm_with_iron=np.log10(np.exp((mu_O2-oxygen.gibbs)/(constants.gas_constant*T)))
                fO2.append(fO2_eqm_with_iron-fO2_wus)
        
            
    return np.array(fO2)

'''
temperatures=np.linspace(850., 1590., 101)
for T in temperatures:
    mt.set_state(Pr, T)
    fcc.set_state(Pr, T)
    oxygen.set_state(Pr, T)
    print T, mt.gibbs - gibbs_mt_ONeill(T)
'''

# First load fO2 data
phase_T_ys=[]
logfO2=[]
sigma_fO2=[]

phase_T_ys_GG=[]
logfO2_GG=[]
for line in open('Giddings_Gordon_1973.dat'):
    content=line.strip().split()
    if content[0] != '%':
        phase_T_ys_GG.append([wus, float(content[0]), float(content[2])/100.])
        logfO2_GG.append(float(content[1]))
        phase_T_ys.append([wus, float(content[0]), float(content[2])/100.])
        logfO2.append(float(content[1]))
        sigma_fO2.append(0.1)

plt.plot( zip(*phase_T_ys_GG)[2], logfO2_GG, marker='o', linestyle='none', label='Giddings and Gordon, 1973')

phase_T_ys_BH=[]
logfO2_BH=[]
for line in open('Bransky_Hed_1968.dat'):
    content=line.strip().split()
    if content[0] != '%':
        phase_T_ys_BH.append([wus, float(content[0]), float(content[2])/100.])
        logfO2_BH.append(float(content[1]))
        phase_T_ys.append([wus, float(content[0]), float(content[2])/100.])
        logfO2.append(float(content[1]))
        sigma_fO2.append(0.005)

plt.plot( zip(*phase_T_ys_BH)[2], logfO2_BH, marker='o', linestyle='none', label='Bransky and Hed, 1968')

# Now load equilibrium with iron
# Set logfO2 as zero 
minerals_temperatures=[[fcc, 1424.62653692], [fcc, 1376.75520075], [fcc, 1323.39397471], [fcc, 1264.61208554], [fcc, 1218.11479518], [bcc, 1166.14020162], [bcc, 1120.99807704], [bcc, 1071.7589884], [bcc, 1023.91911893], [bcc, 976.072956121], [fcc, 1222.07160224], [bcc, 1172.8465432], [bcc, 1121.78241653], [bcc, 1072.56293888], [bcc, 1021.49881221], [bcc, 972.212357893], [bcc, 922.987298852], [bcc, 871.878521071], [fcc, 1223.83253043], [bcc, 1172.77398515], [bcc, 1123.52660055], [bcc, 1074.29037873], [bcc, 1021.4039286], [bcc, 972.134218449], [bcc, 922.903578019], [bcc, 871.805963016], [mt, 1517.188975], [mt, 1565.2554047], [mt, 1679.06205932]]
compositions=[0.511190023891, 0.511202260941, 0.510985839986, 0.511384301614, 0.511472874541, 0.51148616048, 0.511421012761, 0.511433599441, 0.511829264029, 0.512148243109, 0.510939, 0.511131, 0.511467, 0.511611, 0.511948, 0.512672, 0.512865, 0.513591, 0.511467, 0.511755, 0.512141, 0.51243, 0.512768, 0.513349, 0.513591, 0.514222, 0.539080473166, 0.54144548686, 0.544867315425]
compositions=np.array([1.0-x_to_y(compositions[i]) for i in range(len(compositions))])

for i, phase_T in enumerate(minerals_temperatures):
    phase_T_ys.append([phase_T[0], phase_T[1], compositions[i]])
    logfO2.append(0.0)
    sigma_fO2.append(0.001) # compositional uncertainty (y)

# Add O'Neill data 
def mu_O2_Fe_FeO_ONeill(T): # ONeill and Pownceby, 1993
    Fc=96.48456
    Pair=0.97 # bar
    EMF0=1.1 # EMF correction from ONeill et al. (2003)
    mod=-4.*Fc*(EMF0)
    if T<833.:
        raise Exception('T too low')
        return 0.
    elif T<1042.:
        return -605568. + 1366.420*T - 182.7955*T*np.log(T) + 0.10359*T*T + mod
    elif T<1184.:
        return -519113. + 59.129*T + 8.9276*T*np.log(T) + mod
    elif T<1644.:
        return -550915 + 269.106*T - 16.9484*T*np.log(T) + mod
    else:
        raise Exception('T too high')
        return 0.

def mu_O2_FeO_Fe3O4_ONeill(T):
    Fc=96.48456
    Pair=0.97 # bar
    EMF0=1.1 # EMF correction from ONeill et al. (2003)
    mod=-4.*Fc*(EMF0)
    if T<833.:
        raise Exception('T too low')
        return 0.
    elif T<1274.:
        return -581927. - 65.618*T + 38.7410*T*np.log(T) + mod
    else:
        raise Exception('T too high')
        return 0.

def mu_O2_Fe_Fe3O4_ONeill(T):
    Fc=96.48456
    Pair=0.97 # bar
    EMF0=1.1 # EMF correction from ONeill et al. (2003)
    mod=-4.*Fc*(EMF0)
    if T<750.:
        raise Exception('T too low')
        return 0.
    elif T<833.:
        return -607673 + 1060.994*T - 132.3909*T*np.log(T) + 0.06657*T*T +mod
    else:
        raise Exception('T too high')
        return 0.

# BCC
Ts=np.linspace(873.15, 1173.15, 7)
for T in Ts:
    phase_T_ys.append(['ONeill', bcc, T])
    logfO2.append(mu_O2_Fe_FeO_ONeill(T)/(constants.gas_constant*T*np.log(10.)))
    sigma_fO2.append(150./(constants.gas_constant*T*np.log(10.)))
# FCC
Ts=np.linspace(1223.15, 1623.15, 9)
for T in Ts:
    phase_T_ys.append(['ONeill', fcc, T])
    logfO2.append(mu_O2_Fe_FeO_ONeill(T)/(constants.gas_constant*T*np.log(10.)))
    sigma_fO2.append(150./(constants.gas_constant*T*np.log(10.)))
    print 150./(constants.gas_constant*T*np.log(10.))

'''
# MT
Ts=np.linspace(873.15, 1273.15, 9)
for T in Ts:
    phase_T_ys.append(['ONeill', mt, T])
    logfO2.append(mu_O2_FeO_Fe3O4_ONeill(T)/(constants.gas_constant*T*np.log(10.)))
    sigma_fO2.append(306./(constants.gas_constant*T*np.log(10.)))
'''

Ts=np.linspace(751., 832, 9)
for T in Ts:
    mt.set_state(Pr, T)
    bcc.set_state(Pr, T)
    oxygen.set_state(Pr, T)
    print chemical_potentials([mt, bcc], [{'O': 2.0}]) - oxygen.gibbs - mu_O2_Fe_Fe3O4_ONeill(T)


# Variables should be
# Wh, Ws, H_0_FeO, S_0_FeO, H_0_Fe23O, S_0_Fe23O,  (and Cp_0_FeO?)
guesses=[14.e3, 14.e3, -4.0, -429440., -450000., 60.]
popt, pcov = optimize.curve_fit(fO2_wus_eqm_Fe, phase_T_ys, logfO2, guesses)#, sigma_fO2)
Wij, Wji, Ws, H_0_FeO, H_0_Fe34O, S_0_Fe34O = popt

#print popt
#print fO2_wus_eqm_Fe(phase_T_ys, Wij, Wji, Ws, H_0_FeO, H_0_Fe34O, S_0_Fe34O) - logfO2

def fO2_wus_model(ys, T):
    fO2=[]
    for y in ys:
        wus.set_composition([0., 1-3.*y, 3.*y])
        wus.set_state(Pr, T)
        oxygen.set_state(Pr, T)
        fO2.append(np.log10(fugacity(oxygen, [wus])))
    return fO2


Ts=np.linspace(1073.15, 1573.15, 11)
ys=np.linspace(0.04, 0.16, 21)
y_Fe=np.empty_like(Ts)
y_Fe3O4=np.empty_like(Ts)
fO2_Fe=np.empty_like(Ts)
fO2_Fe3O4=np.empty_like(Ts)
for i, T in enumerate(Ts):
    np.savetxt(str(T)+'wus_fO2.dat', zip(*[ys, fO2_wus_model(ys, T)]))
    plt.plot( ys, fO2_wus_model(ys, T), linewidth=1)

    y_Fe[i] = 1.0-f_to_y(optimize.fsolve(wus_eqm(bcc), 0.16, args=(Pr, T, 0))[0])
    fO2_Fe[i]=mu_O2_Fe_FeO_ONeill(T)/(constants.gas_constant*T*np.log(10.))
    if T <1274:
        y_Fe3O4[i] = 1.0-f_to_y(optimize.fsolve(wus_eqm(mt), 0.16, args=(Pr, T, 0))[0])
        fO2_Fe3O4[i]=mu_O2_FeO_Fe3O4_ONeill(T)/(constants.gas_constant*T*np.log(10.))
    else:
        y_Fe3O4[i] = float('nan')
        fO2_Fe3O4[i] = float('nan')


np.savetxt('wus_iron_fO2.dat', zip(*[y_Fe, fO2_Fe]))
np.savetxt('wus_mt_fO2.dat', zip(*[y_Fe3O4, fO2_Fe3O4]))

plt.plot(y_Fe, fO2_Fe, marker='o', c='black', linewidth=1)
plt.plot(y_Fe3O4, fO2_Fe3O4, marker='o', c='black', linewidth=1)

plt.legend(loc="upper left")
plt.ylabel("log10(fO2)")
plt.xlabel("y in Fe(1-y)O")
plt.show()
  

''' 
Entropies
''' 
# Data from Gronvold et al. 1993
y_entropy=np.array([[1.-0.947, 57.4365],[1.-0.9379, 57.0905],[1.-0.9254, 56.2217]])
T=298.15
f=np.array([0., 0., 0.])
entropy_mechanical_mix=np.empty_like(f)
entropy_measured=np.empty_like(f)
for i, (y, s) in enumerate(y_entropy):
    f[i]=3.*y
    wus.set_composition([0.0, 1.-f[i], f[i]])
    wus.set_state(Pr, T)
    entropy_mechanical_mix[i] = s - wus.excess_entropy
    entropy_measured[i] = s

linf=np.linspace(0., 1., 101)
S_linf=np.empty_like(linf)
for i in range(len(linf)):
    wus.set_composition([0.0, 1.-linf[i], linf[i]])
    wus.set_state(Pr, T)  
    S_linf[i]=wus.S

def linear_trend(x, a, b):
    return a + b*x

S_tr, S_tr_var = optimize.curve_fit(linear_trend, f, entropy_mechanical_mix)

plt.plot( f, entropy_measured, 'o', linewidth=3., label='Measured entropy')
plt.plot( f, entropy_mechanical_mix, 'o', linewidth=3., label='Measured-conf-interaction entropy')
plt.plot( np.linspace(0.,1.,101.), linear_trend(np.linspace(0.,1.,101.), S_tr[0], S_tr[1]), 'r--', linewidth=3., label='Best fit linear trend')
plt.plot( np.linspace(0.,1.,101.), linear_trend(np.linspace(0.,1.,101.), wus.endmembers[1][0].params['S_0'], wus.endmembers[2][0].params['S_0']-wus.endmembers[1][0].params['S_0']), 'g-', linewidth=1., label='Modelled-conf-interaction entropy')
plt.plot(linf, S_linf, 'b-', linewidth=1., label='Modelled entropy')
plt.xlim(0.0,1.0)
plt.ylim(0.,70.0)
plt.legend(loc='lower left')
plt.ylabel("Entropy of solution (J/K/mol)")
plt.xlabel("Fraction ferric iron")
plt.show()

'''
Plot the heat capacities of various wustites
a + b*x + c/(x*x) + d/np.sqrt(x) 
'''

def Cp_Wus1993(T,a,b,c,d):
    return a + b*T + c/(T*T) + d*T*T*T

def Cp_phase(phase, Tarr):
    Cparr=np.empty_like(Tarr)
    for i, T in enumerate(Tarr):
        phase.set_state(Pr, T)
        Cparr[i]=phase.C_p
    return Cparr

ferropericlase=fper()
Tarr=np.linspace(300.,1300.,101)
plt.plot( Tarr, Cp_phase(ferropericlase,Tarr), '--', linewidth=1., label='Cp fper (HP)')

wus.set_composition([0., 1., 0.])
plt.plot( Tarr, Cp_phase(wus,Tarr), linewidth=1., label='Cp FeO')
wus.set_composition([0., 1.-0.075*3., 0.075*3.])
plt.plot( Tarr, Cp_phase(wus,Tarr), linewidth=1., label='Cp Fe0.925O')
wus.set_composition([0., 0., 1.])
plt.plot( Tarr, Cp_phase(wus,Tarr), linewidth=1., label='Cp Fe0.66O')
plt.plot( Tarr, Cp_Wus1993(Tarr,51.129,4.85282e-3,-334.9089e3,-40.433488e-12), '--', linewidth=1., label='Cp Fe0.9379O, obs')
plt.plot( Tarr, Cp_Wus1993(Tarr,50.804,4.16562e-3,-248.1707e3,451.70487e-12), '--', linewidth=1., label='Cp Fe0.9254O, obs')


plt.title("Comparison of Gibbs free energies of wuestite")
plt.ylabel("Cp wuestite (J/K/mol)")
plt.xlabel("Temperature (K)")
plt.legend(loc='lower right')
plt.show()


'''
Plot the completed phase diagram!
'''

P=1.e5
XMg=0.0

comp_eqm, T_eqm=optimize.fsolve(wus_2mineral_eqm(bcc,mt), [0.16, 800.], args=(P))

temperatures=np.linspace(T_eqm-50.,1700.,101)
bcc_wus_comp=np.empty_like(temperatures)
fcc_wus_comp=np.empty_like(temperatures)
mt_wus_comp=np.empty_like(temperatures)

for idx, T in enumerate(temperatures):
    bcc_wus_comp[idx]=1.0-f_to_y(optimize.fsolve(wus_eqm(bcc), 0.16, args=(P, T, XMg))[0])
    fcc_wus_comp[idx]=1.0-f_to_y(optimize.fsolve(wus_eqm(fcc), 0.16, args=(P, T, XMg))[0])
    mt_wus_comp[idx]=1.0-f_to_y(optimize.fsolve(wus_eqm(mt), 0.16, args=(P, T, XMg))[0])


plt.plot( [0., 0.25], [T_eqm, T_eqm], 'r-', linewidth=3., label='iron-wus-mt')
plt.plot( bcc_wus_comp, temperatures, 'r-', linewidth=3., label='bcc-wus')
plt.plot( fcc_wus_comp, temperatures, 'r--', linewidth=3., label='fcc-wus')
plt.plot( mt_wus_comp, temperatures, 'r-', linewidth=3., label='wus-mt')

np.savetxt('bcc_wus.dat', zip(*[bcc_wus_comp, temperatures]))
np.savetxt('fcc_wus.dat', zip(*[fcc_wus_comp, temperatures]))
np.savetxt('mt_wus.dat', zip(*[mt_wus_comp, temperatures]))
np.savetxt('T_eqm_iron_mt_wus.dat', zip(*[[0., 0.25], [T_eqm, T_eqm]]))

for filename in ['Lykasov_Fe-FeO.dat', 'Vallet_FeOW1-Fe3O4.dat', 'Darken_Fe-FeO.dat', 'Darken_FeOW1-Fe3O4.dat', 'Asao_et_al_1970.dat', 'Barbi_1964.dat', 'Barbera_et_al_1980.dat']:
    f=open(filename, 'r')
    datastream = f.read()
    f.close()
    datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
    phase_boundaries=np.array(datalines, np.float32).T
    phase_boundaries[0]=[1.0-x_to_y(phase_boundaries[0][i]) for i in range(len(phase_boundaries[0]))]
    plt.plot( phase_boundaries[0], phase_boundaries[1], 'o', linestyle='none', label=filename)
    #print filename, phase_boundaries[0], phase_boundaries[1]

plt.xlim(0.,0.30)
plt.ylim(700.,1700.)
plt.title('Equilibrium at %5.0f K, Fe%6.3f O'%(T_eqm, 1.-comp_eqm))
plt.ylabel("Temperature (K)")
plt.xlabel("(1-y) in FeyO")
plt.legend(loc='lower right')
plt.show()

'''
Plot excess enthalpy
'''

T=1600.
ys=np.linspace(0.00, 0.30)
H_ex=np.empty_like(ys)
for i, y in enumerate(ys):
    wus.set_composition([0., 1-3.*y, 3.*y])
    wus.set_state(P, T)
    H_ex[i]=wus.excess_enthalpy

plt.plot( ys, H_ex, linewidth=1, label='H_ex')
plt.xlabel("y in Fe(1-y)O")
plt.legend(loc='lower right')
plt.show()



Pr=1.e5
T=1473.15
XMgs=np.linspace(0.0, 0.8, 11)
bcc_fper_comp=np.empty_like(XMgs)
for idx, XMg in enumerate(XMgs):
    bcc_fper_comp[idx]=1.0-f_to_y(optimize.fsolve(wus_eqm(bcc), 0.16, args=(Pr, T, XMg))[0])

print wus.endmembers[1][0].params
print wus.endmembers[2][0].params
print wus.solution_model.Wh
print wus.solution_model.Ws


wus.set_composition([0.0, 0.5, 0.5])
wus.set_state(1.e5, 1200.)
bcc.set_state(1.e5, 1200.)
mt.set_state(1.e5, 1200.)
print wus.gibbs, bcc.gibbs, mt.gibbs

'''
SPIEDEL 1967 (MgO in wustite)
'''
'''
T=1300.+273.15 # Spiedel 1967

wt_MgO=40.3044
wt_FeO=71.844
wt_Fe23O=159.69/3.
def FFwt_to_mol(FeOwt, Fe23Owt):
    MgOwt=100.-FeOwt-Fe23Owt
    total_moles=MgOwt/wt_MgO + FeOwt/wt_FeO + Fe23Owt/wt_Fe23O
    MgO_mol=(MgOwt/wt_MgO)/total_moles
    FeO_mol=(FeOwt/wt_FeO)/total_moles
    Fe23O_mol=(Fe23Owt/wt_Fe23O)/total_moles
    return [MgO_mol, FeO_mol, Fe23O_mol]

fO2_data=[]
oxygen_fugacities=[]
compositions=[]
i=0
for line in open('X-fO2_Spiedel_1967.dat'):
    content=line.strip().split()
    i=i+1
    if content[0] != '%' and i < 200:
        fO2_data.append([float(content[0]), FFwt_to_mol(float(content[1]), float(content[2]))])
        oxygen_fugacities.append(float(content[0]))
        compositions.append(FFwt_to_mol(float(content[1]), float(content[2])))

oxygen_fugacities=np.array(oxygen_fugacities)
compositions=np.array(compositions)

wus.endmembers[2][0].params['formula'].pop("Vc", None)
oxygen.set_state(Pr, T)

print wus.solution_model.Wh


def fitfO2(mineral):
    def fit(data, Wh01, Wh10, Wh02, Wh20):

        # Solid solution tweaking
        enthalpy_interaction=[[[Wh01, Wh10], [Wh02, Wh20]],[[wus.solution_model.Wh[1][2], wus.solution_model.Wh[2][1]]]]
        volume_interaction=[[[0., 0.],[0.,0.]],[[0., 0.]]]
        entropy_interaction=[[[0., 0.],[0.,0.]],[[wus.solution_model.Ws[1][2], wus.solution_model.Ws[2][1]]]]
        
        burnman.SolidSolution.__init__(wus, wus.endmembers, \
                                           burnman.solutionmodel.SubregularSolution(wus.endmembers, enthalpy_interaction, volume_interaction, entropy_interaction) )
        
        oxygen_fugacities=[]
        for composition in data:

            mineral.set_composition(composition)
            mineral.set_state(P, T)

            oxygen_fugacities.append(np.log10(fugacity(oxygen, [wus])))

        return oxygen_fugacities
    return fit

guesses=[11000., 11000, 11000, 11000]
popt, pcov = optimize.curve_fit(fitfO2(wus), compositions, oxygen_fugacities, guesses)

print popt

print 'MgO, FeO, diff logfO2'
mgnumbers=[]
ferriccomponents=[]
difffo2=[]
for datum in fO2_data:
    wus.set_composition(datum[1])
    wus.set_state(P, T)
    mgnumbers.append(datum[1][0])
    ferriccomponents.append(datum[1][2])
    difffo2.append(datum[0] - np.log10(fugacity(oxygen, [wus])))
    print datum[1][0], datum[1][2], datum[0] - np.log10(fugacity(oxygen, [wus]))


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mgnumbers, ferriccomponents, difffo2, c='r', marker='o')
ax.set_xlabel('x MgO')
ax.set_ylabel('x Fe2/3O')
ax.set_zlabel('diff logfO2')

plt.show()
'''
