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
from burnman.processchemistry import *
from burnman.chemicalpotentials import *
from burnman import constants

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scipy.optimize as optimize

atomic_masses=read_masses()


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

class fcc_iron (Mineral):
    def __init__(self):
        self.params = {
            'S_0': 35.7999958649,
            'Gprime_0': float('nan'),
            'a_0': 5.13074989862e-05,
            'K_0': 153865172537.0,
            'G_0': float('nan'),
            'Kprime_0': 5.2,
            'Kdprime_0': -3.37958221101e-11,
            'V_0': 6.93863394593e-06,
            'name': 'FCC iron',
            'H_0': 7839.99990299,
            'molar_mass': 0.055845,
            'equation_of_state': 'hp_tmt',
            'n': 1.0,
            'formula': {'Fe': 1.0},
            'Cp': [52.2754, -0.000355156, 790710.86, -619.07],
        }
        Mineral.__init__(self)


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
            'Cp': [42.63803582, 0.008971021, -260780.8155, 196.5978421], # 0.9254
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

###########################
initial_alphas=[1.0, 1.0, 1.0]
initial_enthalpy_interaction=[[11.e3, 0.0e3], [14.e3]]
###########################

# Configurational entropy
class wuestite_ss(burnman.SolidSolution):
    def __init__(self):
        # Name
        self.name='non-stoichiometric wuestite, ferric and ferrous iron treated as separate atoms in Sconf'
        endmembers = [[minerals.HP_2011_ds62.per(), '[Mg]O'],[wustite(), '[Fe]O'],[fe23o(), '[Fef2/3Vc1/3]O']]

        # Interaction parameters
        alphas=initial_alphas
        enthalpy_interaction=initial_enthalpy_interaction
        volume_interaction=[[0., 0.],[0.]]

        burnman.SolidSolution.__init__(self, endmembers, \
                          burnman.solutionmodel.AsymmetricRegularSolution(endmembers, alphas, enthalpy_interaction, volume_interaction) )

'''
Sundman (1991) models
'''

def magnetic_gibbs(T, Tc, beta, p):
    A = (518./1125.) + (11692./15975.)*((1./p) - 1.)
    tau=T/Tc
    if tau < 1: 
        f=1.-(1./A)*(79./(140.*p*tau) + (474./497.)*(1./p - 1.)*(np.power(tau, 3.)/6. + np.power(tau, 9.)/135. + np.power(tau, 15.)/600.))
    else:
        f=-(1./A)*(np.power(tau,-5)/10. + np.power(tau,-15)/315. + np.power(tau, -25)/1500.)
    return constants.gas_constant*T*np.log(beta + 1.)*f

def GHSERFe(T):
    if T < 1811:
        gibbs=1224.83 + 124.134*T - 23.5143*T*np.log(T) - 0.00439752*T*T - 5.89269e-8*T*T*T + 77358.3/T
    else:
        gibbs=-25384.451 + 299.31255*T - 46.*T*np.log(T) + 2.2960305e31*np.power(T,-9.)
    return gibbs 

def gibbs_bcc_1bar(T):
    Tc=1043.
    beta=2.22
    p=0.4
    return GHSERFe(T) + magnetic_gibbs(T, Tc, beta, p)

class bcc_iron():
    def __init__(self):
        self.params = {
            'formula': {'Fe': 1.0}}
    def set_state(self, pressure, temperature):
        self.gibbs = gibbs_bcc_1bar(temperature)

def GO2(T):
    if T < 1000.:
        return -6961.74451-76729.7484/T - 51.0057202*T - 22.2710136*T*np.log(T) - 0.0101977469*T*T + 1.32369208e-6*T*T*T 
    elif T<3300.:
        return -13137.5203 + 525809.556/T + 25.3200332*T - 33.627603*T*np.log(T) - 0.00119159274*T*T + 1.35611111e-8*T*T*T

def gibbs_fcc_1bar(T):
    Tc=201.
    beta=2.1
    p=0.28

    if T<1811.0:
        add=-1462.4 + 8.282*T - 1.15*T*np.log(T) + 6.4e-4*T*T
    else:
        add=-27098.266 + 300.25256*T - 46*T*np.log(T) + 2.78854e31*np.power(T, -9.)

    return GHSERFe(T) + magnetic_gibbs(T, Tc, beta, p) + add

def GO2(T):
    if T < 1000.:
        return -6961.74451-76729.7484/T - 51.0057202*T - 22.2710136*T*np.log(T) - 0.0101977469*T*T + 1.32369208e-6*T*T*T 
    elif T<3300.:
        return -13137.5203 + 525809.556/T + 25.3200332*T - 33.627603*T*np.log(T) - 0.00119159274*T*T + 1.35611111e-8*T*T*T

'''
Sundman model for wustite
'''
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


'''
Model for wustite for a single temperature (asymmetric solution model)
'''
def gibbs_wus_full(ys, aj, Wij, Gi, Gj):
    gibbs=[]
    for y in ys:
        Xj=3.*y
        ai=1.
        Xi=(1.-Xj)
        phii=Xi*ai/(Xi*ai + Xj*aj)
        phij=Xj*aj/(Xi*ai + Xj*aj)
        Hex=(2.*ai/(ai+aj)*phij*phij*Xi + 2.*aj/(ai+aj)*phii*phii*Xj)*Wij
        
        
    # Fe(1-y)O
        Sex=-constants.gas_constant*(Xi*np.log(Xi) + 2.*y*np.log(2.*y) + y*np.log(y))
        gibbs.append(Gi*Xi + Gj*Xj - Sex*T + Hex)
    return gibbs


######################
# Fit to fO2
######################

def gibbs_fixed_PT(Xj, ai, aj, Wij, Gi, Gj):

    # Solid solution tweaking
    burnman.SolidSolution.__init__(wus, wus.endmembers, \
                                       burnman.solutionmodel.AsymmetricRegularSolution(wus.endmembers, [1.0, ai, aj], [[0., 0.], [Wij]], [[0., 0.],[0.]], [[0., 0.],[0.0]]) )


    Xi=(1.-Xj)
    wus.set_composition([0.0, Xi, Xj])
    wus.set_state(Pr, T)
    return Gi*Xi + Gj*Xj + wus.excess_gibbs

def fO2_wus(ys, aj, Wij, Gi, Gj):
    d=0.0001
    fO2=[]
    ai=1.0
    for y in ys:
        Xj=3.*(y - d)
        gibbs0=gibbs_fixed_PT(Xj, ai, aj, Wij, Gi, Gj)
        Xj=3.*(y + d)
        gibbs1=gibbs_fixed_PT(Xj, ai, aj, Wij, Gi, Gj)
        mu_O2=2*((1.-y+d)/(1.-y-d)*gibbs1 - gibbs0)/((1.-y+d)/(1.-y-d)-1.)
        fO2.append(np.log10(np.exp((mu_O2-oxygen.gibbs)/(constants.gas_constant*T))))

    return fO2

def fO2_wus_eqm_Fe(data, Wij, Sij, H_0_FeO, H_0_Fe23O, S_0_Fe23O): # last fO2 should be zero

    wus.endmembers[1][0].params['H_0']=H_0_FeO
    wus.endmembers[1][0].params['S_0']=S_0_FeO
    wus.endmembers[2][0].params['H_0']=H_0_Fe23O
    wus.endmembers[2][0].params['S_0']=S_0_Fe23O

    # Solid solution tweaking
    burnman.SolidSolution.__init__(wus, wus.endmembers, \
                                       burnman.solutionmodel.AsymmetricRegularSolution(wus.endmembers, [1.0, 1.0, aj], [[0., 0.], [Wij]], [[0., 0.],[0.]], [[0., 0.],[Sij]]) )

    fO2=[]
    ai=1.0
    for datum in data:
        phase, T, y = datum
        wus.set_composition([0., 1-3.*y, 3.*y])
        wus.set_state(Pr, T)
        oxygen.set_state(Pr, T)
        if phase == wus:
            fO2.append(np.log10(fugacity(oxygen, [wus])))
        else:
            phase.set_state(Pr, T)
            gibbs_iron=phase.gibbs
            gibbs_wus=wus.gibbs
            mu_O2=2.*(gibbs_wus - (1.-y)*gibbs_iron) 
            fO2_eqm_with_iron=np.log10(np.exp((mu_O2-oxygen.gibbs)/(constants.gas_constant*T)))
            fO2.append(fO2_eqm_with_iron-np.log10(fugacity(oxygen, [wus])))

    return np.array(fO2)


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

fcc=fcc_iron()
bcc=bcc_iron()
wus=wuestite_ss()
mt=minerals.HP_2011_ds62.mt()
hem=minerals.HP_2011_ds62.hem()
oxygen=minerals.HP_2011_fluids.O2()

# First load fO2 data
phase_T_ys=[]
logfO2=[]
for line in open('Bransky_Hed_1968.dat'):
    content=line.strip().split()
    if content[0] != '%':
        phase_T_ys.append([wus, float(content[0]), float(content[2])/100.])
        logfO2.append(float(content[1]))

plt.plot( zip(*phase_T_ys)[2], logfO2, marker='o', linestyle='none', label='Bransky and Hed, 1968')

# Now load equilibrium with iron
# Set logfO2 as zero 
minerals_temperatures=[[fcc, 1424.62653692], [fcc, 1376.75520075], [fcc, 1323.39397471], [fcc, 1264.61208554], [fcc, 1218.11479518], [bcc, 1166.14020162], [bcc, 1120.99807704], [bcc, 1071.7589884], [bcc, 1023.91911893], [bcc, 976.072956121], [fcc, 1222.07160224], [bcc, 1172.8465432], [bcc, 1121.78241653], [bcc, 1072.56293888], [bcc, 1021.49881221], [bcc, 972.212357893], [bcc, 922.987298852], [bcc, 871.878521071], [fcc, 1223.83253043], [bcc, 1172.77398515], [bcc, 1123.52660055], [bcc, 1074.29037873], [bcc, 1021.4039286], [bcc, 972.134218449], [bcc, 922.903578019], [bcc, 871.805963016]]
compositions=[0.511190023891, 0.511202260941, 0.510985839986, 0.511384301614, 0.511472874541, 0.51148616048, 0.511421012761, 0.511433599441, 0.511829264029, 0.512148243109, 0.510939, 0.511131, 0.511467, 0.511611, 0.511948, 0.512672, 0.512865, 0.513591, 0.511467, 0.511755, 0.512141, 0.51243, 0.512768, 0.513349, 0.513591, 0.514222]
compositions=np.array([1.0-x_to_y(compositions[i]) for i in range(len(compositions))])

for i, phase_T in enumerate(minerals_temperatures):
    phase_T_ys.append([phase_T[0], phase_T[1], compositions[i]])
    logfO2.append(0.0)
        
aj=10.
S_0_FeO=58.2
# Variables should be
# Wh, Ws, H_0_FeO, S_0_FeO, H_0_Fe23O, S_0_Fe23O,  (and Cp_0_FeO?)
guesses=[14.e3, 0.0, -429440., -450000., 60.]
popt, pcov = optimize.curve_fit(fO2_wus_eqm_Fe, phase_T_ys, logfO2, guesses)
Wh, Ws, H_0_FeO, H_0_Fe23O, S_0_Fe23O = popt

print popt
print fO2_wus_eqm_Fe(phase_T_ys, Wh, Ws, H_0_FeO, H_0_Fe23O, S_0_Fe23O) - logfO2

def fO2_wus_model(ys, T):
    fO2=[]

    for y in ys:
        wus.set_composition([0.0, 1-3.*y, 3.*y])
        wus.set_state(Pr, T)
        fO2.append(np.log10(fugacity(oxygen, [wus])))
    return fO2

Ts=np.linspace(1273.15, 1573.15, 7)
ys=np.linspace(0.04, 0.16, 21)
for T in Ts:
    oxygen.set_state(Pr, T)
    wus.set_state(Pr, T)
    Gi=wus.endmembers[1][0].gibbs
    Gj=wus.endmembers[2][0].gibbs
    Wij=Wh - T*Ws
    print Wij

    plt.plot( ys, fO2_wus_model(ys, T), linewidth=1)


    # plt.plot( ys, G_wustite(ys, T)[1]-gibbs_wus_full(ys, aj, Wij, Gi, Gj), 'r-', linewidth=1, label='Gibbs difference')

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

print S_tr

#wus.endmembers[0][0].params['S_0']=S_tr[0]
#wus.endmembers[1][0].params['S_0']=S_tr[0]+S_tr[1]
#wus.endmembers[1][0].params['S_0']=wus.endmembers[1][0].params['S_0']+2.

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
Plot the finished phase diagram!
'''



'''
# SPIEDEL 1967 (MgO in wustite)
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

def fitfO2(mineral):
    def fit(data, delH_MgO_Fe2O3):

        # Endmember tweaking
        #mineral.endmembers[2][0].params['H_0'] = H0_Fe2O3

        # Solid solution tweaking
        alphas=initial_alphas
        enthalpy_interaction=initial_enthalpy_interaction

        enthalpy_interaction[0][1]=delH_MgO_Fe2O3

        volume_interaction=[[0., 0.], [0.]]  
        burnman.SolidSolution.__init__(mineral, mineral.endmembers, \
                                           burnman.solutionmodel.AsymmetricRegularSolution(mineral.endmembers, alphas, enthalpy_interaction, volume_interaction) )
        
        oxygen_fugacities=[]
        for composition in data:

            mineral.set_composition(composition)
            mineral.set_state(P, T)

            oxygen_fugacities.append(np.log10(fugacity(oxygen, [wus])))

        return oxygen_fugacities
    return fit

guesses=[16532.]
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
