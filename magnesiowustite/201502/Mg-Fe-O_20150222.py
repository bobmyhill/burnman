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

class wustite (Mineral):
    def __init__(self):
        self.params = {
            'S_0': 6.50917432e+01,
            'Gprime_0': float('nan'),
            'a_0': 3.22e-05,
            'K_0': 1.52e+11,
            'G_0': float('nan'),
            'Kprime_0': 4.9,
            'Kdprime_0': -3.2e-11,
            'V_0': 1.22385753087e-05,
            'name': 'fper',
            'H_0': -2.60098326e+05, #- 1000.,
            'molar_mass': 0.0718444,
            'equation_of_state': 'hp_tmt',
            'n': 2.0,
            'formula': {'Fe': 1.0, 'O': 1.0},
            'Cp': [53.334316, 0.00779203541, -325553.876, -75.023374],
        }
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



###########################
initial_alphas=[1.0, 1.0, 1.0]
initial_enthalpy_interaction=[[11.e3, 0.0e3], [14.e3]]
###########################

# Configurational entropy
class wuestite_ss(burnman.SolidSolution):
    def __init__(self):
        # Name
        self.name='non-stoichiometric wuestite, ferric and ferrous iron treated as separate atoms in Sconf'
        endmembers = [[minerals.HP_2011_ds62.per(), '[Mg]O'],[wustite(), '[Fe]O'],[defect_wustite(), '[Fef2/3Vc1/3]O']]

        # Interaction parameters
        alphas=initial_alphas
        enthalpy_interaction=initial_enthalpy_interaction
        volume_interaction=[[0., 0.],[0.]]

        burnman.SolidSolution.__init__(self, endmembers, \
                          burnman.solutionmodel.AsymmetricRegularSolution(endmembers, alphas, enthalpy_interaction, volume_interaction) )

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
wus=wuestite_ss()
mt=minerals.HP_2011_ds62.mt()
hem=minerals.HP_2011_ds62.hem()
oxygen=minerals.HP_2011_fluids.O2()



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
    Xi=(1.-Xj)
    phii=Xi*ai/(Xi*ai + Xj*aj)
    phij=Xj*aj/(Xi*ai + Xj*aj)
    Hex=(2.*ai/(ai+aj)*phij*phij*Xi + 2.*aj/(ai+aj)*phii*phii*Xj)*Wij
    Sex=-constants.gas_constant*(Xi*np.log(Xi) + 2.*Xj/3.*np.log(2.*Xj/3.) + Xj/3.*np.log(Xj/3.))
    return Gi*Xi + Gj*Xj - Sex*T + Hex

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

def fO2_wus_eqm_Fe(ys, Wij, Gi, Gj): # last fO2 should be zero
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
       
    y=ys[len(ys)-1]
    Xj=3.*y
    gibbs_wus=gibbs_fixed_PT(Xj, ai, aj, Wij, Gi, Gj)
    gibbs_fcc=fcc.gibbs
    mu_O2=2.*(gibbs_wus - (1.-y)*gibbs_fcc) 
    fO2_eqm_with_iron=np.log10(np.exp((mu_O2-oxygen.gibbs)/(constants.gas_constant*T)))
    fO2[len(ys)-1] = fO2[len(ys)-1] - fO2_eqm_with_iron
    return fO2

Ty=[['1273.15', 0.050], ['1373.15', 0.050], ['1473.15', 0.050], ['1573.15', 0.050]]

for strT, y_fcc in Ty:
    temperatures=[]
    logfO2=[]
    ys=[]
    for line in open('Bransky_Hed_1968.dat'):
        content=line.strip().split()
        if content[0] != '%' and content[0]==strT:
            temperatures.append(float(content[0]))
            logfO2.append(float(content[1]))
            ys.append(float(content[2])/100.)
            
    ys_orig=copy.deepcopy(ys)
    logfO2_orig=copy.deepcopy(logfO2)

    temperatures.append(float(strT))
    logfO2.append(0.0)
    ys.append(y_fcc)


    ys=np.array(ys)
    T=float(strT)
    oxygen.set_state(Pr, T)
    fcc.set_state(Pr, T)
    hem.set_state(Pr, T)
#fO2s=np.empty_like(ys)
#d=0.001
#for i, y in enumerate(ys):
#    mu_O2_Sundman=2*((1.-y+d)/(1.-y-d)*G_wustite(y+d, T)[1] - G_wustite(y-d, T)[1])/((1.-y+d)/(1.-y-d)-1.)
#    fO2s[i]=np.log10(np.exp((mu_O2_Sundman-oxygen.gibbs)/(constants.gas_constant*T)))
    
    aj=10.
    guesses=[14.e3, -429440., -450000.]
    popt, pcov = optimize.curve_fit(fO2_wus_eqm_Fe, ys, logfO2, guesses)
    Wij, Gi, Gj = popt

    print popt, hem.gibbs/3.

    plt.plot( ys_orig, fO2_wus(ys_orig, aj, Wij, Gi, Gj), 'r-', linewidth=1, label='Fit for '+strT+'K')
    plt.plot( ys_orig, logfO2_orig, marker='o', linestyle='none', label='Bransky and Hed, 1968')
    # plt.plot( ys, G_wustite(ys, T)[1]-gibbs_wus_full(ys, aj, Wij, Gi, Gj), 'r-', linewidth=1, label='Gibbs difference')

    plt.legend(loc="upper left")
    plt.ylabel("log10(fO2)")
    plt.xlabel("y in Fe(1-y)O")
    plt.show()
    




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
