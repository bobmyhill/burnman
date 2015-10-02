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
import matplotlib.cm as cm
from matplotlib.ternary import Ternary
from matplotlib.path import Path
from matplotlib.patches import PathPatch


import scipy.optimize as optimize
from scipy import interpolate

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

'''
Wustite model endmembers
'''

class wustite (Mineral):
    def __init__(self):
        formula='Fe1.0O1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'fper',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -265450.,
            'S_0': 58.0 ,
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


class fe23o (Mineral): # starting guess is hem/3.
    def __init__(self):
        formula='Fe2/3O'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'hem',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -258542 ,
            'S_0': 34.09 ,
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
Other endmembers
'''

per=minerals.HP_2011_ds62.per()
fper=minerals.HP_2011_ds62.fper()
stoichiometric_wustite=wustite()
defect_wustite=fe23o()
fcc=Myhill_calibration_iron.fcc_iron()
bcc=Myhill_calibration_iron.bcc_iron()
mt=minerals.HP_2011_ds62.mt()
mt.params['H_0'] = -1115.4e3
mft=minerals.HP_2011_ds62.mft()
iron=minerals.HP_2011_ds62.iron()
hem=minerals.HP_2011_ds62.hem()
oxygen=minerals.HP_2011_fluids.O2()

'''
Solid solutions - initial definitions
'''

'''
Sundman parameters for FeO-Fe2/3O (for comparison only)
'''
L0_23=-12324.4 # Note missing decimal point in Sundman 1991
L1_23=20070.0 # Note sign error (reported as -ve) and missing decimal point in Sundman 1991

'''
O'Neill parameters for MgO-'FeO' (for comparision only)
'''
A0=10.57e3
A1=0.41e3


class wuestite_ss(burnman.SolidSolution):
    def __init__(self):
        # Name
        self.name='non-stoichiometric wuestite, ferric and ferrous iron treated as separate atoms in Sconf'
        endmembers = [[per, '[Mg]O'],[stoichiometric_wustite, '[Fe]O'],[defect_wustite, '[Fef2/3Vc1/3]O']]

        # Interaction parameters
        enthalpy_interaction=[[[10000., 8000.],[42500.,19000.]],[[-39612, -4236.]]]
        volume_interaction=[[[0., 0.],[0.,0.]],[[0., 0.]]]
        entropy_interaction=[[[0., 0.],[0.,0.]],[[-4.048, -4.048]]]

        burnman.SolidSolution.__init__(self, endmembers, \
                          burnman.solutionmodel.SubregularSolution(endmembers, enthalpy_interaction, volume_interaction, entropy_interaction) )

class spinel(burnman.SolidSolution):
    def __init__(self):
        # Name
        self.name='Magnetite-magnesioferrite spinel'
        endmembers = [[mft, '[Fef][Fef1/2Mg1/2]2O4'],[mt, '[Fef][Fe1/2Fef1/2]2O4']]

        # Interaction parameters
        enthalpy_interaction=[[[0., 0.]]]
        volume_interaction=[[[0., 0.]]]
        entropy_interaction=[[[0., 0.]]]

        burnman.SolidSolution.__init__(self, endmembers, \
                          burnman.solutionmodel.SubregularSolution(endmembers, enthalpy_interaction, volume_interaction, entropy_interaction) )


wus=wuestite_ss()
sp=spinel()

'''
Compare Sundman model for the Fe-O system with the current model
'''

Pr=1.e5
T=1000.
compositions=np.linspace(0., 1., 101)
Hex_model=np.empty_like(compositions)
Hex_Sundman=np.empty_like(compositions)
for i, f in enumerate(compositions):
    wus.set_composition([0.0, 1.-f, f])
    wus.set_state(Pr, T)

    y3=f
    y2=1.-f

    Hex_model[i]=wus.excess_enthalpy
    Hex_Sundman[i]=y2*y3*(L0_23 + (y2-y3)*L1_23)

plt.plot( compositions, Hex_model, linewidth=1, label="Model")
plt.plot( compositions, Hex_Sundman, linewidth=1, label="Sundman")
plt.xlabel("f")
plt.ylabel("H mixing")
plt.legend(loc='upper right')
plt.show()


'''
Functions to return the equilibrium composition of wustite with another mineral at P and T
'''

wus.endmembers[2][0].params['formula'].pop("Vc", None)

def wus_eqm(mineral):
    def mineral_equilibrium(c, P, T, XMgO):
        wus.set_composition([XMgO, 1.0-XMgO-c[0],c[0]])
        wus.set_state(P, T)
        mineral.set_state(P, T)
        mu_mineral=chemical_potentials([wus], [mineral.params['formula']])[0]   
        return  mu_mineral-mineral.gibbs
    return mineral_equilibrium

def wus_eqm_w_MgFe_mineral(mineral):
    def mineral_equilibrium(c, P, T, pMgmbr):
        XMgO=c[0]
        XFeO=c[1]
        wus.set_composition([XMgO, XFeO, 1.-XMgO-XFeO])
        wus.set_state(P, T)
        mineral.set_composition([pMgmbr, 1.-pMgmbr])
        mineral.set_state(P, T)
        mu_wus=chemical_potentials([wus], [mineral.endmembers[0][0].params['formula'], mineral.endmembers[1][0].params['formula']])
        mu_mineral=mineral.partial_gibbs
        return  [mu_wus[0] - mu_mineral[0], mu_wus[1] - mu_mineral[1]] 
    return mineral_equilibrium

def wus_eqm_cation_fractions(mineral):
    def mineral_equilibrium(c, P, T, XMg):
        XFe3=c[0]
        f=1.5*XFe3/(1. + 0.5*XFe3)
        XMgO=XMg/(1. + 0.5*XFe3)
        wus.set_composition([XMgO, 1.0-XMgO-f,f])
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

'''
FITTING PROCEDURE
'''

def fit_nonideal(data, Wh01, Wh10, Wh02, Wh20):
    # Solid solution tweaking
    enthalpy_interaction=[[[Wh01, Wh10], [Wh02, Wh20]],[[wus.solution_model.Wh[1][2], wus.solution_model.Wh[2][1]]]]
    volume_interaction=[[[0., 0.],[0.,0.]],[[0., 0.]]]
    entropy_interaction=[[[0., 0.],[0.,0.]],[[wus.solution_model.Ws[1][2], wus.solution_model.Ws[2][1]]]]
    
    burnman.SolidSolution.__init__(wus, wus.endmembers, \
                                       burnman.solutionmodel.SubregularSolution(wus.endmembers, enthalpy_interaction, volume_interaction, entropy_interaction) )
    

    # All data at 1 temperature
    XFe3_0=optimize.fsolve(wus_eqm_cation_fractions(fcc), 0.01, args=(Pr, T, 0.))
    mu_0=chemical_potentials([wus], [{'O': 2.0}])

    calc=[]
    for datum in data:
        # 2 types of data to fit
        # Chemical potentials of oxygen and compositions in eqm with iron
        XMg=float(datum[1])
        XFe3=optimize.fsolve(wus_eqm_cation_fractions(fcc), 0.01, args=(Pr, T, XMg))[0]
        Fe3overFe=XFe3/(1.-XMg)
        
        if datum[0] == 'mu':
            calc.append(chemical_potentials([wus], [{'O': 2.0}])-mu_0)
        elif datum[0] == 'comp':
            calc.append(Fe3overFe)
         
    return np.array(calc)


'''
Input data for parameter fitting
'''

obs_x=[]
obs_y=[]
sigma_obs_y=[]

Pr=1.e5
T=1473.15

data_Fe3=[]
weighting=20.
for line in open('ONeill_et_al_2003_Mg_Fe3_periclase.dat'):
    content=line.strip().split()
    if content[0] != '%':
        data_Fe3.append(map(float, content[1:]))
        obs_x.append(['comp', float(content[1])])
        obs_y.append(float(content[3]))
        sigma_obs_y.append(float(content[4])/weighting)

XMga, XMgerra, Fe3overFea, Fe3overFeerra, XFe3a, XFe3erra = zip(*data_Fe3)

Fc=96.48456
Pair=0.97 # bar
EMF0=1.1
data_EMF=[]
for line in open('ONeill_et_al_2003_Mg_EMF_periclase.dat'):
    content=line.strip().split()
    if content[0] != '%':
        data_EMF.append([float(content[2]), float(content[3]), float(content[6]), float(content[7]), float(content[8])])
        if content[2] != 'nan' and content[3] != 'nan' and content[6] != 'nan' :
            obs_x.append(['mu', float(content[6])])
            obs_y.append(-4.*Fc*(float(content[2])+EMF0)+constants.gas_constant*T*np.log(0.20946*Pair) + 337.06e3) # muO2
            sigma_obs_y.append(-4.*Fc*float(content[3])) # muO2err
        data_EMF.append([float(content[2]), float(content[3]), float(content[10]), float(content[11]), float(content[12])])
        if content[2] != 'nan' and content[3] != 'nan' and content[10] != 'nan' :
            obs_x.append(['mu', float(content[10])])
            obs_y.append(-4.*Fc*(float(content[2])+EMF0)+constants.gas_constant*T*np.log(0.20946*Pair) + 337.06e3)  # muO2
            sigma_obs_y.append(-4.*Fc*float(content[3])) # muO2err

EMFb, EMFberr, XMgb, XMgberr, volumeb = zip(*data_EMF)

muO2b=-4.*Fc*(np.array(EMFb)+EMF0)+constants.gas_constant*T*np.log(0.20946*Pair)
muO2berr=-4.*Fc*np.array(EMFberr)

obs_y=np.array(obs_y)
sigma_obs_y=np.array(sigma_obs_y)

oxygen.set_state(Pr, T)

guesses=[11000., 11000., 30000., 30000.]
popt, pcov = optimize.curve_fit(fit_nonideal, obs_x, obs_y, guesses, sigma_obs_y) 

print 
print 'Optimised interaction parameters'
for i in range(len(pcov)):
    print popt[i], '+/-', np.sqrt(pcov[i][i])

print
print 'Covariance matrix'
print pcov


'''
Calculate compositions of coexisting spinels
'''
XMg_spinels=np.linspace(0.01, 0.99, 101)
sp_eqm_XMgO=np.empty_like(XMg_spinels)
sp_eqm_XFeO=np.empty_like(XMg_spinels)
for i, XMg_spinel in enumerate(XMg_spinels):
    comp=optimize.fsolve(wus_eqm_w_MgFe_mineral(sp), [0.3,0.3] , args=(Pr, T, XMg_spinel))
    sp_eqm_XMgO[i] = comp[0]
    sp_eqm_XFeO[i] = comp[1]


'''
Create arrays for plotting output data
'''

XMgOs=np.linspace(0.0, 0.95, 101)
gibbs_wus=np.empty_like(XMgOs)

XMg=np.empty_like(XMgOs)
Xdefect=np.empty_like(XMgOs)

XFe3=np.empty_like(XMgOs)
XFe3_ONeill=np.empty_like(XMgOs)

Fe3overFe=np.empty_like(XMgOs)
Fe3overFe_ONeill=np.empty_like(XMgOs)

muO2_fcc_fper=np.empty_like(XMgOs)
muO2_fcc_fper_ONeill=np.empty_like(XMgOs)


# IW buffer check from O'Neill - compositional difference of 0.000005 mole fraction Fe2/3O accounts for the difference in mu_O2.
T=1473.15
f=optimize.fsolve(wus_eqm(fcc), 0.01, args=(Pr, T, 0.0))[0]
print (f*2./3.)/(1.-f/3.)
oxygen.set_state(Pr, T)
print chemical_potentials([wus], [{'O': 2.0}])- oxygen.gibbs


for i, XMgO in enumerate(XMgOs):
    wus.set_composition([XMgO, 1.-XMgO, 0.0])
    wus.set_state(Pr, T)
    gibbs_wus[i]=wus.excess_gibbs

    Xdefect[i]=optimize.fsolve(wus_eqm(fcc), 0.01, args=(Pr, T, XMgO))[0]
    oxygen.set_state(Pr, T)
    #print wus.excess_enthalpy, T*wus.excess_entropy, wus.excess_gibbs
    muO2_fcc_fper[i]=chemical_potentials([wus], [{'O': 2.0}])- oxygen.gibbs

    XMg[i]=wus.molar_fractions[0]/(wus.molar_fractions[0]+wus.molar_fractions[1]+wus.molar_fractions[2]*2./3.)

    XFe3[i]=wus.molar_fractions[2]*(2./3.)/(wus.molar_fractions[0]+wus.molar_fractions[1]+wus.molar_fractions[2]*2./3.)

    Fe3overFe[i]=wus.molar_fractions[2]*(2./3.)/(wus.molar_fractions[1]+wus.molar_fractions[2]*(2./3.))

    mu_O2_IW=-337.06e3
    muO2_fcc_fper_ONeill[i]=mu_O2_IW +2.*constants.gas_constant*T*np.log(1.-XMg[i]) + 2.*XMg[i]*XMg[i]*(10.57e3 + 0.41e3*(3.-4.*XMg[i]))
    XFe3_ONeill[i]=np.power(1.-XMg[i], 1.5)*(0.103 - 0.203*XMg[i] + 0.213*XMg[i]*XMg[i])
    Fe3overFe_ONeill[i]=XFe3_ONeill[i]/(1.-XMg[i])


'''
Plot chemical potential of oxygen along the iron-buffered wustite curve
'''

plt.plot( XMg, muO2_fcc_fper, linewidth=1, label='Model muO2')
plt.errorbar( XMgb, muO2b, xerr=[XMgberr, XMgberr], yerr=[muO2berr, muO2berr], fmt='--o', linestyle='none', label="O'Neill data")
plt.plot( XMg, muO2_fcc_fper_ONeill, linewidth=1, label='ONeill muO2')
plt.xlabel("XMgO")
plt.ylabel("Chemical potential of O2")
plt.legend(loc='upper right')
plt.show()


'''
Plot Fe3+/sum(Fe) of wustite along the iron-buffered wustite curve
'''

plt.plot( XMg, Fe3overFe, linewidth=1, label='Model composition')
plt.errorbar( XMga, Fe3overFea, xerr=[XMgerra, XMgerra], yerr=[Fe3overFeerra, Fe3overFeerra], fmt='--o', linestyle='none', label="O'Neill data")
plt.plot( XMg, Fe3overFe_ONeill, linewidth=1, label="O'Neill composition")
plt.xlabel("XMgO")
plt.ylabel("Fe3+/sum(Fe)")
plt.legend(loc='upper right')
plt.show()

'''
Plot Fe3+ of wustite along the iron-buffered wustite curve
'''

plt.plot( XMg, XFe3, linewidth=1, label="Model composition")
plt.errorbar( XMga, XFe3a, xerr=[XMgerra, XMgerra], yerr=[XFe3erra, XFe3erra], fmt='--o', linestyle='none', label="O'Neill data")
plt.plot( XMg, XFe3_ONeill, linewidth=1, label="O'Neill composition")
plt.xlabel("XMgO")
plt.ylabel("XFe2/3O")
plt.legend(loc='upper right')
plt.show()

'''
Plot ternary phase diagram
'''

ter = Ternary()
ter.set_title("Points and Lines on a Ternary Plot")
ter.ab.plot((1.-XMg)/(1+0.5*XFe3), 1.5*XFe3/(1.+0.5*XFe3), 'r-', label="Iron-Magnesiowustite")
ter.ab.plot((1.-sp_eqm_XMgO), 1.-sp_eqm_XMgO-sp_eqm_XFeO, 'b-', label="Spinel-Magnesiowustite")
#ter.ab.plot([0.0, 1.0], [0.5, 0.0], 'r:', label="Equal B & C")
ter.ab.set_tiplabel("FeO")
ter.bc.set_tiplabel("FeO1.5")
ter.ca.set_tiplabel("MgO")
ter.legend()
plt.show()


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
