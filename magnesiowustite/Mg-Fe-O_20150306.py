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
            'V_0': 1.22385753087e-05 ,
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
            'V_0': 1.10419547534e-05 ,
            'Cp': [163.9/3., 0.0, -2257200.0/3., -657.6/3.] ,
            'a_0': 2.79e-05 ,
            'K_0': 150.0e+9 ,
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
hcp=Myhill_calibration_iron.hcp_iron()
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
        self.endmembers = [[per, '[Mg]O'],[stoichiometric_wustite, '[Fe]O'],[defect_wustite, '[Fef2/3Vc1/3]O']]
        self.type='full_subregular'
        self.averaging_scheme='KV_linear'
        # Interaction parameters
        self.enthalpy_interaction=[[[10000., 8000.],[42500.,19000.]],[[-39612, -4236.]]]
        self.volume_interaction=[[[0., 0.],[0.,0.]],[[0., 0.]]]
        self.bulk_modulus_interaction=[[[0., 0.],[0.,0.]],[[-20.e9, -20.e9]]]
        self.entropy_interaction=[[[0., 0.],[0.,0.]],[[-4.048, -4.048]]]

        burnman.SolidSolution.__init__(self)

class spinel(burnman.SolidSolution):
    def __init__(self):
        # Name
        self.name='Magnetite-magnesioferrite spinel'
        self.endmembers = [[mft, '[Fef][Fef1/2Mg1/2]2O4'],[mt, '[Fef][Fe1/2Fef1/2]2O4']]
        self.type='subregular'
        # Interaction parameters
        self.enthalpy_interaction=[[[0., 0.]]]
        self.volume_interaction=[[[0., 0.]]]
        self.entropy_interaction=[[[0., 0.]]]

        burnman.SolidSolution.__init__(self)


wus=wuestite_ss()
sp=spinel()

'''
Volumes
'''
Z=4.
nA=6.02214e23
voltoa=1.e30


# Compiled data of wustite edge lengths based on their composition (in terms of y, where wustite formula is FeyO)
ausn=np.array([431.7, 428.0, 431, 430.6, 430.7])
yusn=1.-np.array([0.98, 0.92, 0.95, 0.94, 0.943])

ashk=np.array([430.8])
yshk=1.-np.array([0.94])

axrd=np.array([430.4, 428.9, 429.9, 429, 430.2, 430.8, 430.8, 432.5, 432.3])
yxrd=1.-np.array([0.94, 0.924, 0.941, 0.90, 0.93, 0.947, 0.94, 0.98, 0.98])

atheory=np.array([433.2])
ytheory=1.-np.array([1.00])

afit=np.array([431, 430.6, 430.7, 430.8, 430.4, 429, 430.2, 430.8, 430.8, 432.5, 432.3, 433.2])
yfit=1.-np.array([0.95, 0.94, 0.943, 0.94, 0.94, 0.90, 0.93, 0.947, 0.94, 0.98, 0.98, 1.00])
# Katsura et al., 1967 also found 432.3 for "stoichiometric" wustite

vusn=ausn*ausn*ausn*1e-36*nA/Z
vshk=ashk*ashk*ashk*1e-36*nA/Z
vxrd=axrd*axrd*axrd*1e-36*nA/Z
vtheory=atheory*atheory*atheory*1e-36*nA/Z
vfit=afit*afit*afit*1e-36*nA/Z

def linear_trend(x, a, b):
    return a + b*x

#def HH1992_volume(temperature, y):
#    return (11.8563 + 0.617125e-3*(temperature) + 369.056*np.exp(-temperature/300.) - 23.5379*y + 2.270*y*y - 0.07298*y*y*y)

# 1.00456 is a fudge to fit the tabulated data without the given (useless) pressure term
def HH1992_volume(temperature, y):
    return (1.00456*(11.8563 + 0.617125e-3*temperature + 0.369056*np.exp(-temperature/300.)) - 3.49*y)*1.e-6

v_0, v_grad = optimize.curve_fit(linear_trend, yfit, vfit)[0]

compositions=np.linspace(0., 1./3., 101)
volumes=np.empty_like(compositions)
for idx, y in enumerate(compositions):
    volumes[idx]=HH1992_volume(298.15, y)


volume_FeO=linear_trend(0., v_0, v_grad)
volume_defect=linear_trend(1./3., v_0, v_grad)

print volume_FeO, volume_defect

wus.endmembers[1][0].params['V_0']=volume_FeO
wus.endmembers[2][0].params['V_0']=volume_defect

# Plot data and fit
plt.plot( compositions, volumes, '--', linewidth='1', label='Haas and Hemingway (1992)')
plt.plot( [0., 1./3.], [wus.endmembers[1][0].params['V_0'],wus.endmembers[2][0].params['V_0']], linewidth='1', label='This study')
plt.plot( yusn, vusn, marker='o', linestyle='none', label='S+G compilation, ultrasound')
plt.plot( yxrd, vxrd, marker='o', linestyle='none', label='S+G compilation, XRD')
plt.plot( yshk, vshk, marker='o', linestyle='none', label='S+G compilation, shock')
plt.plot( ytheory, vtheory, marker='o', linestyle='none', label='Theory; Jette and Foote, 1933')


plt.xlim(0., 1./3.)
plt.legend(loc="upper right")
plt.ylabel("V (m^3)")
plt.xlabel("y in Fe(1-y)O")
plt.show()


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
Plot the effect of pressure at different temperatures
'''

# Find stable form of iron
def stable_iron(P, T):
    Gmin=10000000.
    for iron_phase in [bcc, fcc, hcp]:
        iron_phase.set_state(P, T)
        if iron_phase.gibbs < Gmin:
            stable_phase=iron_phase
            Gmin = iron_phase.gibbs
    return stable_phase

temperatures=np.linspace(973.15, 1773.15, 5)
pressures=np.linspace(1.e5, 30.e9, 101)
for T in temperatures:
    f=np.empty_like(pressures)
    for i, P in enumerate(pressures):
        stable_phase=stable_iron(P, T)
        #print P, T, stable_phase.params['name']
        f[i]=f_to_y(optimize.fsolve(wus_eqm(stable_phase), 0.01, args=(P, T, 0.))[0])

    plt.plot( pressures/1.e9, f, linewidth=1, label=str(T)+'K')

McC_P=[]
McC_Perr=[]
McC_c=[]
McC_cerr=[]
for line in open('McCammon_FeO_pressure.dat'):
    content=line.strip().split()
    if content[0] != '%':
        
        McC_P.append(float(content[1]))
        McC_Perr.append(float(content[2]))
        McC_c.append(float(content[7]))
        McC_cerr.append(float(content[8]))

plt.errorbar( McC_P, McC_c, xerr=[McC_Perr, McC_Perr], yerr=[McC_cerr, McC_cerr], fmt='--o', linestyle='none', label="McCammon data")
plt.xlabel("Pressure (GPa)")
plt.ylabel("x")
plt.legend(loc='upper right')
plt.show()


'''
FITTING PROCEDURE
'''

def fit_nonideal(data, Wh01, Wh10, Wh02, Wh20):
    # Solid solution tweaking
    wus.enthalpy_interaction[0]=[Wh01, Wh10], [Wh02, Wh20]
    burnman.SolidSolution.__init__(wus)
    

    # T is 1473.15 where mu_0 is required 
    XFe3_0=optimize.fsolve(wus_eqm_cation_fractions(fcc), 0.01, args=(Pr, 1473.15, 0.))
    mu_0=chemical_potentials([wus], [{'O': 2.0}])

    calc=[]
    for datum in data:
        # 2 types of data to fit
        # Chemical potentials of oxygen and compositions in eqm with iron
        
        
        if datum[0] == 'mu':
            XMg=float(datum[1])
            T=float(datum[2])
            XFe3=optimize.fsolve(wus_eqm_cation_fractions(fcc), 0.01, args=(Pr, T, XMg))[0]
            calc.append(chemical_potentials([wus], [{'O': 2.0}])-mu_0)
        elif datum[0] == 'comp':
            XMg=float(datum[1])
            T=float(datum[2])
            XFe3=optimize.fsolve(wus_eqm_cation_fractions(fcc), 0.01, args=(Pr, T, XMg))[0]
            Fe3overFe=XFe3/(1.-XMg)
            calc.append(Fe3overFe)
        elif datum[0] == 'fO2':
            composition=datum[1]
            T=float(datum[2])
            wus.set_composition(composition)
            wus.set_state(Pr, T)
            oxygen.set_state(Pr, T)
            calc.append(np.log10(fugacity(oxygen, [wus])))
         
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
        obs_x.append(['comp', float(content[1]), 1473.15])
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
            obs_x.append(['mu', float(content[6]), 1473.15])
            obs_y.append(-4.*Fc*(float(content[2])+EMF0)+constants.gas_constant*T*np.log(0.20946*Pair) + 337.06e3) # muO2
            sigma_obs_y.append(-4.*Fc*float(content[3])) # muO2err
        data_EMF.append([float(content[2]), float(content[3]), float(content[10]), float(content[11]), float(content[12])])
        if content[2] != 'nan' and content[3] != 'nan' and content[10] != 'nan' :
            obs_x.append(['mu', float(content[10]), 1473.15])
            obs_y.append(-4.*Fc*(float(content[2])+EMF0)+constants.gas_constant*T*np.log(0.20946*Pair) + 337.06e3)  # muO2
            sigma_obs_y.append(-4.*Fc*float(content[3])) # muO2err

EMFb, EMFberr, XMgb, XMgberr, volumeb = zip(*data_EMF)

muO2b=-4.*Fc*(np.array(EMFb)+EMF0)+constants.gas_constant*T*np.log(0.20946*Pair)
muO2berr=-4.*Fc*np.array(EMFberr)

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

'''
for line in open('X-fO2_Spiedel_1967.dat'):
    content=line.strip().split()
    if content[0] != '%':
        obs_x.append(['fO2', FFwt_to_mol(float(content[1]), float(content[2])), 1573.15])
        obs_y.append(float(content[0])) # log10fO2
        sigma_obs_y.append(0.01)
'''



obs_y=np.array(obs_y)
sigma_obs_y=np.array(sigma_obs_y)


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


print popt

print 'MgO, FeO, diff logfO2'
mgnumbers=[]
ferriccomponents=[]
difffo2=[]
for datum in fO2_data:
    wus.set_composition(datum[1])
    wus.set_state(Pr, T)
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
