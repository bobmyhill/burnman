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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scipy.optimize as optimize

atomic_masses=read_masses()

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
            'Gprime_0': 'nan',
            'a_0': 5.13074989862e-05,
            'K_0': 153865172537.0,
            'G_0': 'nan',
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
Excess properties
'''

class wustite (Mineral):
    def __init__(self):
       formula='Fe1.0O1.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'fper',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -264500. , # -264353.6
            'S_0': 60.,
            'V_0': 1.2239e-05 , # From Simons (1980)
            'Cp': [5.33343160e+01,   7.79203541e-03,  -3.25553876e+05,  -7.50233740e+01] ,
            'a_0': 3.22e-05 ,
            'K_0': 1.52e+11 ,
            'Kprime_0': 4.9 ,
            'Kdprime_0': -3.2e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)

class defect_wustite (Mineral):
    def __init__(self):
       formula='Fe2/3Vc1/3O1.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'fper',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -259200. ,
            'S_0': 39.,
            'V_0': 1.10701e-05 , # From Simons (1980)
            'Cp': [-3.64959181e+00,   1.29193873e-02,  -1.07988127e+06,   1.11241795e+03] ,
            'a_0': 3.22e-05 ,
            'K_0': 1.52e+11 ,
            'Kprime_0': 4.9 ,
            'Kdprime_0': -3.2e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)


# Configurational entropy
class wuestite_ss(burnman.SolidSolution):
    def __init__(self):
        # Name
        self.name='non-stoichiometric wuestite, ferric and ferrous iron treated as separate atoms in Sconf'
        endmembers = [[minerals.HP_2011_ds62.per(), '[Mg]O'],[wustite(), '[Fe]O'],[defect_wustite(), '[Fe2/3Vc1/3]O']]

        # Interaction parameters
        alphas=[1.0, 1.0, 0.7]
        enthalpy_interaction=[[11.e3, 23.931e3], [-18.0e3]]
        volume_interaction=[[0., 0.],[0.]]

        burnman.SolidSolution.__init__(self, endmembers, \
                          burnman.solutionmodel.AsymmetricRegularSolution(endmembers, alphas, enthalpy_interaction, volume_interaction) )

wus=wuestite_ss()

'''
Volumes
'''
Z=4.
nA=6.02214e23
voltoa=1.e30

Pr=1.e5 # 1 atm.
P=1.e5 # 1 atm.
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

oxygen=minerals.HP_2011_fluids.O2()
wus.endmembers[2][0].params['formula'].pop("Vc", None)
oxygen.set_state(Pr, T)

delH_MgO_FeO=11.e3
def fitfO2(mineral):
    def fit(data, delH_MgO_Fe2O3):

        # Endmember tweaking
        #mineral.endmembers[2][0].params['H_0'] = H0_Fe2O3

        # Solid solution tweaking
        alphas=[1.0, 1.0, 0.7]
        enthalpy_interaction=[[11.e3, delH_MgO_Fe2O3], [-18.0e3]] 
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

guesses=[23931.]
popt, pcov = optimize.curve_fit(fitfO2(wus), compositions, oxygen_fugacities, guesses)

print popt

print 'MgO, FeO, diff logfO2'
for datum in fO2_data:
    wus.set_composition(datum[1])
    wus.set_state(P, T)
    print datum[1][0], datum[1][1], datum[0] - np.log10(fugacity(oxygen, [wus]))


fcc=fcc_iron()
mt=minerals.HP_2011_ds62.mt()

print Pr, T
oxygen.set_state(Pr, T)
fcc.set_state(Pr, T)
mt.set_state(Pr, T)


boundaries=[[fcc, 1.e5, 1160.+273.15, 11.7, -12.4], [mt, 1.e5, 1160.+273.15, 33.3, -9.8], [fcc, 1.e5, 1300.+273.15, 11.3, -10.82], [mt, 1.e5, 1300.+273.15, 38.9, -7.72]]

arr_wus_gibbs=[]
arr_mu_O2=[]
arr_compositions=[]
for boundary in boundaries:
    P=boundary[1]
    T=boundary[2]
    log10fO2=boundary[4]
    boundary[0].set_state(P, T) # set state of mineral
    oxygen.set_state(Pr, T)
    composition=FFwt_to_mol(100.-boundary[3], boundary[3]) # wt% to mol%
    mu_O2=constants.gas_constant*T*np.log(np.power(10., log10fO2)) + oxygen.gibbs # find mu_oxygen
    # x FeO + (1-x)Fe2O3 = xFeO + 3*(1-x)Fe2/3O3
    # 
    x=composition[1]
    y=(1.-x)/3.
    if boundary[0] == fcc:
        wus_gibbs=(1.-y)*boundary[0].gibbs + 0.5*mu_O2
    if boundary[0] == mt:
        wus_gibbs=(1.-y)/3.*boundary[0].gibbs + (0.5-((1.-y)/3.*2.))*mu_O2

    arr_compositions.append(1./(2.-y))
    arr_mu_O2.append(mu_O2)
    arr_wus_gibbs.append(wus_gibbs)

    wus.set_composition(composition)
    wus.set_state(boundary[1], boundary[2])
    print wus.gibbs - wus_gibbs, np.log10(fugacity(oxygen, [wus])) - log10fO2


phases, pressures, temperatures, compositions, logfO2s = zip(*boundaries)

print arr_compositions

plt.plot( [11.3, 19.7, 26.8, 35.7], [-10.82, -10.6, -9, -8], 'o', linestyle='none', label='gibbs')

plt.ylabel("Gibbs")
plt.xlabel("p(O)")
plt.legend(loc='lower right')
plt.show()

'''

# Can use these four values (at each temperature) to find values of Gwus, Gwusdefect, the enthalpy of mixing and an alpha.

def fit_wus(params, boundaries):
    wus.endmembers[1][0].params['H_0'] = params[0]
    wus.endmembers[2][0].params['H_0'] = params[1]
    enthalpy_interaction=[[0.0, 0.0],[params[2]]]
    volume_interaction=[[0.0, 0.0],[0.0]]
    alphas=[1.0, 1.0, params[3]]

    burnman.SolidSolution.__init__(wus, wus.endmembers, \
                                       burnman.solutionmodel.AsymmetricRegularSolution(wus.endmembers, alphas, enthalpy_interaction, volume_interaction) )

    minimize_array=[]
    for boundary in boundaries:
        P=boundary[1]
        T=boundary[2]
        log10fO2=boundary[4]
        boundary[0].set_state(P, T) # set state of mineral
        oxygen.set_state(Pr, T)
        composition=FFwt_to_mol(100.-boundary[3], boundary[3]) # wt% to mol%
        mu_O2=constants.gas_constant*T*np.log(np.power(10., log10fO2)) + oxygen.gibbs # find mu_oxygen
        # x FeO + (1-x)Fe2O3 = xFeO + 3*(1-x)Fe2/3O3
        # 
        x=composition[1]
        y=(1.-x)/3.
        if boundary[0] == fcc:
            wus_gibbs=(1.-y)*boundary[0].gibbs + 0.5*mu_O2
        if boundary[0] == mt:
            wus_gibbs=(1.-y)/3.*boundary[0].gibbs + (0.5-((1.-y)/3.*2.))*mu_O2

        wus.set_composition(composition)
        wus.set_state(boundary[1], boundary[2])
        if T > 1500.:
            minimize_array.append(wus.gibbs - wus_gibbs)
            minimize_array.append(fugacity(oxygen, [wus]) - np.power(10.,log10fO2))

    return minimize_array


print optimize.fsolve(fit_wus, [wus.endmembers[1][0].params['H_0'], wus.endmembers[2][0].params['H_0'], 0.0, 1.0], args=(boundaries))
'''
