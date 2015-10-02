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


'''
Iron from Sundman (1991)
'''

def magnetic_gibbs(T, Tc, beta, p):
    A = (518./1125.) + (11692./15975.)*((1./p) - 1.)
    tau=T/Tc
    if tau < 1: 
        f=1.-(1./A)*(79./(140.*p*tau) + (474./497.)*(1./p - 1.)*(np.power(tau, 3.)/6. + np.power(tau, 9.)/135. + np.power(tau, 15.)/600.))
    else:
        f=-(1./A)*(np.power(tau,-5)/10. + np.power(tau,-15)/315. + np.power(tau, -25)/1500.)
    return constants.gas_constant*T*np.log(beta + 1.)*f

def HSERFe(T):
    if T < 1811:
        gibbs=1224.83 + 124.134*T - 23.5143*T*np.log(T) - 0.00439752*T*T - 5.89269e-8*T*T*T + 77358.3/T
    else:
        gibbs=-25384.451 + 299.31255*T - 46.*T*np.log(T) + 2.2960305e31*np.power(T,-9.)
    return gibbs 

def gibbs_bcc_1bar(T):
    Tc=1043.
    beta=2.22
    p=0.4
    return HSERFe(T) + magnetic_gibbs(T, Tc, beta, p)

class bcc_iron():
    def __init__(self):
        self.params = {
            'formula': {'Fe': 1.0}}
    def set_state(self, pressure, temperature):
        self.gibbs = gibbs_bcc_1bar(temperature)

class fcc_iron (Mineral):
    def __init__(self):
        self.params = {
            'S_0': 35.7999958649,
            'a_0': 5.13074989862e-05,
            'K_0': 153865172537.0,
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


class hcp_iron (Mineral):
    def __init__(self):
        self.params = {
            'S_0': 28.3697737759,
            'a_0': 4.47488573521e-05,
            'K_0': 159072252939.0,
            'Kprime_0': 5.28929578177,
            'Kdprime_0': -3.32509012982e-11,
            'V_0': 6.76617030516e-06,
            'name': 'HCP iron',
            'H_0': 3282.46177656,
            'molar_mass': 0.055845,
            'equation_of_state': 'hp_tmt',
            'n': 1.0,
            'formula': {'Fe': 1.0},
            'Cp': [52.2754, -0.000355156, 790710.86, -619.07],
        }
        Mineral.__init__(self)

bcc=bcc_iron()
fcc=fcc_iron()
hcp=hcp_iron()

'''
Wustite endmembers and solid solution (first guess at properties)
'''

class wustite (Mineral):
    def __init__(self):
       formula='Fe1.0O1.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'fper',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -264500. , 
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
        endmembers = [[wustite(), '[Fe]O'],[defect_wustite(), '[Fe2/3Vc1/3]O']]

        # Interaction parameters
        alphas=[1.0, 0.7]
        enthalpy_interaction=[[-18.0e3]]
        volume_interaction=[[0.e-6]]
        entropy_interaction=[[0.]]

        burnman.SolidSolution.__init__(self, endmembers, \
                          burnman.solutionmodel.AsymmetricRegularSolution(endmembers, alphas, enthalpy_interaction, volume_interaction, entropy_interaction) )

wus=wuestite_ss()

acoeffs=np.array([[511623.74,-421.21897,0.22664278,-1.4495132e-5],[-2217082.6,1640.3836,-0.84216746,4.3221147e-5],[1857120.4,-1440.5715,0.74067011,-1.3350707e-5]])

# Gibbs on a 1 atom basis
def Gwus(xO,T):
    a=np.array([0.,0.,0])
    for i in range(3):
        a[i] = acoeffs[i][0] + acoeffs[i][1]*T + acoeffs[i][2]*T*T + acoeffs[i][3]*T*T*T
    return a[0] + a[1]*xO + a[2]*xO*xO

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

# Hazen and Jeanloz, 1984 -> VI[Fe2+1-3xFe3+2x-tvacx+t] IVFe3+t O
# 0.04 < x < 0.12
# t < x < 3.5 t

hp_fe=minerals.HP_2011_ds62.iron()
fper=minerals.HP_2011_ds62.fper()
mt=minerals.HP_2011_ds62.mt()
hem=minerals.HP_2011_ds62.hem()
O2=minerals.HP_2011_fluids.O2()
wus_SLB=minerals.SLB_2011.wuestite()


temperatures=np.linspace(800., 2000., 101)
gibbs_Sundman_iron=np.empty_like(temperatures)
gibbs_HP_iron=np.empty_like(temperatures)

for idx, temperature in enumerate(temperatures):
    hp_fe.set_state(1.e5, temperature)
    gibbs_Sundman_iron[idx]=gibbs_bcc_1bar(temperature)
    gibbs_HP_iron[idx]=hp_fe.gibbs

#plt.plot( temperatures, gibbs_Sundman_iron, linewidth=1., label='Sundman (1991)')
#plt.plot( temperatures, gibbs_HP_iron, linewidth=1., label='HP (2011)')
plt.plot( temperatures, gibbs_Sundman_iron-gibbs_HP_iron, linewidth=1., label='Sundman (1991) - HP (2011)')
plt.ylabel("Difference in Gibbs (J/mol)")
plt.xlabel("Temperature (K)")
plt.show()


''' 
Entropies
''' 

O2.set_state( 1e5, 298.15 )
wus.set_composition( np.array([1.0,0.0]) )
wus.set_state( 1e5, 298.15 )
x=0.5
print (Gwus(x,298.15) + (1.-x)*gibbs_bcc_1bar(298.15) + x*O2.gibbs*0.5) / x - wus.gibbs

# Data from Gronvold et al. 1993
S_ss=np.array([[0., 0., 0.],[0., 0., 0.]])
y_Sf=np.array([[0.947, 57.4365],[0.9379, 57.0905],[0.9254, 56.2217]])

for i, ysf in enumerate(y_Sf):
    f=x_to_f(y_to_x(ysf[0]))
    S_ss[0][i]=f
    S_ss[1][i]=ysf[1]-(wus.solution_model._configurational_entropy( [1.0-f, f] ) - wus.solution_model._endmember_configurational_entropy_contribution( [1.0-f, f]))
print S_ss


def linear_trend(x, a, b):
    return a + b*x

S_tr, S_tr_var = optimize.curve_fit(linear_trend, S_ss[0], S_ss[1])


wus.endmembers[0][0].params['S_0']=S_tr[0]
wus.endmembers[1][0].params['S_0']=S_tr[0]+S_tr[1]

print wus.endmembers[0][0].params['S_0'], wus.endmembers[1][0].params['S_0']
plt.plot( S_ss[0], S_ss[1], 'o', linewidth=3.)
plt.plot( np.linspace(0.,1.,101.), linear_trend(np.linspace(0.,1.,101.), S_tr[0], S_tr[1]), 'r--', linewidth=3.)
plt.plot( np.linspace(0.,1.,101.), linear_trend(np.linspace(0.,1.,101.), wus.endmembers[0][0].params['S_0'], wus.endmembers[1][0].params['S_0']-wus.endmembers[0][0].params['S_0']), 'b-', linewidth=1.)
plt.xlim(0.0,1.0)
plt.ylim(0.,70.0)
plt.ylabel("Configurational entropy of solution (J/K/mol)")
plt.xlabel("Fraction ferric iron")
plt.show()


'''
Plot the heat capacities of various wustites
'''

def Cp(x, a, b, c, d):
    return a + b*x + c/(x*x) + d/np.sqrt(x) 

f=x_to_f(y_to_x(0.9))
Cp_ss=[(1.-f)*wustite().params['Cp'][i] + f*defect_wustite().params['Cp'][i] for i in range(4)]

# Data from Gronvold et al. (1993; real and extrapolated)
Cp_FeO=np.array([[298.15, 300.0, 350., 400.0, 450., 500.0, 550., 600.0, 650., 700.0, 750., 800.0, 850., 900.0, 950., 1000.0, 1050., 1100.0, 1150., 1200.0, 1250.],[5.730, 5.739, 5.943, 6.095, 6.218, 6.323, 6.415, 6.498, 6.576, 6.648, 6.718, 6.784, 6.848, 6.910, 6.970, 7.029, 7.086, 7.142, 7.197, 7.251, 7.303]])

Cp_Fe90O=np.array([[298.15, 300.0, 350., 400.0, 450., 500.0, 550., 600.0, 650., 700.0, 750., 800.0, 850., 900.0, 950., 1000.0, 1050., 1100.0, 1150., 1200.0, 1250.],[5.909, 5.913, 6.010, 6.077, 6.127, 6.168, 6.203, 6.234, 6.263, 6.292, 6.320, 6.349, 6.379, 6.410, 6.443, 6.478, 6.514, 6.553, 6.595, 6.639, 6.686]])

Cp_Fe9427O=np.array([[844.61,850.66,853.76,854.41,858.79,871.08,885.84,899.46,912.82,925.52,938.37,951.47,964.58],[10.282,9.787,348.67,184.01,89.69,11.195,6.898,6.937,6.884,6.905,6.721,6.744,6.753]])

Ci, Ci_var = optimize.curve_fit(Cp, Cp_FeO[0], Cp_FeO[1])
Ci90, Ci90_var = optimize.curve_fit(Cp, Cp_Fe90O[0], Cp_Fe90O[1])
Ci_defect=(Ci90 - 0.7*Ci)/0.3

plt.plot( Cp_FeO[0], Cp(Cp_FeO[0], Ci[0], Ci[1], Ci[2], Ci[3]), 'b--', linewidth=3., label='est Cp FeO')
plt.plot( Cp_FeO[0], Cp_FeO[1], 'o', linewidth=3., label='est Cp FeO')

plt.plot( Cp_Fe90O[0], Cp(Cp_Fe90O[0], Cp_ss[0], Cp_ss[1], Cp_ss[2], Cp_ss[3])/8.3145, 'g--', linewidth=3., label='est Cp Fe0.9O')
plt.plot( Cp_Fe90O[0], Cp_Fe90O[1], 'o', linewidth=3., label='est Cp Fe0.9O')

plt.plot( Cp_Fe9427O[0], Cp_Fe9427O[1], 'o', linewidth=3., label='Cp Fe0.9427O')

def Cp_Wus1993(T,a,b,c,d):
    return a + b*T + c/(T*T) + d*T*T*T

Tarr=np.linspace(300.,1300.,101)
plt.plot( Tarr, Cp_Wus1993(Tarr,51.129,4.85282e-3,-334.9089e3,-40.433488e-12)/8.3145, 'r--', linewidth=3., label='Cp Fe0.9379O')
plt.plot( Tarr, Cp_Wus1993(Tarr,50.804,4.16562e-3,-248.1707e3,451.701487e-12)/8.3145, 'r-', linewidth=3., label='Cp Fe0.9254O')


plt.xlim(250.0,1300.0)
plt.ylim(5.0,8.0)
plt.title("Comparison of Gibbs free energies of wuestite")
plt.ylabel("Cp/R wuestite")
plt.xlabel("Temperature (K)")
plt.legend(loc='lower right')
plt.show()

'''
Functions to find the chemical potentials of Fe and Fe3O4 for different compositions of wuestite at each pressure. 
'''

# Function to return the equilibrium composition of wustite with another mineral at P and T
wus.endmembers[1][0].params['formula'].pop("Vc", None)
def wus_eqm(mineral):
    def mineral_equilibrium(c, P, T):
        wus.set_composition([1.0-c[0],c[0]])
        wus.set_state(P, T)
        mineral.set_state(P, T)
        mu_mineral=chemical_potentials([wus], [mineral.params['formula']])[0]   
        return  mu_mineral-mineral.gibbs
    return mineral_equilibrium

# Function to return the equilibrium composition of wustite with bcc iron at P and T
def bcc_eqm(comp, P, T):
    c=comp[0]
    Gexcesses=wus.solution_model.excess_partial_gibbs_free_energies(P, T, [1.0-c,c])
    mu=[wus.endmembers[i][0].calcgibbs(P,T) + Gexcesses[i] for i in range(2)]
    mu_iron=3*mu[0] - 3*mu[1]

    return  mu_iron-gibbs_bcc_1bar(T)

# Function to return the equilibrium composition of wustite with fcc iron at P and T
def fcc_eqm(comp, P, T):
    c=comp[0]
    Gexcesses=wus.solution_model.excess_partial_gibbs_free_energies(P, T, [1.0-c,c])
    mu=[wus.endmembers[i][0].calcgibbs(P,T) + Gexcesses[i] for i in range(2)]
    mu_iron=3*mu[0] - 3*mu[1]
    fcc.set_state(P, T)
    return  mu_iron-fcc.gibbs

# Function to return the equilibrium composition of wustite with magnetite at P and T
def mt_eqm(comp, P, T):
    c=comp[0]
    Gexcesses=wus.solution_model.excess_partial_gibbs_free_energies(P, T, [1.0-c,c])
    mu=[wus.endmembers[i][0].calcgibbs(P,T) + Gexcesses[i] for i in range(2)]
    mu_mt=mu[0] + 3*mu[1]
    return mu_mt-mt.calcgibbs(P,T)

# Function to return the equilibrium temperature between bcc iron, wustite and magnetite
def iron_wus_mt_eqm(T, P):
    return optimize.fsolve(bcc_eqm, 0.16, args=(P, T))[0] - optimize.fsolve(mt_eqm, 0.16, args=(P, T))[0]


''' 
Fitting function
'''
# data should be the temperatures and compositions of the W1 defect structure of wustite (see Vallet and Raccah, 1964, 1965)
# as published in Hazen and Jeanloz, 1984 (Figure 3)
# Data taken from 'Lykasov_Fe-FeO.dat', 'Vallet_FeOW1-Fe3O4.dat' and 'Darken_FeOW1-Fe3O4.dat'
# Variables to fit should be H_0 for the two wustites, alpha for the defect wustite and W between the wustites


def fit_params(mineral):
    def fit(data, H0_FeO, H0_defect_wustite, alpha_defect_wustite, interaction):

        # Endmember tweaking
        mineral.endmembers[0][0].params['H_0'] = H0_FeO
        mineral.endmembers[1][0].params['H_0'] = H0_defect_wustite

        # Solid solution tweaking
        alphas=[1.0, alpha_defect_wustite]
        enthalpy_interaction=[[interaction]] 
        burnman.SolidSolution.__init__(wus, wus.endmembers, \
                                           burnman.solutionmodel.AsymmetricRegularSolution(wus.endmembers, alphas, enthalpy_interaction) )
        
        compositions=[]
        for coexisting_mineral, temperature in data:
            compositions.append(1.0-f_to_y(optimize.fsolve(wus_eqm(coexisting_mineral), 0.16, args=(1.e5, temperature))[0]))
        compositions=np.array(compositions)
        return compositions
    return fit

minerals_temperatures=[[fcc, 1424.62653692], [fcc, 1376.75520075], [fcc, 1323.39397471], [fcc, 1264.61208554], [fcc, 1218.11479518], [bcc, 1166.14020162], [bcc, 1120.99807704], [bcc, 1071.7589884], [bcc, 1023.91911893], [bcc, 976.072956121], [fcc, 1222.07160224], [bcc, 1172.8465432], [bcc, 1121.78241653], [bcc, 1072.56293888], [bcc, 1021.49881221], [bcc, 972.212357893], [bcc, 922.987298852], [bcc, 871.878521071], [fcc, 1223.83253043], [bcc, 1172.77398515], [bcc, 1123.52660055], [bcc, 1074.29037873], [bcc, 1021.4039286], [bcc, 972.134218449], [bcc, 922.903578019], [bcc, 871.805963016], [mt, 1517.188975], [mt, 1565.2554047], [mt, 1679.06205932], [mt, 920.]] # The last temperature is tweaked manually

compositions=[0.511190023891, 0.511202260941, 0.510985839986, 0.511384301614, 0.511472874541, 0.51148616048, 0.511421012761, 0.511433599441, 0.511829264029, 0.512148243109, 0.510939, 0.511131, 0.511467, 0.511611, 0.511948, 0.512672, 0.512865, 0.513591, 0.511467, 0.511755, 0.512141, 0.51243, 0.512768, 0.513349, 0.513591, 0.514222, 0.539080473166, 0.54144548686, 0.544867315425, 0.515879] # The last composition is that of wus with coexisting iron and magnetite
compositions=np.array([1.0-x_to_y(compositions[i]) for i in range(len(compositions))])


guesses=[wus.endmembers[0][0].params['H_0'], wus.endmembers[1][0].params['H_0'], 0.2, -50.e3]
popt, pcov = optimize.curve_fit(fit_params(wus), minerals_temperatures, compositions, guesses)

print popt


'''
Plot Gibbs at 900 K
'''
comp = np.linspace(0.0, 1.00, 101)
wus_entropies = np.empty_like(comp)
wus_gibbs = np.empty_like(comp)
wus_gibbs_SG1996 = np.empty_like(comp)
T=900.
O2.set_state( 1e5, T )
for i,c in enumerate(comp):
        molar_fractions=[1.0-c, c]
        wus.set_composition( np.array(molar_fractions) )
        wus.set_state( 1e5, T )
        wus_entropies[i] = wus.solution_model._configurational_entropy( molar_fractions ) - wus.solution_model._endmember_configurational_entropy_contribution( molar_fractions )
        wus_gibbs[i] = wus.gibbs
        x=f_to_x(c)
        wus_gibbs_SG1996[i] = (Gwus(x,T) + (1.-x)*gibbs_bcc_1bar(T) + x*O2.gibbs*0.5) / x

plt.plot( comp, wus_gibbs/1000., 'r--', linewidth=3., label='This study')
plt.plot( comp, wus_gibbs_SG1996/1000., 'b--', linewidth=3., label='SG1996')
plt.xlim(0.0,1.0)
plt.title(T)
plt.ylabel("Gibbs free energy of wuestite (kJ/mol)")
plt.xlabel("Fraction ferric iron")
plt.legend(loc='lower right')
plt.show()

'''
Plot phase diagram
'''

P=1.e5

T_eqm=optimize.fsolve(iron_wus_mt_eqm, 800., args=(P))[0]
comp_eqm=f_to_y(optimize.fsolve(bcc_eqm, 0.16, args=(P, T_eqm))[0])

temperatures=np.linspace(T_eqm-50.,1700.,101)
bcc_wus_comp=np.empty_like(temperatures)
fcc_wus_comp=np.empty_like(temperatures)
mt_wus_comp=np.empty_like(temperatures)

for idx, T in enumerate(temperatures):
    bcc_wus_comp[idx]=1.0-f_to_y(optimize.fsolve(bcc_eqm, 0.16, args=(P, T))[0])
    fcc_wus_comp[idx]=1.0-f_to_y(optimize.fsolve(wus_eqm(fcc), 0.16, args=(P, T))[0])
    mt_wus_comp[idx]=1.0-f_to_y(optimize.fsolve(wus_eqm(mt), 0.16, args=(P, T))[0])


plt.plot( [0., 0.25], [T_eqm, T_eqm], 'r-', linewidth=3., label='iron-wus-mt')
plt.plot( bcc_wus_comp, temperatures, 'r-', linewidth=3., label='bcc-wus')
plt.plot( fcc_wus_comp, temperatures, 'r--', linewidth=3., label='fcc-wus')
plt.plot( mt_wus_comp, temperatures, 'r-', linewidth=3., label='wus-mt')


for filename in ['Lykasov_Fe-FeO.dat', 'Vallet_FeOW1-Fe3O4.dat', 'Darken_FeOW1-Fe3O4.dat', 'Asao_et_al_1970.dat', 'Barbi_1964.dat', 'Barbera_et_al_1980.dat']:
    f=open(filename, 'r')
    datastream = f.read()
    f.close()
    datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
    phase_boundaries=np.array(datalines, np.float32).T
    phase_boundaries[0]=[1.0-x_to_y(phase_boundaries[0][i]) for i in range(len(phase_boundaries[0]))]
    plt.plot( phase_boundaries[0], phase_boundaries[1], 'o', linestyle='none', label=filename)

plt.xlim(0.,0.30)
plt.ylim(700.,1700.)
plt.title('Equilibrium at %5.0f K, Fe%6.3f O'%(T_eqm, comp_eqm))
plt.ylabel("Temperature (K)")
plt.xlabel("(1-y) in FeyO")
plt.legend(loc='lower right')
plt.show()

'''
Make a gibbs free energy plot of different minerals and models for wustite
'''

temperature = np.linspace(300., 900., 101)
wus_gibbs_HP = np.empty_like(comp)
mt_gibbs_HP = np.empty_like(comp)
hem_gibbs_HP = np.empty_like(comp)
wus_gibbs_SLB = np.empty_like(comp)
wus_gibbs_model = np.empty_like(comp)
wus_gibbs_model_2 = np.empty_like(comp)
wus_gibbs_model_3 = np.empty_like(comp)
wus_gibbs_model_4 = np.empty_like(comp)
wus_gibbs_SG1996 = np.empty_like(comp)
wus_gibbs_SG1996_2 = np.empty_like(comp)
wus_gibbs_SG1996_3 = np.empty_like(comp)
wus_gibbs_SG1996_4 = np.empty_like(comp)
for i,T in enumerate(temperature):
    O2.set_state( 1e5, T )

    fper.set_state( 1e5, T )
    wus_SLB.set_state( 1e5, T )

    mt.set_state( 1e5, T )
    hem.set_state( 1e5, T )
    mt_gibbs_HP[i] = mt.gibbs
    hem_gibbs_HP[i] = hem.gibbs

    wus_gibbs_HP[i] = fper.gibbs
    wus_gibbs_SLB[i] = wus_SLB.gibbs

    wus.set_composition( np.array([1.0, 0.0]) )
    wus.set_state( 1e5, T )
    wus_gibbs_model[i] = wus.gibbs
    
    wus.set_composition( np.array([0.0, 1.0]) )
    wus.set_state( 1e5, T )
    wus_gibbs_model_2[i] = wus.gibbs
 
    f=x_to_f(y_to_x(0.9))
    wus.set_composition( np.array([1.-f, f]) )
    wus.set_state( 1e5, T )
    wus_gibbs_model_3[i] = wus.gibbs
  
    f=x_to_f(y_to_x(0.75))
    wus.set_composition( np.array([1.-f, f]) )
    wus.set_state( 1e5, T )
    wus_gibbs_model_4[i] = wus.gibbs

    x=0.5
    wus_gibbs_SG1996[i] = (Gwus(x,T) + (1.-x)*gibbs_bcc_1bar(T) + x*O2.gibbs*0.5) / x

    x=0.6
    wus_gibbs_SG1996_2[i] = (Gwus(x,T) + (1.-x)*gibbs_bcc_1bar(T) + x*O2.gibbs*0.5) / x
   
    x=y_to_x(0.90)
    wus_gibbs_SG1996_3[i] = (Gwus(x,T) + (1.-x)*gibbs_bcc_1bar(T) + x*O2.gibbs*0.5) / x

    x=y_to_x(0.75)
    wus_gibbs_SG1996_4[i] = (Gwus(x,T) + (1.-x)*gibbs_bcc_1bar(T) + x*O2.gibbs*0.5) / x




plt.subplot(2,2,1)
plt.plot( temperature, wus_gibbs_HP/1000., 'g-', linewidth=3., label='wus HP2011')
plt.plot( temperature, wus_gibbs_SLB/1000., 'g--', linewidth=3., label='SLB2011')
plt.plot( temperature, wus_gibbs_model/1000., 'r--', linewidth=3., label='Ferrous, This study')
plt.plot( temperature, wus_gibbs_SG1996/1000., 'b--', linewidth=3., label='Ferrous, SG1996')
plt.xlim(300.0,900.0)
plt.title("FeO")
plt.ylabel("Gibbs free energy (kJ/mol)")
plt.xlabel("Temperature (K)")
plt.legend(loc='upper right')

plt.subplot(2,2,2)
plt.plot( temperature, wus_gibbs_model_3/1000., 'r--', linewidth=3., label='Fe0.9O, This study')
plt.plot( temperature, wus_gibbs_SG1996_3/1000., 'b--', linewidth=3., label='Fe0.9O, SG1996')
plt.xlim(300.0,900.0)
plt.title("Fe0.9O")
plt.ylabel("Gibbs free energy (kJ/mol)")
plt.xlabel("Temperature (K)")
plt.legend(loc='upper right')

plt.subplot(2,2,3)
plt.plot( temperature, wus_gibbs_model_4/1000., 'r--', linewidth=3., label='Fe0.75O, This study')
plt.plot( temperature, mt_gibbs_HP/1000./4., 'g--', linewidth=3., label='mt HP2011')
plt.plot( temperature, wus_gibbs_SG1996_4/1000., 'b--', linewidth=3., label='Fe0.75O, SG1996')
plt.xlim(300.0,900.0)
plt.title("\'Magnetite\'")
plt.ylabel("Gibbs free energy (kJ/mol)")
plt.xlabel("Temperature (K)")
plt.legend(loc='upper right')

plt.subplot(2,2,4)
plt.plot( temperature, wus_gibbs_model_2/1000., 'r--', linewidth=3., label='Ferric, This study')
plt.plot( temperature, hem_gibbs_HP/1000./3., 'g--', linewidth=3., label='hem HP2011')
plt.plot( temperature, wus_gibbs_SG1996_2/1000., 'b--', linewidth=3., label='Ferric, SG1996')
plt.xlim(300.0,900.0)
plt.title("\'Hematite\'")
plt.ylabel("Gibbs free energy (kJ/mol)")
plt.xlabel("Temperature (K)")
plt.legend(loc='upper right')

plt.show()


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

wus.endmembers[0][0].params['V_0']=volume_FeO
wus.endmembers[1][0].params['V_0']=volume_defect

# Plot data and fit
plt.plot( compositions, volumes, linewidth='2', label='Haas and Hemingway (1992)')
plt.plot( [0., 1./3.], [linear_trend(0., v_0, v_grad), linear_trend(1./3., v_0, v_grad)], linewidth='2', label='This study, linear trend')
plt.plot( yusn, vusn, marker='o', linestyle='none', label='S+G compilation, ultrasound')
plt.plot( yxrd, vxrd, marker='o', linestyle='none', label='S+G compilation, XRD')
plt.plot( yshk, vshk, marker='o', linestyle='none', label='S+G compilation, shock')
plt.plot( ytheory, vtheory, marker='o', linestyle='none', label='Theory; Jette and Foote, 1933')
plt.plot( yfit, vfit, marker='x', linestyle='none', label='fitted volumes')

plt.xlim(0., 1./3.)
plt.legend(loc="upper right")
plt.ylabel("V (m^3)")
plt.xlabel("y in Fe(1-y)O")
plt.show()


tools.print_mineral_class(wus.endmembers[0][0], 'wustite')
tools.print_mineral_class(wus.endmembers[1][0], 'defect_wustite')
