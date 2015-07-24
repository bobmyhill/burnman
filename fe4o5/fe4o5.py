import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

# Benchmarks for the solid solution class
import burnman
from burnman import minerals
from burnman.mineral import Mineral
from burnman.processchemistry import *
from burnman.chemicalpotentials import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scipy.optimize as optimize

atomic_masses=read_masses()

'''
Excess properties
'''

# High magnetite
class high_mt (Mineral):
    def __init__(self):
       formula='Fe3.0O4.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'High pressure magnetite',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -1057460.0 ,
            'S_0': 172.4 ,
            'V_0': 4.189e-05 ,
            'Cp': [262.5, -0.007205, -1926200.0, -1655.7] ,
            'a_0': 3.59e-05 ,
            'K_0': 2.020e+11 ,
            'Kprime_0': 4.00 ,
            'Kdprime_0': 0.0e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)

# Fe4O5
class Fe4O5 (Mineral):
    def __init__(self):
       formula='Fe4.0O5.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'Fe4O5',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -1357000.0 ,
            'S_0': 218. ,
            'V_0': 5.376e-05 ,
            'Cp': [306.9, 0.001075, -3140400.0, -1470.5] ,
            'a_0': 2.36e-05 ,
            'K_0': 1.857e+11 ,
            'Kprime_0': 4.00 ,
            'Kdprime_0': -2.154e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)

class periclase (Mineral):
    def __init__(self):
       formula='Mg1.0O1.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'periclase',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -601530.0 ,
            'S_0': 26.5 ,
            'V_0': 1.125e-05 ,
            'Cp': [60.5, 0.000362, -535800.0, -299.2] ,
            'a_0': 3.11e-05 ,
            'K_0': 1.616e+11 ,
            'Kprime_0': 3.95 ,
            'Kdprime_0': -2.4e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)
       
class wustite (Mineral):
    def __init__(self):
       formula='Fe1.0O1.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'fper',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -264800. , # -264353.6
            'S_0': 59.,
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
       formula='Fe2/3O1.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'fper',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -271800. ,
            'S_0': 28.,
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
class ferropericlase(burnman.SolidSolution):
    def __init__(self):
        # Name
        self.name='non-stoichiometric wuestite, ferric and ferrous iron treated as separate atoms in Sconf'

        base_material = [[periclase(), '[Mg]O'],[wustite(), '[Fe]O'],[defect_wustite(), '[Fef1/2Vc1/2]Fef1/6O']]

        # Interaction parameters
        enthalpy_interaction=[[11.0e3, 11.0e3], [2.0e3]]

        burnman.SolidSolution.__init__(self, base_material, \
                          burnman.solutionmodel.SymmetricRegularSolution(base_material, enthalpy_interaction) )


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

wus=ferropericlase()

iron=minerals.HP_2011_ds62.iron()
fper=minerals.HP_2011_ds62.fper()
mt=minerals.HP_2011_ds62.mt()
hem=minerals.HP_2011_ds62.hem()
high_mt=high_mt()

fa=minerals.HP_2011_ds62.fa()
frw=minerals.HP_2011_ds62.frw()

q=minerals.HP_2011_ds62.q()
coe=minerals.HP_2011_ds62.coe()
stv=minerals.HP_2011_ds62.stv()

Re=minerals.Metal_Metal_oxides.Re()
ReO2=minerals.Metal_Metal_oxides.ReO2()

Mo=minerals.Metal_Metal_oxides.Mo()
MoO2=minerals.Metal_Metal_oxides.MoO2()

O2=minerals.HP_2011_fluids.O2()

wus_SLB=minerals.SLB_2011.wuestite()

iron.set_state( 1e5, 298.15 )
O2.set_state( 1e5, 298.15 )


wus.set_composition( np.array([0.0,1.0,0.0]) )
wus.set_state( 1e5, 298.15 )
x=0.5
S_ss=np.array([[0., 0., 0.],[0., 0., 0.]])
y_Sf=np.array([[0.947, 57.4365],[0.9379, 57.0905],[0.9254, 56.2217]])

for i, ysf in enumerate(y_Sf):
    f=x_to_f(y_to_x(ysf[0]))
    molar_fractions=[0.0, 1.0-f, f]
    S_ss[0][i]=f
    S_ss[1][i]=ysf[1]-(wus.solution_model._configurational_entropy( molar_fractions ) - wus.solution_model._endmember_configurational_entropy_contribution( molar_fractions ))



def S_trend(x, a, b):
    return a + b*x

S_tr, S_tr_var = optimize.curve_fit(S_trend, S_ss[0], S_ss[1])

# Observed configurational entropy of solution
'''
plt.plot( S_ss[0], S_ss[1], 'o', linewidth=3.)
plt.plot( np.linspace(0.,1.,101.), S_trend(np.linspace(0.,1.,101.), S_tr[0], S_tr[1]), 'r--', linewidth=3.)
plt.xlim(0.0,1.0)
plt.ylim(0.,70.0)
plt.ylabel("Configurational entropy of solution (J/K/mol)")
plt.xlabel("Fraction ferric iron")
plt.show()
'''

comp = np.linspace(0.0, 1.00, 101)
wus_entropies = np.empty_like(comp)
wus_gibbs = np.empty_like(comp)
wus_gibbs_SG1996 = np.empty_like(comp)
T=900.
iron.set_state( 1e5, T )
O2.set_state( 1e5, T )
for i,c in enumerate(comp):
        molar_fractions=[0.0, 1.0-c, c]
        wus.set_composition( np.array(molar_fractions) )
        wus.set_state( 1e5, T )
        wus_entropies[i] = wus.solution_model._configurational_entropy( molar_fractions ) - wus.solution_model._endmember_configurational_entropy_contribution( molar_fractions )
        wus_gibbs[i] = wus.gibbs
        x=f_to_x(c)
        wus_gibbs_SG1996[i] = (Gwus(x,T) + (1.-x)*iron.gibbs + x*O2.gibbs*0.5) / x


# Model configurational entropy of solution
'''
plt.plot( comp, wus_entropies, 'r--', linewidth=3.)
plt.xlim(0.0,1.0)
plt.ylabel("Configurational entropy of solution (J/K/mol)")
plt.xlabel("Fraction ferric iron")
plt.show()
'''

# Model Gibbs free energy of solution vs. SG1996
'''
plt.plot( comp, wus_gibbs/1000., 'r--', linewidth=3., label='This study')
plt.plot( comp, wus_gibbs_SG1996/1000., 'b--', linewidth=3., label='SG1996')
plt.xlim(0.0,1.0)
plt.title(T)
plt.ylabel("Gibbs free energy of wuestite (kJ/mol)")
plt.xlabel("Fraction ferric iron")
plt.legend(loc='lower right')
plt.show()
'''


# Figure 1 from Stolen and Gronvold, 1996
'''
comp = np.linspace(0.5, 0.54, 101)
wus_gibbs_SG1996 = np.empty_like(comp)
for i,c in enumerate(comp):
    wus_gibbs_SG1996[i] = Gwus(c,900.)/1000.

plt.plot( comp, wus_gibbs_SG1996, 'b--', linewidth=3.)
plt.xlim(0.5,0.54)
plt.ylabel("Gibbs free energy of formation (kJ/mol)")
plt.xlabel("x in Fe1-xOx")
plt.show()
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
    iron.set_state( 1e5, T )
    O2.set_state( 1e5, T )

    fper.set_state( 1e5, T )
    wus_SLB.set_state( 1e5, T )

    mt.set_state( 1e5, T )
    hem.set_state( 1e5, T )
    mt_gibbs_HP[i] = mt.gibbs
    hem_gibbs_HP[i] = hem.gibbs

    wus_gibbs_HP[i] = fper.gibbs
    wus_gibbs_SLB[i] = wus_SLB.gibbs

    wus.set_composition( np.array([0.0, 1.0, 0.0]) )
    wus.set_state( 1e5, T )
    wus_gibbs_model[i] = wus.gibbs
    
    wus.set_composition( np.array([0.0, 0.0, 1.0]) )
    wus.set_state( 1e5, T )
    wus_gibbs_model_2[i] = wus.gibbs
 
    f=x_to_f(y_to_x(0.9))
    wus.set_composition( np.array([0.0, 1.-f, f]) )
    wus.set_state( 1e5, T )
    wus_gibbs_model_3[i] = wus.gibbs
  
    f=x_to_f(y_to_x(0.75))
    wus.set_composition( np.array([0.0, 1.-f, f]) )
    wus.set_state( 1e5, T )
    wus_gibbs_model_4[i] = wus.gibbs

    x=0.5
    wus_gibbs_SG1996[i] = (Gwus(x,T) + (1.-x)*iron.gibbs + x*O2.gibbs*0.5) / x

    x=0.6
    wus_gibbs_SG1996_2[i] = (Gwus(x,T) + (1.-x)*iron.gibbs + x*O2.gibbs*0.5) / x
   
    x=y_to_x(0.90)
    wus_gibbs_SG1996_3[i] = (Gwus(x,T) + (1.-x)*iron.gibbs + x*O2.gibbs*0.5) / x

    x=y_to_x(0.75)
    wus_gibbs_SG1996_4[i] = (Gwus(x,T) + (1.-x)*iron.gibbs + x*O2.gibbs*0.5) / x




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

'''
def Cp(x, a, b, c, d):
    return a + b*x + c/(x*x) + d/np.sqrt(x) 

f=x_to_f(y_to_x(0.9))
Cp_ss=[(1.-f)*wustite().params['Cp'][i] + f*defect_wustite().params['Cp'][i] for i in range(4)]

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

# Observed and modelled heat capacities of wuestite
'''
plt.xlim(250.0,1300.0)
plt.ylim(5.0,8.0)
plt.title("Comparison of Cp wuestite")
plt.ylabel("Cp/R wuestite")
plt.xlabel("Temperature (K)")
plt.legend(loc='lower right')
plt.show()
'''

# Find the chemical potentials of Fe and Fe3O4 for different compositions of wuestite at each pressure. 
def iron_eqm(comp, P, T, XMgO):
    c=comp[0]
    wus.set_composition([XMgO, (1.0-c)*(1.-XMgO),c*(1.-XMgO)])
    wus.set_state(P,T)
    mu_iron=chemical_potentials([wus],[dictionarize_formula('Fe')])[0]
    return mu_iron-iron.calcgibbs(P,T)

def mt_eqm(comp, P, T, XMgO):
    c=comp[0]
    wus.set_composition([XMgO, (1.0-c)*(1.-XMgO), c*(1.-XMgO)])
    wus.set_state(P,T)
    mu_mt=chemical_potentials([wus],[dictionarize_formula('Fe3O4')])[0]
    return mu_mt-mt.calcgibbs(P,T)



P=1.e5
XMgO=0.0

def iron_wus_mt_eqm(arg, P):
    XMgO=0.0
    c=arg[0]
    T=arg[1]
    wus.set_composition([XMgO, (1.0-c)*(1.-XMgO), c*(1.-XMgO)])
    wus.set_state(P,T)
    mu_iron=chemical_potentials([wus],[dictionarize_formula('Fe')])[0]
    mu_mt=chemical_potentials([wus],[dictionarize_formula('Fe3O4')])[0]
    return [mu_iron-iron.calcgibbs(P,T),mu_mt-mt.calcgibbs(P,T)]


T_eqm=optimize.fsolve(iron_wus_mt_eqm, [0.5,800.], args=(P))[1]
comp_eqm=f_to_y(optimize.fsolve(iron_eqm, 0.16, args=(P, T_eqm, XMgO))[0])

temperatures=np.linspace(T_eqm-50.,1700.,101)
iron_wus_comp=np.empty_like(temperatures)
mt_wus_comp=np.empty_like(temperatures)

for idx, T in enumerate(temperatures):
    iron_wus_comp[idx]=1.0-f_to_y(optimize.fsolve(iron_eqm, 0.16, args=(P, T, XMgO))[0])
    mt_wus_comp[idx]=1.0-f_to_y(optimize.fsolve(mt_eqm, 0.16, args=(P, T, XMgO))[0])

crosses=np.array([[1222.07160224,1172.8465432,1121.78241653,1072.56293888,1021.49881221,972.212357893,922.987298852,871.878521071,871.325963574,923.883111764,972.862589695,1023.7090422,1072.76107818,1121.84102112,1172.74328751,1223.66787946],[0.0428177641654,0.0435528330781,0.0448392036753,0.0453905053599,0.0466768759571,0.0494333843798,0.0501684532925,0.0529249617152,0.0711179173047,0.08067381317,0.0880245022971,0.0939050535988,0.0988667687596,0.102909647779,0.106952526799,0.110260336907]])

invt=np.array([[1223.83253043,1172.77398515,1123.52660055,1074.29037873,1021.4039286,972.134218449,922.903578019,871.805963016,842.459020396,874.992936055,923.977995374,973.052356916,1023.88206525,1074.78433165,1123.87543735,1172.91631056,1225.64090042],[0.0448392036753,0.0459418070444,0.0474119448698,0.0485145482389,0.0498009188361,0.0520061255743,0.0529249617152,0.0553139356815,0.0615620214395,0.070382848392,0.077549770291,0.0817764165391,0.0882082695253,0.0922511485452,0.0959264931087,0.101255742726,0.105298621746]])

plusses=np.array([[1172.38886932,1073.97223957,971.888637339,862.459927372,918.54451332,962.006788364,1034.46995992,1172.33584612],[0.0586217457887,0.058989280245,0.0600918836141,0.0630321592649,0.0764471669219,0.0854517611026,0.0996018376723,0.120367534456]])

# Iron-Wuestite-Magnetite phase diagram
'''
f=open('Fe-O_boundaries_int.dat', 'r')
datastream = f.read()  # We need to re-open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
phase_boundaries=np.array(datalines, np.float32).T
phase_boundaries[0]=[1.0-x_to_y(phase_boundaries[0][i]) for i in range(len(phase_boundaries[0]))]

plt.plot( [0., 0.25], [T_eqm, T_eqm], 'r-', linewidth=3., label='iron-wus-mt')
plt.plot( iron_wus_comp, temperatures, 'r-', linewidth=3., label='iron-wus')
plt.plot( mt_wus_comp, temperatures, 'r-', linewidth=3., label='wus-mt')


plt.plot( phase_boundaries[0], phase_boundaries[1], '--', linewidth=3., label='phase boundaries')

plt.plot( crosses[1], crosses[0], 'o', linewidth=3., label='Barbera et al., 1980')
plt.plot( plusses[1], plusses[0], 'o', linewidth=3., label='Asao et al., 1970')
plt.plot( invt[1], invt[0], 'o', linewidth=3., label='Barbi, 1964')

plt.xlim(0.,0.30)
plt.ylim(700.,1700.)
plt.title('Equilibrium at %5.0f K, Fe%6.3f O'%(T_eqm, comp_eqm))
plt.ylabel("Temperature (K)")
plt.xlabel("(1-y) in FeyO")
plt.legend(loc='lower right')
plt.show()
'''

# Equilibrium with metallic iron
'''
P=20.e9 # Pa
T=1300 # K

XMgOs=np.linspace(0.,0.999,101)

pressures=np.linspace(1.e5, 25.e9, 6)
iron_wus_comp=np.empty_like([XMgOs]*len(pressures))

# Loop over Mg content
for pidx, P in enumerate(pressures):
    for idx,XMgO in enumerate(XMgOs):
        # Find equilibrium with metallic iron
        Xdefect=optimize.fsolve(iron_eqm, 0.01, args=(P, T, XMgO))[0]
        iron_wus_comp[pidx][idx]=(2./3.*Xdefect)/(1.-Xdefect/3.)

for pidx, P in enumerate(pressures):
    plt.plot( XMgOs, iron_wus_comp[pidx], '-', linewidth=3., label='%5.0f GPa'%(P/1e9))
plt.xlim(0.0,1.0)
plt.ylim(0.0,0.15)
plt.title('Equilibrium at %5.0f GPa, %5.0f K'%(P/1.e9, T))
plt.ylabel("Fe3+/sum(Fe)")
plt.xlabel("X(MgO)")
plt.legend(loc='lower right')
plt.show()
'''


fe4o5=Fe4O5()
fe4o5.set_method('hp_tmt')

P=11.5e9
T=1366.
V=345.753 # Angstroms

Nb=6.022e23
Z=4
A3_to_m3=1e-30

V=V*A3_to_m3*Nb/Z

fe4o5.set_state(P,T)
print fe4o5.V, V


# 2Fe3O4 -> Fe4O5 + Fe2O3
def eqm_mt_breakdown(P, T):
    return 2.*mt.calcgibbs(P,T) - fe4o5.calcgibbs(P,T) - hem.calcgibbs(P,T)

# Fe4O5 -> wustite + mt
def fe4o5_eqm(comp, P, T, XMgO):
    c=comp[0]
    wus.set_composition([XMgO, (1.0-c)*(1.-XMgO), c*(1.-XMgO)])
    wus.set_state(P,T)
    mu_fe4o5=chemical_potentials([wus],[dictionarize_formula('Fe4O5')])[0]
    return  mu_fe4o5-fe4o5.calcgibbs(P,T)

def wus_fe4o5_mt_eqm(arg, T):
    XMgO=0.0
    c=arg[0]
    P=arg[1]
    wus.set_composition([XMgO, (1.0-c)*(1.-XMgO), c*(1.-XMgO)])
    wus.set_state(P,T)
    mu_fe4o5=chemical_potentials([wus],[dictionarize_formula('Fe4O5')])[0]
    mu_mt=chemical_potentials([wus],[dictionarize_formula('Fe3O4')])[0]
    return [mu_fe4o5-fe4o5.calcgibbs(P,T),mu_mt-mt.calcgibbs(P,T)]

def hem_fe4o5_rhenium_eqm(P,T):
    return 4.*hem.calcgibbs(P,T) + Re.calcgibbs(P,T) - 2*fe4o5.calcgibbs(P,T) - ReO2.calcgibbs(P,T)

def mt_fe4o5_rhenium_eqm(P,T):
    return 8.*mt.calcgibbs(P,T) + Re.calcgibbs(P,T) - 6*fe4o5.calcgibbs(P,T) - ReO2.calcgibbs(P,T)

def mt_high_mt_eqm(P,T):
    return mt.calcgibbs(P,T) - high_mt.calcgibbs(P,T)

def hem_mt_fe4o5_rhenium_univariant(T):
    return optimize.fsolve(mt_fe4o5_rhenium_eqm, 10.e9, args=(T))[0] - optimize.fsolve(hem_fe4o5_rhenium_eqm, 10.e9, args=(T))[0]

univariant=optimize.fsolve(hem_mt_fe4o5_rhenium_univariant, 1100., args=())[0]

temperatures=np.linspace(873.15, 1773.15, 91)
low_temperatures=np.linspace(873.15, univariant, 91)
high_temperatures=np.linspace(univariant, 1773.15, 91)

fe4o5_breakdown_pressures=np.empty_like(temperatures)
mt_high_mt_pressures=np.empty_like(temperatures)
for idx, T in enumerate(temperatures):
    fe4o5_breakdown_pressures[idx]=optimize.fsolve(wus_fe4o5_mt_eqm, [0.5,10.e9], args=(T))[1]
    mt_high_mt_pressures[idx]=optimize.fsolve(mt_high_mt_eqm, 10.e9, args=(T))[0]

stable_mt_breakdown_pressures=np.empty_like(low_temperatures)
hem_re_pressures=np.empty_like(low_temperatures)
for idx, T in enumerate(low_temperatures):
    stable_mt_breakdown_pressures[idx]=optimize.fsolve(eqm_mt_breakdown, 10.e9, args=(T))[0]
    hem_re_pressures[idx]=optimize.fsolve(hem_fe4o5_rhenium_eqm, 10.e9, args=(T))[0]

metastable_mt_breakdown_pressures=np.empty_like(high_temperatures)
mt_re_pressures=np.empty_like(high_temperatures)
for idx, T in enumerate(high_temperatures):
    metastable_mt_breakdown_pressures[idx]=optimize.fsolve(eqm_mt_breakdown, 10.e9, args=(T))[0]
    mt_re_pressures[idx]=optimize.fsolve(mt_fe4o5_rhenium_eqm, 10.e9, args=(T))[0]


# Experimental phase diagram
reaction_lines=[[low_temperatures, stable_mt_breakdown_pressures], [high_temperatures, metastable_mt_breakdown_pressures],[temperatures, fe4o5_breakdown_pressures],[temperatures, mt_high_mt_pressures], [low_temperatures, hem_re_pressures], [high_temperatures, mt_re_pressures] ]
f = open('TP-pseudosection.dat', 'w')

for rxn, reaction in enumerate(reaction_lines):
    if rxn == 1 or rxn==3:
        f.write('>> -Wthin,black,- \n')
    else:
        f.write('>> -Wthin,black \n')
    for i, T in enumerate(reaction[0]):
        f.write(str(T-273.15))
        f.write(' ')
        f.write(str(reaction[1][i]/1.e9))
        f.write('\n')
f.close()

print 'TP-pseudosection.dat (over)written'

plt.plot( low_temperatures-273.15, stable_mt_breakdown_pressures/1.e9, '-', linewidth=3., label='Fe3O4 breakdown')
plt.plot( high_temperatures-273.15, metastable_mt_breakdown_pressures/1.e9, '--', linewidth=3., label='Fe3O4 breakdown')

plt.plot( temperatures-273.15, fe4o5_breakdown_pressures/1.e9, '-', linewidth=3., label='Fe4O5 breakdown')
plt.plot( temperatures-273.15, mt_high_mt_pressures/1.e9, '--', linewidth=3., label='Fe3O4--high-Fe3O4')

plt.plot( low_temperatures-273.15, hem_re_pressures/1.e9, '-', linewidth=3., label='4hem + Re -> 2Fe4O5 + ReO2')
plt.plot( high_temperatures-273.15, mt_re_pressures/1.e9, '-', linewidth=3., label='8mt + Re -> 6Fe4O5 + ReO2')


plt.title('')
plt.ylabel("Pressure (GPa)")
plt.xlabel("Temperature (C)")
plt.legend(loc='lower right')
plt.show()

# Fe-O oxygen fugacity diagram
T=1473.15
O2.set_state(1e5,T)
XMgO=0.0

pressures=np.linspace(1.e5, 24.e9, 101)
fe4o5_hem_fO2=np.empty_like(pressures)
fe4o5_mt_fO2=np.empty_like(pressures)
mt_hem_fO2=np.empty_like(pressures)
iron_wus_fO2=np.empty_like(pressures)
wus_mt_fO2=np.empty_like(pressures)
wus_fe4o5_fO2=np.empty_like(pressures)

fmq_fO2=np.empty_like(pressures)
fmc_fO2=np.empty_like(pressures)
rmc_fO2=np.empty_like(pressures)
rms_fO2=np.empty_like(pressures)
rfs_fO2=np.empty_like(pressures)

Re_fO2=np.empty_like(pressures)
Mo_fO2=np.empty_like(pressures)

for i, P in enumerate(pressures):
    iron.set_state(P,T)
    fe4o5.set_state(P,T)
    hem.set_state(P,T)
    mt.set_state(P,T)

    fa.set_state(P,T)
    frw.set_state(P,T)

    q.set_state(P,T)
    coe.set_state(P,T)
    stv.set_state(P,T)
 
    Re.set_state(P,T)
    ReO2.set_state(P,T)
    Mo.set_state(P,T)
    MoO2.set_state(P,T)

    c=optimize.fsolve(mt_eqm, 0.16, args=(P, T, XMgO))[0]
    wus.set_composition([XMgO, (1.0-c)*(1.-XMgO), c*(1.-XMgO)])
    wus.set_state(P,T)
    assemblage=[wus, mt]
    wus_mt_fO2[i]=np.log10(fugacity(O2, assemblage))

    c=optimize.fsolve(fe4o5_eqm, 0.16, args=(P, T, XMgO))[0]
    wus.set_composition([XMgO, (1.0-c)*(1.-XMgO), c*(1.-XMgO)])
    wus.set_state(P,T)
    assemblage=[wus, fe4o5]
    wus_fe4o5_fO2[i]=np.log10(fugacity(O2, assemblage))

    c=optimize.fsolve(iron_eqm, 0.16, args=(P, T, XMgO))[0]
    wus.set_composition([XMgO, (1.0-c)*(1.-XMgO), c*(1.-XMgO)])
    wus.set_state(P,T)
    assemblage=[iron, wus]
    iron_wus_fO2[i]=np.log10(fugacity(O2, assemblage))

    assemblage=[fe4o5, mt]
    fe4o5_mt_fO2[i]=np.log10(fugacity(O2, assemblage))

    assemblage=[fe4o5, hem]
    fe4o5_hem_fO2[i]=np.log10(fugacity(O2, assemblage))

    assemblage=[mt, hem]
    mt_hem_fO2[i]=np.log10(fugacity(O2, assemblage))


    # FMQ and analogues
    assemblage=[fa, mt, q]
    fmq_fO2[i]=np.log10(fugacity(O2, assemblage))

    assemblage=[fa, mt, coe]
    fmc_fO2[i]=np.log10(fugacity(O2, assemblage))

    assemblage=[frw, mt, coe]
    rmc_fO2[i]=np.log10(fugacity(O2, assemblage))

    assemblage=[frw, mt, stv]
    rms_fO2[i]=np.log10(fugacity(O2, assemblage))

    assemblage=[frw, fe4o5, stv]
    rfs_fO2[i]=np.log10(fugacity(O2, assemblage))

    assemblage=[Re, ReO2]
    Re_fO2[i]=np.log10(fugacity(O2, assemblage))

    assemblage=[Mo, MoO2]
    Mo_fO2[i]=np.log10(fugacity(O2, assemblage))


plt.plot( pressures/1.e9, iron_wus_fO2, '-', linewidth=3., label='iron-wus')
plt.plot( pressures/1.e9, wus_fe4o5_fO2, '-', linewidth=3., label='wus-Fe4O5')
plt.plot( pressures/1.e9, wus_mt_fO2, '-', linewidth=3., label='wus-mt')


plt.plot( pressures/1.e9, fe4o5_mt_fO2, '-', linewidth=3., label='Fe4O5-mt')
plt.plot( pressures/1.e9, fe4o5_hem_fO2, '-', linewidth=3., label='Fe4O5-hem')
plt.plot( pressures/1.e9, mt_hem_fO2, '-', linewidth=3., label='mt-hem')

plt.title('')
plt.xlabel("Pressure (GPa)")
plt.ylabel("log10(fO2)")
plt.legend(loc='lower right')
plt.show()



plt.plot( pressures/1.e9, fmq_fO2, '-', linewidth=3., label='FMQ')
plt.plot( pressures/1.e9, fmc_fO2, '-', linewidth=3., label='FMC')
plt.plot( pressures/1.e9, rmc_fO2, '-', linewidth=3., label='RMC')
plt.plot( pressures/1.e9, rms_fO2, '-', linewidth=3., label='RMS')
plt.plot( pressures/1.e9, rfs_fO2, '-', linewidth=3., label='RFS')

plt.plot( pressures/1.e9, Re_fO2, '-', linewidth=3., label='Re-ReO2')
plt.plot( pressures/1.e9, Mo_fO2, '-', linewidth=3., label='Mo-MoO2')

plt.title('')
plt.xlabel("Pressure (GPa)")
plt.ylabel("log10(fO2)")
plt.legend(loc='lower right')
plt.show()

columns=[pressures, iron_wus_fO2, wus_fe4o5_fO2, wus_mt_fO2, fe4o5_mt_fO2, fe4o5_hem_fO2, mt_hem_fO2, fmq_fO2, fmc_fO2, rmc_fO2, rms_fO2, rfs_fO2, Re_fO2, Mo_fO2]
f = open('P-fO2.dat', 'w')
f.write('P, iron_wus_fO2, wus_fe4o5_fO2, wus_mt_fO2, fe4o5_mt_fO2, fe4o5_hem_fO2, mt_hem_fO2, fmq_fO2, fmc_fO2, rmc_fO2, rms_fO2, rfs_fO2, Re_fO2, Mo_fO2\n')
for i, P in enumerate(pressures):
    for column in columns:
        f.write(str(column[i]))
        f.write(' ') 
    f.write('\n')
f.close()

print 'P-fO2.dat (over)written'

# (Mg,Fe)2Fe2O5 - ol/wad/rw equilibrium
# Fe4O5
class Fe4O5 (Mineral):
    def __init__(self):
       formula='Fe4.0O5.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'Fe4O5',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -1357000.0 ,
            'S_0': 218. ,
            'V_0': 5.376e-05 ,
            'Cp': [306.9, 0.001075, -3140400.0, -1470.5] ,
            'a_0': 2.36e-05 ,
            'K_0': 1.857e+11 ,
            'Kprime_0': 4.00 ,
            'Kdprime_0': -2.154e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)

'''
class MgFe3O5 (Mineral): # for SLB
    def __init__(self):
       formula='Mg1.0Fe3.0O5.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'MgFe3O5',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -1705000.0 ,
            'S_0': 185.9 ,
            'V_0': 5.333e-05 ,
            'Cp': [323.0, -0.006843, -2462000.0, -1954.9] ,
            'a_0': 2.36e-05 ,
            'K_0': 1.500e+11 ,
            'Kprime_0': 4.00 ,
            'Kdprime_0': -3.080e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
'''

# Best fit for change in Kd with pressure for each mineral
class MgFe3O5 (Mineral):
    def __init__(self):
       formula='Mg1.0Fe3.0O5.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'MgFe3O5',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -1688000.0 ,
            'S_0': 185.9 ,
            'V_0': 5.333e-05 ,
            'Cp': [323.0, -0.006843, -2462000.0, -1954.9] ,
            'a_0': 2.36e-05 ,
            'K_0': 1.400e+11 ,
            'Kprime_0': 4.00 ,
            'Kdprime_0': -3.080e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
       Mineral.__init__(self)

# Best fit for ol and wad (but pressure dependence within minerals ill-fitting
'''
class MgFe3O5 (Mineral):
    def __init__(self):
       formula='Mg1.0Fe3.0O5.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'MgFe3O5',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -1685500.0 ,
            'S_0': 185.9 ,
            'V_0': 5.333e-05 ,
            'Cp': [323.0, -0.006843, -2462000.0, -1954.9] ,
            'a_0': 2.36e-05 ,
            'K_0': 1.200e+11 ,
            'Kprime_0': 4.00 ,
            'Kdprime_0': -3.080e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
'''

class MgFeFe2O5(burnman.SolidSolution):
    def __init__(self):
        # Name
        self.name='(Mg,Fe)2Fe2O5'

        base_material = [[MgFe3O5(), '[Mg]Fe3O5'],[Fe4O5(), '[Fe]Fe3O5']]

        # Interaction parameters
        enthalpy_interaction=[[0.0e3]]

        burnman.SolidSolution.__init__(self, base_material, \
                          burnman.solutionmodel.SymmetricRegularSolution(base_material, enthalpy_interaction) )

class olivine(burnman.SolidSolution):
    def __init__(self):
        # Name
        self.name='olivine'

        base_material = [[minerals.HP_2011_ds62.fo(), '[Mg]2SiO4'],[minerals.HP_2011_ds62.fa(), '[Fe]2SiO4']]

        # Interaction parameters
        enthalpy_interaction=[[9.0e3]]

        burnman.SolidSolution.__init__(self, base_material, \
                          burnman.solutionmodel.SymmetricRegularSolution(base_material, enthalpy_interaction) )

class wadsleyite(burnman.SolidSolution):
    def __init__(self):
        # Name
        self.name='wadsleyite'

        base_material = [[minerals.HP_2011_ds62.mwd(), '[Mg]2SiO4'],[minerals.HP_2011_ds62.fwd(), '[Fe]2SiO4']]

        # Interaction parameters
        enthalpy_interaction=[[13.0e3]]

        burnman.SolidSolution.__init__(self, base_material, \
                          burnman.solutionmodel.SymmetricRegularSolution(base_material, enthalpy_interaction) )

class ringwoodite(burnman.SolidSolution):
    def __init__(self):
        # Name
        self.name='ringwoodite'

        base_material = [[minerals.HP_2011_ds62.mrw(), '[Mg]2SiO4'],[minerals.HP_2011_ds62.frw(), '[Fe]2SiO4']]

        # Interaction parameters
        enthalpy_interaction=[[4.0e3]]

        burnman.SolidSolution.__init__(self, base_material, \
                          burnman.solutionmodel.SymmetricRegularSolution(base_material, enthalpy_interaction) )


# Required constraints
# 1. bulk composition (use composition of ol polymorph)
# 2. 2*MgFe3O5 + Fe2SiO4 <-> 2Fe4O5 + Mg2SiO4

fm45=MgFeFe2O5()
fm45.set_method('hp_tmt')

ol=olivine()
wad=wadsleyite()
rw=ringwoodite()

ol.set_method('hp_tmt')
wad.set_method('hp_tmt')
rw.set_method('hp_tmt')

'''
ol=minerals.SLB_2011.mg_fe_olivine()
wad=minerals.SLB_2011.mg_fe_wadsleyite()
rw=minerals.SLB_2011.mg_fe_ringwoodite()

ol.set_method('slb3')
wad.set_method('slb3')
rw.set_method('slb3')
'''

data=[]
with open('phasePTX.dat','r') as f:
    for expt in f:
        data.append([var for var in expt.split()])

def G_min(arg, ol_polymorph, bulk_XMg, P, T):
    XMg_ol_polymorph=arg[0]
    XMg=(bulk_XMg*2.) - XMg_ol_polymorph

    ol_polymorph.set_composition([XMg_ol_polymorph, 1.0-XMg_ol_polymorph])
    ol_polymorph.set_state(P,T)

    fm45.set_composition([XMg, 1.0-XMg])
    fm45.set_state(P,T)

    return ol_polymorph.gibbs + 2*fm45.gibbs

print ''
for expt in data:
    if len(expt) > 1:
        ol_polymorph_name=expt[2]

        if ol_polymorph_name == 'ol':
            ol_polymorph=ol

        if ol_polymorph_name == 'wad':
            ol_polymorph=wad

        if ol_polymorph_name == 'rw':
            ol_polymorph=rw

        if ol_polymorph_name == 'ol' or ol_polymorph_name == 'wad' or ol_polymorph_name == 'rw':
            P=float(expt[3])*1.e8 # converting from kbar to Pa
            T=float(expt[4])+273.15 # converting from C to K
            XMg_fm45_obs=float(expt[5])
            XMg_ol_polymorph_obs=float(expt[9])
            
            bulk_XMg=(XMg_fm45_obs+XMg_ol_polymorph_obs)/2. # Two moles (Mg,Fe)Fe3O5 to every 1 mole ol_polymorph


            XMg_ol_polymorph_calc=optimize.minimize(G_min, [0.9], method='nelder-mead', args=(ol_polymorph, bulk_XMg, P, T)).x
            if '_' not in expt[0]: 
                print expt[0], ol_polymorph_name, P/1.e9, T-273.15, XMg_ol_polymorph_calc[0], XMg_ol_polymorph_obs, XMg_ol_polymorph_calc[0]-XMg_ol_polymorph_obs
                
                #print ol_polymorph.partial_gibbs[0]-ol_polymorph.partial_gibbs[1], (fm45.partial_gibbs[0]-fm45.partial_gibbs[1])*2.0

'''

# Variables should be entropy, thermal expansion (although we have essentially no constraints on these), enthalpy and bulk modulus

class enstatite (Mineral):
    def __init__(self):
       formula='Mg2.0Si2.0O6.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'en',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -3090220.0 ,
            'S_0': 132.5 ,
            'V_0': 6.262e-05 ,
            'Cp': [356.2, -0.00299, -596900.0, -3185.3] ,
            'a_0': 2.27e-05 ,
            'K_0': 1.059e+11 ,
            'Kprime_0': 8.65 ,
            'Kdprime_0': -8.2e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

class ferrosilite (Mineral):
    def __init__(self):
       formula='Fe2.0Si2.0O6.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'fs',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -2388710.0 ,
            'S_0': 189.9 ,
            'V_0': 6.592e-05 ,
            'Cp': [398.7, -0.006579, 1290100.0, -4058.0] ,
            'a_0': 3.26e-05 ,
            'K_0': 1.01e+11 ,
            'Kprime_0': 4.08 ,
            'Kdprime_0': -4e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

DeltaH_fm=-6950.
class ordered_fm_opx (Mineral):
    def __init__(self):
       formula='Fe2.0Si2.0O6.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'fm',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': (-2388710.0-3090220.0)/2.+DeltaH_fm ,
            'S_0': (132.5+189.9)/2. ,
            'V_0': (6.592e-05+6.262e-05)/2. ,
            'Cp': [(398.7+356.2)/2., (-0.006579-0.00299)/2., (1290100.0-596900.0)/2., (-4058.0-3185.3)/2.] ,
            'a_0': (3.26e-05+2.27e-05)/2. ,
            'K_0': (1.01e+11+1.059e+11)/2. ,
            'Kprime_0': (4.08+8.65)/2. ,
            'Kdprime_0': -1.*(4.08+8.65)/(1.01e+11+1.059e+11) ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}




en=enstatite()
fs=ferrosilite()
fm=ordered_fm_opx()

en.set_method('hp_tmt')
fs.set_method('hp_tmt')
fm.set_method('hp_tmt')

P=0.1e9
T=1273.

en.set_state(P,T)
fs.set_state(P,T)
fm.set_state(P,T)

print en.gibbs, fs.gibbs, fm.gibbs, 0.5*(en.gibbs + fs.gibbs) - fm.gibbs

class orthopyroxene(burnman.SolidSolution):
    def __init__(self):
        # Name
        self.name='Fe-Mg orthopyroxene'

        base_material = [[minerals.HP_2011_ds62.en(), '[Mg][Mg]Si2O6'],[minerals.HP_2011_ds62.fs(), '[Fe][Fe]Si2O6'],[ordered_fm_opx(), '[Mg][Fe]Si2O6']]

        # Interaction parameters
        enthalpy_interaction=[[6.8e3, 4.5e3],[4.5e3]]

        burnman.SolidSolution.__init__(self, base_material, \
                          burnman.solutionmodel.SymmetricRegularSolution(base_material, enthalpy_interaction) )


opx=orthopyroxene()


def opx_composition(arg, X_Mg, P, T):
    Q=arg[0]
    opx.set_composition([X_Mg-0.5*Q, 1-X_Mg-0.5*Q, Q])
    opx.set_state(P,T)
    return opx.gibbs

P=1.e9
T=1473.15
X_Mgs=np.linspace(0.0,1.0,101)
G=np.empty_like(X_Mgs)
Q=np.empty_like(X_Mgs)
for idx, X_Mg in enumerate(X_Mgs):
    optimize.minimize(opx_composition, [-0.01], method='nelder-mead', args=(X_Mg, P, T))
    G[idx]=opx.gibbs
    Q[idx]=opx.molar_fraction[2]

plt.plot( X_Mgs, G, '-', linewidth=3., label='Gibbs opx')

X_Mg=0.5
optimize.minimize(opx_composition, [-0.01], method='nelder-mead', args=(X_Mg, P, T))
print opx.partial_gibbs, opx.gibbs, (opx.partial_gibbs[0]+opx.partial_gibbs[1])/2.

plt.title('')
plt.xlabel("X_Mg opx")
plt.ylabel("G")
plt.legend(loc='lower right')
plt.show()


# (Mg,Fe)2Fe2O5 - ol/wad/rw - opx/hpx diagram
# i.e. 
# 2(Mg,Fe)Fe3O5
# (Mg,Fe)2SiO4
# (Mg,Fe)2Si2O6


#bulk composition = (Mg,Fe)6 Fe6 Si3 O20
# X_Mg_bulk = sum(X_Mg)/3.

def foo_compositions(arg, ol_polymorph, bulk_XMg, P, T):
    # args are X_Mg_Fe4O5, X_Mg_ol_polymorph, Q_opx
    X_Mg_Fe4O5=float(arg[0])
    X_Mg_ol_polymorph=float(arg[1])
    X_Mg_opx=3.*bulk_XMg -1.0*X_Mg_Fe4O5 -1.0*X_Mg_ol_polymorph
    Q_opx=float(arg[2])

    fm45.set_composition([X_Mg_Fe4O5, 1.0-X_Mg_Fe4O5])
    ol_polymorph.set_composition([X_Mg_ol_polymorph, 1.0-X_Mg_ol_polymorph])


    p_en=X_Mg_opx - Q_opx/2.
    p_fm=Q_opx
    p_fs=1. - p_en - p_fm

    opx.set_composition([p_en, p_fs, p_fm])

    fm45.set_state(P,T)
    ol_polymorph.set_state(P,T)
    opx.set_state(P,T)

    return 2.*fm45.gibbs + ol_polymorph.gibbs + opx.gibbs


P=10.5e9
T=1373.
O2=minerals.HP_2011_fluids.O2()
O2.set_method('cork')
O2.set_state(1.e5,T)

Re=minerals.Metal_Metal_oxides.Re()
ReO2=minerals.Metal_Metal_oxides.ReO2()
Re.set_method('hp_tmt')
ReO2.set_method('hp_tmt')

bulk_XMgs=np.linspace(0.01, 0.8, 11)
for bulk_XMg in bulk_XMgs:
    ol_polymorph=ol
    optimize.minimize(foo_compositions, [0.00,0.0,-0.01], method='nelder-mead', args=(ol_polymorph, bulk_XMg, P, T))

    assemblage=[ol, opx, fm45]
    #print ol.partial_gibbs, opx.partial_gibbs
    print ol.molar_fraction[0], opx.molar_fraction[0] + 0.5*opx.molar_fraction[2], fm45.molar_fraction[0], np.log10(fugacity(dictionarize_formula('O2'), O2, assemblage))

    Re.set_state(P,T)
    ReO2.set_state(P,T)
    print np.log10(fugacity(dictionarize_formula('O2'), O2, [Re, ReO2]))
# find oxygen fugacity
 
'''
