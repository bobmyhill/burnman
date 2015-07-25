# Benchmarks for the solid solution class
import burnman
from burnman import minerals
from burnman.mineral import Mineral
from burnman.processchemistry import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scipy.optimize as optimize

atomic_masses=read_masses()

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
            'equation_of_state': 'mtait',
            'H_0': -263800. , # -264353.6
            'S_0': 60.,
            'V_0': 1.2239e-05 , # From Simons (1980)
            'Cp': [5.33343160e+01,   7.79203541e-03,  -3.25553876e+05,  -7.50233740e+01] ,
            'a_0': 3.22e-05 ,
            'K_0': 1.52e+11 ,
            'Kprime_0': 4.9 ,
            'Kdprime_0': -3.2e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

class defect_wustite (Mineral):
    def __init__(self):
       formula='Fe2/3Vc1/3O1.0'
       formula = dictionarize_formula(formula)
       self.params = {
            'name': 'fper',
            'formula': formula,
            'equation_of_state': 'mtait',
            'H_0': -260000. ,
            'S_0': 39.,
            'V_0': 1.10701e-05 , # From Simons (1980)
            'Cp': [-3.64959181e+00,   1.29193873e-02,  -1.07988127e+06,   1.11241795e+03] ,
            'a_0': 3.22e-05 ,
            'K_0': 1.52e+11 ,
            'Kprime_0': 4.9 ,
            'Kdprime_0': -3.2e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

# Configurational entropy
class wuestite_ss(burnman.SolidSolution):
    def __init__(self):
        # Name
        self.name='non-stoichiometric wuestite, ferric and ferrous iron treated as separate atoms in Sconf'

        base_material = [[wustite(), '[Fe]O'],[defect_wustite(), '[Fe2/3Vc1/3]O']]

        # Interaction parameters
        enthalpy_interaction=[[-17.0e3]]

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

wus=wuestite_ss()

iron=minerals.HP_2011_ds62.iron()
fper=minerals.HP_2011_ds62.fper()
mt=minerals.HP_2011_ds62.mt()
hem=minerals.HP_2011_ds62.hem()
O2=minerals.HP_2011_fluids.O2()

wus_SLB=minerals.SLB_2011.wuestite()

iron.set_method('mtait')
wus.set_method('mtait')
fper.set_method('mtait')
mt.set_method('mtait')
hem.set_method('mtait')
O2.set_method('cork')

wus_SLB.set_method('slb3')

iron.set_state( 1e5, 298.15 )
O2.set_state( 1e5, 298.15 )

'''
wus.set_composition( np.array([1.0,0.0]) )
wus.set_state( 1e5, 298.15 )
x=0.5
print (Gwus(x,298.15) + (1.-x)*iron.gibbs + x*O2.gibbs*0.5) / x - wus.gibbs

S_ss=np.array([[0., 0., 0.],[0., 0., 0.]])
y_Sf=np.array([[0.947, 57.4365],[0.9379, 57.0905],[0.9254, 56.2217]])

for i, ysf in enumerate(y_Sf):
    f=x_to_f(y_to_x(ysf[0]))
    S_ss[0][i]=f
    S_ss[1][i]=ysf[1]-(wus.solution_model._configurational_entropy( [1.0-f, f] ) - wus.solution_model._endmember_configurational_entropy_contribution( [1.0-f, f]))
print S_ss


def S_trend(x, a, b):
    return a + b*x

S_tr, S_tr_var = optimize.curve_fit(S_trend, S_ss[0], S_ss[1])

print S_tr[0], S_tr[0]+S_tr[1]

print S_tr

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
        molar_fractions=[1.0-c, c]
        wus.set_composition( np.array(molar_fractions) )
        wus.set_state( 1e5, T )
        wus_entropies[i] = wus.solution_model._configurational_entropy( molar_fractions ) - wus.solution_model._endmember_configurational_entropy_contribution( molar_fractions )
        wus_gibbs[i] = wus.gibbs
        x=f_to_x(c)
        wus_gibbs_SG1996[i] = (Gwus(x,T) + (1.-x)*iron.gibbs + x*O2.gibbs*0.5) / x

'''
plt.plot( comp, wus_entropies, 'r--', linewidth=3.)
plt.xlim(0.0,1.0)
plt.ylabel("Configurational entropy of solution (J/K/mol)")
plt.xlabel("Fraction ferric iron")
plt.show()
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

def Cp(x, a, b, c, d):
    return a + b*x + c/(x*x) + d/np.sqrt(x) 

f=x_to_f(y_to_x(0.9))
Cp_ss=[(1.-f)*wustite().params['Cp'][i] + f*defect_wustite().params['Cp'][i] for i in range(4)]

#print Cp_ss

Cp_FeO=np.array([[298.15, 300.0, 350., 400.0, 450., 500.0, 550., 600.0, 650., 700.0, 750., 800.0, 850., 900.0, 950., 1000.0, 1050., 1100.0, 1150., 1200.0, 1250.],[5.730, 5.739, 5.943, 6.095, 6.218, 6.323, 6.415, 6.498, 6.576, 6.648, 6.718, 6.784, 6.848, 6.910, 6.970, 7.029, 7.086, 7.142, 7.197, 7.251, 7.303]])

Cp_Fe90O=np.array([[298.15, 300.0, 350., 400.0, 450., 500.0, 550., 600.0, 650., 700.0, 750., 800.0, 850., 900.0, 950., 1000.0, 1050., 1100.0, 1150., 1200.0, 1250.],[5.909, 5.913, 6.010, 6.077, 6.127, 6.168, 6.203, 6.234, 6.263, 6.292, 6.320, 6.349, 6.379, 6.410, 6.443, 6.478, 6.514, 6.553, 6.595, 6.639, 6.686]])

Cp_Fe9427O=np.array([[844.61,850.66,853.76,854.41,858.79,871.08,885.84,899.46,912.82,925.52,938.37,951.47,964.58],[10.282,9.787,348.67,184.01,89.69,11.195,6.898,6.937,6.884,6.905,6.721,6.744,6.753]])

Ci, Ci_var = optimize.curve_fit(Cp, Cp_FeO[0], Cp_FeO[1])
#print Ci*8.3145

Ci90, Ci90_var = optimize.curve_fit(Cp, Cp_Fe90O[0], Cp_Fe90O[1])
#print Ci90*8.3145

Ci_defect=(Ci90 - 0.7*Ci)/0.3
#print Ci_defect*8.3145

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


# Find the chemical potentials of Fe and Fe3O4 for different compositions of wuestite at each pressure. 

def iron_eqm(comp, P, T):
    c=comp[0]
    Gexcesses=wus.solution_model.excess_partial_gibbs_free_energies(P, T, [1.0-c,c])
    # FeO and Fe2/3O chemical potentials
    mu=[wus.base_material[i][0].calcgibbs(P,T) + Gexcesses[i] for i in range(2)]
    # Fe and Fe3/4O chemical potentials
    mu_iron=3*mu[0] - 3*mu[1]

    return  mu_iron-iron.calcgibbs(P,T)

def mt_eqm(comp, P, T):
    c=comp[0]
    Gexcesses=wus.solution_model.excess_partial_gibbs_free_energies(P, T, [1.0-c,c])
    # FeO and Fe2/3O chemical potentials
    mu=[wus.base_material[i][0].calcgibbs(P,T) + Gexcesses[i] for i in range(2)]
    # Fe and Fe3/4O chemical potentials
    mu_mt=mu[0] + 3*mu[1]

    return mu_mt-mt.calcgibbs(P,T)



P=1.e9

def iron_wus_mt_eqm(T, P):
    return optimize.fsolve(iron_eqm, 0.16, args=(P, T))[0] - optimize.fsolve(mt_eqm, 0.16, args=(P, T))[0]

T_eqm=optimize.fsolve(iron_wus_mt_eqm, 800., args=(P))[0]
comp_eqm=f_to_y(optimize.fsolve(iron_eqm, 0.16, args=(P, T_eqm))[0])

temperatures=np.linspace(T_eqm-50.,1600.,101)
iron_wus_comp=np.empty_like(temperatures)
mt_wus_comp=np.empty_like(temperatures)

for idx, T in enumerate(temperatures):
    iron_wus_comp[idx]=1.0-f_to_y(optimize.fsolve(iron_eqm, 0.16, args=(P, T))[0])
    mt_wus_comp[idx]=1.0-f_to_y(optimize.fsolve(mt_eqm, 0.16, args=(P, T))[0])

crosses=np.array([[1222.07160224,1172.8465432,1121.78241653,1072.56293888,1021.49881221,972.212357893,922.987298852,871.878521071,871.325963574,923.883111764,972.862589695,1023.7090422,1072.76107818,1121.84102112,1172.74328751,1223.66787946],[0.0428177641654,0.0435528330781,0.0448392036753,0.0453905053599,0.0466768759571,0.0494333843798,0.0501684532925,0.0529249617152,0.0711179173047,0.08067381317,0.0880245022971,0.0939050535988,0.0988667687596,0.102909647779,0.106952526799,0.110260336907]])

invt=np.array([[1223.83253043,1172.77398515,1123.52660055,1074.29037873,1021.4039286,972.134218449,922.903578019,871.805963016,842.459020396,874.992936055,923.977995374,973.052356916,1023.88206525,1074.78433165,1123.87543735,1172.91631056,1225.64090042],[0.0448392036753,0.0459418070444,0.0474119448698,0.0485145482389,0.0498009188361,0.0520061255743,0.0529249617152,0.0553139356815,0.0615620214395,0.070382848392,0.077549770291,0.0817764165391,0.0882082695253,0.0922511485452,0.0959264931087,0.101255742726,0.105298621746]])

plusses=np.array([[1172.38886932,1073.97223957,971.888637339,862.459927372,918.54451332,962.006788364,1034.46995992,1172.33584612],[0.0586217457887,0.058989280245,0.0600918836141,0.0630321592649,0.0764471669219,0.0854517611026,0.0996018376723,0.120367534456]])

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
compositions=np.linspace(0.0,1.0,101)
for c in compositions:
    Gexcesses=wus.solution_model.excess_partial_gibbs_free_energies(P, T, [1.0-c,c])
    # FeO and Fe2/3O chemical potentials
    mu=[wus.base_material[i][0].calcgibbs(P,T) + Gexcesses[i] for i in range(2)]
    # Fe and Fe3/4O chemical potentials
    mu_iron=3*mu[0] - 3*mu[1]
    mu_mt=mu[0] + 3*mu[1]

    print c, mu_iron-iron.calcgibbs(P,T), mu_mt-mt.calcgibbs(P,T)
'''


