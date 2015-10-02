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

fcc=fcc_iron()

'''
Excess properties
'''

nan=float('nan')
class wustite (Mineral):
    def __init__(self):
        self.params = {
            'S_0': 6.50917432e+01,
            'Gprime_0': nan,
            'a_0': 3.22e-05,
            'K_0': 1.52e+11,
            'G_0': nan,
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
            'Gprime_0': nan,
            'a_0': 3.22e-05,
            'K_0': 1.52e+11,
            'G_0': nan,
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
initial_alphas=[1.0, 1.0, 0.68]
initial_enthalpy_interaction=[[11.e3, 0.0e3], [-3.9]]
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

'''
print arr_compositions


plt.plot( [11.3, 19.7, 26.8, 35.7], [-10.82, -10.6, -9, -8], 'o', linestyle='none', label='gibbs')

plt.ylabel("Gibbs")
plt.xlabel("p(O)")
plt.legend(loc='lower right')
plt.show()
'''

xO=0.54
y=(xO-1.0)/xO
f_Fe23O=3.*y

oxygen.set_state(Pr, T)
wus.set_composition([0., 1.-f_Fe23O, f_Fe23O])
wus.set_state(Pr, 1673.)
print np.log10(fugacity(oxygen, [wus]))


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



y=1.-0.947
f_Fe23O=3.*y
T=1700.
wus.set_composition([0., 1.-f_Fe23O, f_Fe23O])
wus.set_state(Pr, T)

oxygen.set_state(Pr, T)
print 298.15, G_wustite(y, 298.15)[1], '-282664 J/mol'
print 1700.00, G_wustite(y, 1700.)[1], '-445923 J/mol'


def quadratic(X, a, b, c):
    return a + b*X + c*X*X

def asymmetric_model(y, aj, Wij, y0, y1):
    Xj=3.*y
    ai=1.
    Xi=(1.-Xj)
    phii=Xi*ai/(Xi*ai + Xj*aj)
    phij=Xj*aj/(Xi*ai + Xj*aj)
    Gex=(2.*ai/(ai+aj)*phij*phij*Xi + 2.*aj/(ai+aj)*phii*phii*Xj)*Wij
    return Gex + y0*Xi + y1*Xj

T=1700.
#ys=np.linspace(0.048, (2.*0.545-1.0)/0.545, 101)
ys=np.linspace(0.00, 0.33333, 101)
gibbs_Sundman=np.empty_like(ys)
excess_enthalpy_Sundman=np.empty_like(ys)
gibbs_model=np.empty_like(ys)
for i, y in enumerate(ys):
    f_Fe23O=3.*y
    wus.set_composition([0., 1.-f_Fe23O, f_Fe23O])
    wus.set_state(Pr, T)
    gibbs_model[i]=wus.gibbs
    excess_enthalpy_Sundman[i]=G_wustite(y, T)[0] #- (float(100-i)/100.*G_wustite(ys[0], T)[0] + float(i)/100.*G_wustite(ys[100], T)[0])
    gibbs_Sundman[i]=G_wustite(y, T)[1]# - wus.gibbs


plt.plot( ys, gibbs_model-gibbs_Sundman, 'b-', linewidth=1, label='Gibbs (model-Sundman)')
#plt.plot( ys, gibbs_model, 'r-', linewidth=1, label='Gibbs (model)')
#plt.xlim(0.048, (2.*0.545-1.0)/0.545)
plt.legend(loc="upper left")
plt.ylabel("Gibbs excess")
plt.xlabel("y in Fe(1-y)O")
plt.show()

guesses=[0., -2., 2.]
popt, pcov = optimize.curve_fit(quadratic, ys, excess_enthalpy_Sundman, guesses)

guesses=[0.24, -15000., 1700., 150.]
popt1, pcov1 = optimize.curve_fit(asymmetric_model, ys, excess_enthalpy_Sundman, guesses)
print popt1

x=np.linspace(0.0, 1./3., 101)
#plt.plot( x, asymmetric_model(x, 0.2, 000, 1000., -4000.), 'r-', linewidth=1, label='Gibbs (model2)')

plt.plot( x, G_wustite(x, T)[0], 'b-', linewidth=1, label='H_ex (Sundman)')
plt.plot( ys, quadratic(ys, popt[0], popt[1], popt[2]), 'r-', linewidth=1, label='Gibbs (model)')
plt.plot( ys, asymmetric_model(ys, popt1[0], popt1[1], popt1[2], popt1[3]), 'r-', linewidth=1, label='Gibbs (model2)')
plt.xlim(0., 1./3.)
plt.legend(loc="lower left")
plt.ylabel("Gibbs excess")
plt.xlabel("y in Fe(1-y)O")
plt.show()

T=1700.
fcc.set_state(Pr, T)
print gibbs_fcc_1bar(T), fcc.gibbs



def G_wustite_modified(y, T):
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
    DeltamixGex=asymmetric_model(y, 0.082577, -54., 1430., 3219.)
    DeltamixG=y2*G_2O + y3*G_3O + yv*G_VO + constants.gas_constant*T*(y2*np.log(y2) + y3*np.log(y3) + yv*np.log(yv)) + DeltamixGex 
    return DeltamixGex, DeltamixG


#y-d, y+d
#Fe(1-y-d)O Fe(1-y+d)O
#(1-y+d)/(1-y-d) * G(Fe(1-y-d)O) - 1*G(Fe(1-y+d)O) = 0.5*((1-y+d)/(1-y-d)-1.)*mu(O2)

#Plotting fO2s
temperatures=np.linspace(1673.15, 873.15, 9)

d=0.0001
xOs=np.linspace(0.505, 0.550, 101)
for T in temperatures:
    sundman_fO2=np.empty_like(xOs)
    model_fO2=np.empty_like(xOs)
    for idx, xO in enumerate(xOs):
        y=(2.0*xO-1.0)/xO
        mu_O2=2*((1.-y+d)/(1.-y-d)*G_wustite(y+d, T)[1] - G_wustite(y-d, T)[1])/((1.-y+d)/(1.-y-d)-1.)
        
        oxygen.set_state(Pr, T)
        wus.set_composition([0.0, 1.-3.*y, 3.*y])
        wus.set_state(Pr, T)
        sundman_fO2[idx]=np.log10(np.exp((mu_O2-oxygen.gibbs)/(constants.gas_constant*T)))
        model_fO2[idx]=np.log10(fugacity(oxygen, [wus]))

    plt.plot( xOs, sundman_fO2, 'b-', linewidth=1, label='Sundman'+str(T))
    plt.plot( xOs, model_fO2, 'r-', linewidth=1, label='Model'+str(T))

plt.legend(loc="upper left")
plt.ylabel("log10(fO2)")
plt.xlabel("Mole fraction O")
plt.show()


# Fitting MAGNETITE
temperatures=[298.15, 350., 400., 450., 500., 550., 600., 650., 700., 750., 800., 848., 850., 900., 950., 1000., 1050., 1100., 1150., 1200., 1250., 1300., 1350., 1400., 1450., 1500., 1550., 1600., 1650., 1700., 1750., 1800., 1850., 1900.]
gibbs_mt=[-1159210., -1167570., -1176680., -1186870., -1198070., -1210220., -1223270., -1237200., -1251960., -1267550., -1283960., -1300510., -1301210., -1319210., -1337820., -1356990., -1376690., -1396870., -1417520., -1438620., -1460140., -1482070., -1504390., -1527090., -1550150., -1573570., -1597320., -1621410., -1645820., -1670540., -1695570., -1720890., -1746500., -1772400.]


def fit_mt(temperatures, H_0, S_0, Smax):
    mt.params['H_0']=H_0
    mt.params['S_0']=S_0
    mt.params['landau_Smax']=Smax
    gibbs=np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        mt.set_state(1.e5, T)
        gibbs[i]=mt.gibbs
    return gibbs

guesses=[mt.params['H_0'], mt.params['S_0'], mt.params['landau_Smax']]
popt, pcov = optimize.curve_fit(fit_mt, temperatures, gibbs_mt, guesses)

print popt

gibbs_mt_model=np.empty_like(temperatures)
for i, temperature in enumerate(temperatures):
    mt.set_state(1.e5, temperature)
    gibbs_mt_model[i] = mt.gibbs

plt.plot( temperatures, gibbs_mt_model - gibbs_mt, 'b-', linewidth=1, label='Model - Sundman')
plt.legend(loc="upper left")
plt.ylabel("Gibbs magnetite")
plt.xlabel("Mole fraction O")
plt.show()


print wus.solution_model._configurational_entropy( [0.0, 0.5, 0.5] )
print constants.gas_constant*(0.5*np.log(0.5) + 1./3.*np.log(1./3.)  + 1./6.*np.log(1./6.)) 


# Fit to Gibbs

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

def gibbs_wus(ys, Wij, Gi, Gj):
    gibbs=[]
    aj=1.0
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

T=1700.
ys=np.linspace(0.05, 0.16, 101)
gibbs=np.empty_like(ys)
for i, y in enumerate(ys):
    gibbs[i]=G_wustite(y, T)[1]

guesses=[-20.e3, -260000, -250000]
popt, pcov = optimize.curve_fit(gibbs_wus, ys, gibbs, guesses)

print popt


plt.plot( ys,  gibbs_wus(ys, popt[0], popt[1], popt[2])-gibbs, 'r-', linewidth=1, label='model')
plt.legend(loc="upper left")
plt.ylabel("Gibbs magnetite")
plt.xlabel("Mole fraction O")
plt.show()



d=0.0001
xOs=np.linspace(0.505, 0.550, 101)

sundman_fO2=np.empty_like(xOs)
model_fO2=np.empty_like(xOs)
for idx, xO in enumerate(xOs):
    y=(2.0*xO-1.0)/xO
    mu_O2_Sundman=2*((1.-y+d)/(1.-y-d)*G_wustite(y+d, T)[1] - G_wustite(y-d, T)[1])/((1.-y+d)/(1.-y-d)-1.)
    mu_O2_model=2*((1.-y+d)/(1.-y-d)*gibbs_wus([y+d], popt[0], popt[1], popt[2])[0] - gibbs_wus([y-d], popt[0], popt[1], popt[2])[0])/((1.-y+d)/(1.-y-d)-1.)

    sundman_fO2[idx]=np.log10(np.exp((mu_O2_Sundman-oxygen.gibbs)/(constants.gas_constant*T)))
    model_fO2[idx]=np.log10(np.exp((mu_O2_model-oxygen.gibbs)/(constants.gas_constant*T)))
    


plt.plot( xOs, sundman_fO2, 'b-', linewidth=1, label='Sundman'+str(T))
plt.plot( xOs, model_fO2, 'r-', linewidth=1, label='Model'+str(T))

plt.legend(loc="upper left")
plt.ylabel("log10(fO2)")
plt.xlabel("Mole fraction O")
plt.show()


# Fit to fO2

def fO2_wus(ys, Wij, Gj):
    fO2=[]
    ai=1.0
    for y in ys:
        Xj=3.*(y - d)
        Xi=(1.-Xj)
        phii=Xi*ai/(Xi*ai + Xj*aj)
        phij=Xj*aj/(Xi*ai + Xj*aj)
        Hex=(2.*ai/(ai+aj)*phij*phij*Xi + 2.*aj/(ai+aj)*phii*phii*Xj)*Wij
        
        
    # Fe(1-y)O
        Sex=-constants.gas_constant*(Xi*np.log(Xi) + 2.*Xj/3.*np.log(2.*Xj/3.) + Xj/3.*np.log(Xj/3.))
        gibbs0=Gi*Xi + Gj*Xj - Sex*T + Hex


        Xj=3.*(y + d)
        Xi=(1.-Xj)
        phii=Xi*ai/(Xi*ai + Xj*aj)
        phij=Xj*aj/(Xi*ai + Xj*aj)
        Hex=(2.*ai/(ai+aj)*phij*phij*Xi + 2.*aj/(ai+aj)*phii*phii*Xj)*Wij
        
        
    # Fe(1-y)O
        Sex=-constants.gas_constant*(Xi*np.log(Xi) + 2.*Xj/3.*np.log(2.*Xj/3.) + Xj/3.*np.log(Xj/3.))
        gibbs1=Gi*Xi + Gj*Xj - Sex*T + Hex


        mu_O2=2*((1.-y+d)/(1.-y-d)*gibbs1 - gibbs0)/((1.-y+d)/(1.-y-d)-1.)

        fO2.append(np.log10(np.exp((mu_O2-oxygen.gibbs)/(constants.gas_constant*T))))

    return fO2

T=1700.
oxygen.set_state(Pr, T)
ys=np.linspace(0.05, 0.16, 101)
fO2s=np.empty_like(ys)
d=0.001
for i, y in enumerate(ys):
    mu_O2_Sundman=2*((1.-y+d)/(1.-y-d)*G_wustite(y+d, T)[1] - G_wustite(y-d, T)[1])/((1.-y+d)/(1.-y-d)-1.)
    fO2s[i]=np.log10(np.exp((mu_O2_Sundman-oxygen.gibbs)/(constants.gas_constant*T)))


Gi=-449000.
aj=1.0
guesses=[-20.e3, -250000]
popt, pcov = optimize.curve_fit(fO2_wus, ys, fO2s, guesses)

print popt


Wij, Gj = popt

plt.plot( ys, fO2s, 'b-', linewidth=1, label='Sundman'+str(T))
plt.plot( ys, fO2_wus(ys, Wij, Gj), 'r-', linewidth=1, label='Model'+str(T))

#plt.plot( ys, G_wustite(ys, T)[1], 'b-', linewidth=1, label='Sundman'+str(T))
#plt.plot( ys, gibbs_wus_full(ys, aj, Wij, Gi, Gj), 'r-', linewidth=1, label='Model'+str(T))

plt.legend(loc="upper left")
plt.ylabel("log10(fO2)")
plt.xlabel("Mole fraction O")
plt.show()


