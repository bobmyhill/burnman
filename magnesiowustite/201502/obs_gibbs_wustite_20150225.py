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

def gibbs_fcc_1bar(T):
    Tc=201.
    beta=2.1
    p=0.28

    if T<1811.0:
        add=-1462.4 + 8.282*T - 1.15*T*np.log(T) + 6.4e-4*T*T
    else:
        add=-27098.266 + 300.25256*T - 46*T*np.log(T) + 2.78854e31*np.power(T, -9.)

    return GHSERFe(T) + magnetic_gibbs(T, Tc, beta, p) + add

class fcc_iron_Sundman():
    def __init__(self):
        self.params = {
            'formula': {'Fe': 1.0}}
    def set_state(self, pressure, temperature):
        self.gibbs = gibbs_fcc_1bar(temperature)

class bcc_iron():
    def __init__(self):
        self.params = {
            'formula': {'Fe': 1.0}}
    def set_state(self, pressure, temperature):
        self.gibbs = gibbs_bcc_1bar(temperature)

class fcc_iron (Mineral):
    def __init__(self):
        self.params = {
            'S_0': 35.8,
            'Gprime_0': float('nan'),
            'a_0': 5.13074989862e-05,
            'K_0': 153865172537.0,
            'G_0': float('nan'),
            'Kprime_0': 5.2,
            'Kdprime_0': -3.37958221101e-11,
            'V_0': 6.93863394593e-06,
            'name': 'FCC iron',
            'H_0': 7840.,
            'molar_mass': 0.055845,
            'equation_of_state': 'hp_tmt',
            'n': 1.0,
            'formula': {'Fe': 1.0},
            'Cp': [52.2754, -0.000355156, 790710.86, -619.07],
        }
        Mineral.__init__(self)

T_mt_Sundman=[298.15, 350., 400., 450., 500., 550., 600., 650., 700., 750., 800., 848., 850., 900., 950., 1000., 1050., 1100., 1150., 1200., 1250., 1300., 1350., 1400., 1450., 1500., 1550., 1600., 1650., 1700., 1750., 1800., 1850., 1900.]
G_mt_Sundman=[-1159210., -1167570., -1176680., -1186870., -1198070., -1210220., -1223270., -1237200., -1251960., -1267550., -1283960., -1300510., -1301210., -1319210., -1337820., -1356990., -1376690., -1396870., -1417520., -1438620., -1460140., -1482070., -1504390., -1527090., -1550150., -1573570., -1597320., -1621410., -1645820., -1670540., -1695570., -1720890., -1746500., -1772400.]

def gibbs_mt_Sundman(T):
    s = interpolate.InterpolatedUnivariateSpline(T_mt_Sundman, G_mt_Sundman)
    return s(T)
    
def gibbs_mt_ONeill(T):
    if T<600.:
        raise Exception('T too low')
        return 0.
    elif T<848:
        return -1215345 + 2121.987*T - 264.7818*T*np.log(T) + 0.133140*T*T
    elif T<1042:
        return -1354650 + 4245.621*T - 577.5478*T*np.log(T) + 0.309515*T*T
    elif T<1184:
        return -1079617 + 220.694*T - 10.0979*T*np.log(T)
    elif T<1600:
        return -1116959 + 465.536*T - 20.0438*T*np.log(T)
    else:
        raise Exception('T too high')
        return 0.

def mu_O2_Fe_Fe3O4(T):
    if T<750.:
        raise Exception('T too low')
        return 0.
    elif T<833.:
        return -607673 + 1060.994*T - 132.3909*T*np.log(T) + 0.06657*T*T 
    else:
        raise Exception('T too high')
        return 0.

def mu_O2_Fe_FeO_ONeill(T):
    if T<833.:
        raise Exception('T too low')
        return 0.
    elif T<1042.:
        return -605812. + 1366.718*T - 182.7955*T*np.log(T) + 0.10359*T*T
    elif T<1184.:
        return -519357 + 59.427*T + 8.9276*T*np.log(T)
    elif T<1450.:
        return -551159 + 269.404*T - 16.9484*T*np.log(T)
    else:
        raise Exception('T too high')
        return 0.

def mu_O2_FeO_Fe3O4_ONeill(T):
    if T<833.:
        raise Exception('T too low')
        return 0.
    elif T<1274.:
        return -581927. - 65.618*T + 38.7410*T*np.log(T)
    else:
        raise Exception('T too high')
        return 0.


Pr=1.e5

fcc=fcc_iron()
fcc_Sundman=fcc_iron_Sundman()
bcc=bcc_iron()
iron=minerals.HP_2011_ds62.iron()
mt=minerals.HP_2011_ds62.mt()
hem=minerals.HP_2011_ds62.hem()
oxygen=minerals.HP_2011_fluids.O2()


wus=[]
# Bransky fO2 data
phase_T_ys=[]
logfO2=[]
ys_1573=[]
logfO2_1573=[]
for line in open('Bransky_Hed_1968.dat'):
    content=line.strip().split()
    if content[0] != '%':
        phase_T_ys.append([wus, float(content[0]), float(content[2])/100.])
        logfO2.append(float(content[1]))
        if content[0] == '1573.15':
            ys_1573.append(float(content[2])/100.)
            logfO2_1573.append(float(content[1]))


plt.plot( [0.049], [mu_O2_Fe_FeO_ONeill(1373.15)/(constants.gas_constant*1373.15*np.log(10.))], marker='o', linestyle='none', label='ONeill, 1373 (composition guessed)')

plt.plot( [0.122], [mu_O2_FeO_Fe3O4_ONeill(1273.15)/(constants.gas_constant*1273.15*np.log(10.))], marker='o', linestyle='none', label='ONeill, 1273 (composition guessed)')

plt.plot( zip(*phase_T_ys)[2], logfO2, marker='o', linestyle='none', label='Bransky and Hed, 1968')

ys_1573=np.array(ys_1573)
logfO2_1573=np.array(logfO2_1573)

def linear(x, a, b):
    return a + b*x

def quadratic(x, a, b, c):
    return a + b*x + c*x*x

def cubic(x, a, b, c, d):
    return a + b*x + c*x*x + d*x*x*x

def quartic(x, a, b, c, d, e):
    return a + b*x + c*x*x + d*x*x*x + e*x*x*x*x

def mod_quartic(x, a, b, c, e):
    return a + b*x + c*x*x + e*x*x*x*x

guesses=[0., 1., 1.]
popt, pcov = optimize.curve_fit(quadratic, ys_1573, logfO2_1573, guesses)
a, b, c = popt

ylin=np.linspace(0.05, 0.15, 101)

plt.plot( ylin, quadratic(ylin, a, b, c), 'r-', linewidth=1, label='Quadratic fit')

# One continuous dataset: Bransky data
def entropy(y):
    return -constants.gas_constant*((1.-3.*y)*np.log(1.-3.*y) + 2.*y*np.log(2.*y) + y*np.log(y))


####################################################
def fO2_to_gibbs_FeO_mt(function, params, y_Fe, y_Fe3O4, gibbs_FeO_Fe_obs):
    ys=np.linspace(y_Fe, y_Fe3O4, 1001)
    G0=gibbs_FeO_Fe_obs
    gibbs=np.empty_like(ys)
    for i in range(len(ys)-1):
        gibbs[i]=G0
        y_mid=(ys[i] + ys[i+1])/2.
        log10fO2_FeO=function(y_mid, *params)
        mu_O2_FeO=constants.gas_constant*T*np.log(np.power(10., log10fO2_FeO)) + oxygen.gibbs
        G1=(1.-ys[i+1])/(1.-ys[i])*G0 + (0.5 - (0.5*(1.-ys[i+1])/(1.-ys[i])))*mu_O2_FeO
        G0=G1
    gibbs[len(ys)-1]=G0
    return G0, ys, gibbs
    
def fit_fO2_and_eqm(data, a, b, c, d, e):
    array_to_minimize=[]
    for datum in data:
        if datum[0] == 'fO2':
            y=datum[1]
            fO2_obs=datum[2]
            array_to_minimize.append(quartic(y, a, b, c, d, e) - fO2_obs)
        else:
            y_Fe=datum[1]
            y_Fe3O4=datum[2]
            gibbs_Fe_obs=datum[3]
            gibbs_mt_obs=datum[4]
            log10fO2_Fe=quartic(y_Fe, a, b, c, d, e)
            log10fO2_Fe3O4=quartic(y_Fe3O4, a, b, c, d, e)
            mu_O2_Fe=constants.gas_constant*T*np.log(np.power(10., log10fO2_Fe)) + oxygen.gibbs
            mu_O2_Fe3O4=constants.gas_constant*T*np.log(np.power(10., log10fO2_Fe3O4)) + oxygen.gibbs

            gibbs_FeO_Fe = (1.-y_Fe)*gibbs_Fe_obs + 0.5*mu_O2_Fe
            gibbs_FeO_mt = (1.-y_Fe3O4)*(gibbs_mt_obs/3. - 2./3.*mu_O2_Fe3O4) + 0.5*mu_O2_Fe3O4

            array_to_minimize.append(fO2_to_gibbs_FeO_mt(quartic, [a, b, c, d, e], y_Fe, y_Fe3O4, gibbs_FeO_Fe)[0] - gibbs_FeO_mt)

    print fO2_to_gibbs_FeO_mt(quartic, [a, b, c, d, e], y_Fe, y_Fe3O4, gibbs_FeO_Fe),  gibbs_FeO_mt
    return array_to_minimize

T=1573.15
fO2_eqm_data=[]
zeros=[]
for i, y in enumerate(ys_1573):
    fO2_eqm_data.append(['fO2', y, logfO2_1573[i]])
    zeros.append(0.0)

oxygen.set_state(Pr, T)
bcc.set_state(Pr, T)
fcc.set_state(Pr, T)
fcc_Sundman.set_state(Pr, T)
mt.set_state(Pr, T)

# Two pins: Composition and mu_O2 of wustite in equilibrium with iron and magnetite
y_Fe=0.0462
y_Fe3O4=0.1539
gibbs_mt_ONeill_1573=gibbs_mt_ONeill(T) + (3.*fcc.gibbs + 2.*oxygen.gibbs)
gibbs_mt_Sundman=-1608473.67

fO2_eqm_data.append(['fe-fe3o4', y_Fe, y_Fe3O4, fcc.gibbs, gibbs_mt_ONeill_1573])
zeros.append(0.0)
zeros=np.array(zeros)

guesses=[0., 0., 0., 0., 0.]
popt_all_quartic, pcov = optimize.curve_fit(fit_fO2_and_eqm, fO2_eqm_data, zeros, guesses)

print popt_all_quartic

print fit_fO2_and_eqm(fO2_eqm_data, *popt_all_quartic)






log10fO2_Fe=quartic(y_Fe, *popt_all_quartic)
log10fO2_Fe3O4=quartic(y_Fe3O4, *popt_all_quartic)

mu_O2_Fe=constants.gas_constant*T*np.log(np.power(10., log10fO2_Fe)) + oxygen.gibbs
mu_O2_Fe3O4=constants.gas_constant*T*np.log(np.power(10., log10fO2_Fe3O4)) + oxygen.gibbs

gibbs_FeO_Fe = (1.-y_Fe)*fcc.gibbs + 0.5*mu_O2_Fe
gibbs_FeO_Fe_Sundman = (1.-y_Fe)*fcc_Sundman.gibbs + 0.5*mu_O2_Fe
gibbs_FeO_Fe3O4_ONeill = (1.-y_Fe3O4)*(gibbs_mt_ONeill_1573/3. - 2./3.*mu_O2_Fe3O4) + 0.5*mu_O2_Fe3O4
gibbs_FeO_Fe3O4_Sundman = (1.-y_Fe3O4)*(gibbs_mt_Sundman/3. - 2./3.*mu_O2_Fe3O4) + 0.5*mu_O2_Fe3O4
gibbs_FeO_Fe3O4_HP2011 = (1.-y_Fe3O4)*(mt.gibbs/3. - 2./3.*mu_O2_Fe3O4) + 0.5*mu_O2_Fe3O4




plt.plot( ylin, quartic(ylin, *popt_all_quartic), 'b-', linewidth=1, label='Quartic fit')


plt.legend(loc="upper left")
plt.ylabel("log10(fO2)")
plt.xlabel("y in Fe(1-y)O")
plt.show()

####################################################

# The exact shape of the gibbs free energy curve should satisfy both the iron and the magnetite pin...
ys=np.linspace(y_Fe, y_Fe3O4, 1001)
gibbs=np.empty_like(ys)
gibbs_sub_conf=np.empty_like(ys)

G0=gibbs_FeO_Fe
gibbs[0]=G0
gibbs_sub_conf[0]=G0 + T*entropy(y_Fe)

for i in range(len(ys)-1):
    y_mid=(ys[i] + ys[i+1])/2.
    log10fO2_FeO=quartic(y_mid, *popt_all_quartic)
    mu_O2_FeO=constants.gas_constant*T*np.log(np.power(10., log10fO2_FeO)) + oxygen.gibbs
    G1=(1.-ys[i+1])/(1.-ys[i])*G0 + (0.5 - (0.5*(1.-ys[i+1])/(1.-ys[i])))*mu_O2_FeO
    gibbs[i+1]=G1
    gibbs_sub_conf[i+1]=G1 + T*entropy(ys[i+1])
    G0=G1

print gibbs_FeO_Fe3O4_ONeill, gibbs[len(ys)-1]



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



plt.plot( ys, gibbs, 'r-', linewidth=1, label='From data')
plt.plot(fO2_to_gibbs_FeO_mt(quartic, popt_all_quartic, y_Fe, y_Fe3O4, gibbs_FeO_Fe)[1], fO2_to_gibbs_FeO_mt(quartic, popt_all_quartic, y_Fe, y_Fe3O4, gibbs_FeO_Fe)[2], 'r-', linewidth=1, label='Modelled')
plt.plot( ys, G_wustite(np.array(ys), 1573.15)[1], 'b-', linewidth=1, label='Sundman')
#plt.plot( y_Fe, [gibbs_FeO_Fe_Sundman], marker='o', linestyle='none', label='Fe-FeO Sundman')
plt.plot( y_Fe3O4, [gibbs_FeO_Fe3O4_HP2011], marker='o', linestyle='none', label='HP')
plt.plot( y_Fe3O4, [gibbs_FeO_Fe3O4_ONeill], marker='o', linestyle='none', label='ONeill')
plt.plot( y_Fe, [gibbs_FeO_Fe], c='r', marker='o', linestyle='none', label='Fe-FeO')
#plt.plot( y_Fe3O4, [gibbs_FeO_Fe3O4_Sundman], marker='o', linestyle='none', label='Sundman')
#plt.plot( [1.-0.947], [-426889], marker='o', linestyle='none', label='Sundman')
plt.legend(loc="upper left")
plt.ylabel("log10(fO2)")
plt.xlabel("y in Fe(1-y)O")
plt.show()

guesses=[1., 1., 1., 1., 1.]
popt_quartic, pcov = optimize.curve_fit(quartic, ys, gibbs_sub_conf, guesses)
print 'Quartic:', popt_quartic

guesses=[1., 1., 1., 1.]
popt_mod_quartic, pcov = optimize.curve_fit(mod_quartic, ys, gibbs_sub_conf, guesses)
print 'Mod. Quartic:', popt_mod_quartic

guesses=[1., 1., 1., 1.]
popt_cubic, pcov = optimize.curve_fit(cubic, ys, gibbs_sub_conf, guesses)
print 'Cubic:', popt_cubic

guesses=[1., 1., 1.]
popt_quadratic, pcov = optimize.curve_fit(quadratic, ys, gibbs_sub_conf, guesses)
print 'Quadratic:', popt_quadratic

y_full=np.linspace(0.0001, 0.33333, 101)
print popt
#plt.plot( ys, gibbs_sub_conf, 'r-', linewidth=1, label='From data')
a, b, c, d, e = popt_quartic
plt.plot( y_full, quartic(y_full, 0, 0, c, d, e) - 3.*y_full*quartic(1./3., 0, 0, c, d, e), '--', linewidth=1, label='Quartic')
a, b, c, e = popt_mod_quartic
plt.plot( y_full, mod_quartic(y_full, 0, 0, c, e) - 3.*y_full*mod_quartic(1./3., 0, 0, c, e), '--', linewidth=1, label='Mod. Quartic')
a, b, c, d = popt_cubic
plt.plot( y_full, cubic(y_full, 0, 0, c, d) - 3.*y_full*cubic(1./3., 0, 0, c, d), '--', linewidth=1, label='Cubic')
a, b, c = popt_quadratic
plt.plot( y_full, quadratic(y_full, 0, 0, c) - 3.*y_full*quadratic(1./3., 0, 0, c), '--', linewidth=1, label='Quadratic')
#plt.plot( ys, gibbs_sub_conf, linewidth=1, label='observed')
plt.legend(loc="lower left")
plt.ylabel("Gibbs misfit")
plt.xlabel("y in Fe(1-y)O")
plt.show()

# quartic doesn't really improve the fit...
# cubic yields a roughly symmetric misfit about y=0.1
gibbs_fit_quadratic=np.empty_like(ys)
gibbs_fit_cubic=np.empty_like(ys)
gibbs_fit_quartic=np.empty_like(ys)
logfO2_fit_quadratic=np.empty_like(ys)
logfO2_fit_cubic=np.empty_like(ys)
logfO2_fit_quartic=np.empty_like(ys)

for i in range(len(ys)):
    delta=0.000001
    a, b, c = popt_quadratic
    gibbs_fit_quadratic[i]= quadratic(ys[i], a, b, c) - T*entropy(ys[i])
    gibbs0=quadratic(ys[i]-delta, a, b, c) - T*entropy(ys[i]-delta)
    gibbs1=quadratic(ys[i]+delta, a, b, c) - T*entropy(ys[i]+delta)
    mu_O2=2*((1.-ys[i]+delta)/(1.-ys[i]-delta)*gibbs1 - gibbs0)/((1.-ys[i]+delta)/(1.-ys[i]-delta)-1.)
    logfO2_fit_quadratic[i]=np.log10(np.exp((mu_O2-oxygen.gibbs)/(constants.gas_constant*T)))

    a, b, c, d = popt_cubic
    gibbs_fit_cubic[i]= cubic(ys[i], a, b, c, d) - T*entropy(ys[i])
    gibbs0=cubic(ys[i]-delta, a, b, c, d) - T*entropy(ys[i]-delta)
    gibbs1=cubic(ys[i]+delta, a, b, c, d) - T*entropy(ys[i]+delta)
    mu_O2=2*((1.-ys[i]+delta)/(1.-ys[i]-delta)*gibbs1 - gibbs0)/((1.-ys[i]+delta)/(1.-ys[i]-delta)-1.)
    logfO2_fit_cubic[i]=np.log10(np.exp((mu_O2-oxygen.gibbs)/(constants.gas_constant*T)))

    a, b, c, d, e = popt_quartic
    gibbs_fit_quartic[i]= quartic(ys[i], a, b, c, d, e) - T*entropy(ys[i])
    gibbs0=quartic(ys[i]-delta, a, b, c, d, e) - T*entropy(ys[i]-delta)
    gibbs1=quartic(ys[i]+delta, a, b, c, d, e) - T*entropy(ys[i]+delta)
    mu_O2=2*((1.-ys[i]+delta)/(1.-ys[i]-delta)*gibbs1 - gibbs0)/((1.-ys[i]+delta)/(1.-ys[i]-delta)-1.)
    logfO2_fit_quartic[i]=np.log10(np.exp((mu_O2-oxygen.gibbs)/(constants.gas_constant*T)))


#plt.plot( ys, gibbs, 'r-', linewidth=1, label='From data')
#plt.plot( ys, gibbs_fit, 'b-', linewidth=1, label='Fit')
plt.plot( ys, logfO2_fit_quadratic, '-', linewidth=1, label='Fit quadratic')
plt.plot( ys, logfO2_fit_cubic, '-', linewidth=1, label='Fit cubic')
plt.plot( ys, logfO2_fit_quartic, '-', linewidth=1, label='Fit quartic')

plt.plot( zip(*phase_T_ys)[2], logfO2, marker='o', linestyle='none', label='Bransky and Hed, 1968')
plt.legend(loc="upper left")
plt.ylabel("Gibbs")
plt.xlabel("y in Fe(1-y)O")
plt.show()

plt.plot( ys, gibbs_fit_quadratic, '-', linewidth=1, label='Quadratic')
plt.plot( ys, gibbs_fit_cubic, '-', linewidth=1, label='Cubic')
plt.plot( ys, gibbs_fit_quartic, '-', linewidth=1, label='Quartic')
plt.plot( ys, G_wustite(np.array(ys), 1573.15)[1], '-', linewidth=1, label='Sundman')
#plt.plot( y_Fe, [gibbs_FeO_Fe_Sundman], marker='o', linestyle='none', label='Fe-FeO Sundman')
plt.plot( y_Fe3O4, [gibbs_FeO_Fe3O4_HP2011], marker='o', linestyle='none', label='HP')
plt.plot( y_Fe3O4, [gibbs_FeO_Fe3O4_ONeill], marker='o', linestyle='none', label='ONeill')
plt.plot( y_Fe, [gibbs_FeO_Fe], c='r', marker='o', linestyle='none', label='Fe-FeO')
#plt.plot( y_Fe3O4, [gibbs_FeO_Fe3O4_Sundman], marker='o', linestyle='none', label='Sundman')
#plt.plot( [1.-0.947], [-426889], marker='o', linestyle='none', label='Sundman')
plt.legend(loc="upper left")
plt.ylabel("log10(fO2)")
plt.xlabel("y in Fe(1-y)O")
plt.show()


print 'THIS IS NOT ABSOLUTELY THE BEST WAY TO DO THINGS'
print 'BETTER TO INVERT DIRECTLY FOR THE ENDMEMBER GIBBS AND THEIR EXCESSES (PROBABLY BEST TO REWRITE THE GIBBS EXCESS FUNCTION IN THE MAIN PROGRAM, RATHER THAN FAFFING HERE)'

def Fe_eqm(y, function, params):
    gibbs_FeO_Fe=function(y, *params) - T*entropy(ys[i])
    mu_O2_method_one=2.*(gibbs_FeO_Fe - (1-y)*fcc.gibbs)
    
    delta=0.00001
    gibbs0=function(y-delta, *params) - T*entropy(y-delta)
    gibbs1=function(y+delta, *params) - T*entropy(y+delta)
    mu_O2_method_two=2*((1.-y+delta)/(1.-y-delta)*gibbs1 - gibbs0)/((1.-y+delta)/(1.-y-delta)-1.)
    
    return mu_O2_method_one - mu_O2_method_two

print optimize.fsolve(Fe_eqm, 0.046, args=(quadratic, popt_quadratic))
print optimize.fsolve(Fe_eqm, 0.046, args=(cubic, popt_cubic))
print optimize.fsolve(Fe_eqm, 0.046, args=(quartic, popt_quartic))

def Fe_eqm(y):

    gibbs_FeO_Fe=G_wustite(y, T)[1]
    mu_O2_method_one=2.*(gibbs_FeO_Fe - (1-y)*fcc.gibbs)
    
    delta=0.00001
    gibbs0=G_wustite(y-delta, T)[1]
    gibbs1=G_wustite(y+delta, T)[1]
    mu_O2_method_two=2*((1.-y+delta)/(1.-y-delta)*gibbs1 - gibbs0)/((1.-y+delta)/(1.-y-delta)-1.)
    
    return mu_O2_method_one - mu_O2_method_two

print optimize.fsolve(Fe_eqm, 0.046)

