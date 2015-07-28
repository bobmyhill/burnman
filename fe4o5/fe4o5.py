import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

# Benchmarks for the solid solution class
import burnman
from burnman import minerals
from burnman.chemicalpotentials import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scipy.optimize as optimize

from mineral_models import *
from equilibrium_functions import *

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


# EMOD minerals
hen = minerals.HP_2011_ds62.hen()
mag = minerals.HP_2011_ds62.mag()
fo = minerals.HP_2011_ds62.fo()
mwd = minerals.HP_2011_ds62.mwd()
mrw = minerals.HP_2011_ds62.mrw()
diam = minerals.HP_2011_ds62.diam()
gph = minerals.HP_2011_ds62.gph()
EMOG=[hen, mag, fo, gph]
EMOD=[hen, mag, fo, diam]
EMWD=[hen, mag, mwd, diam]
EMRD=[hen, mag, mrw, diam]

O2=minerals.HP_2011_fluids.O2()
fe4o5=Fe4O5()

P=11.5e9
T=1366.
V=345.753 # Angstroms^3

Nb=6.022e23
Z=4
A3_to_m3=1e-30

V=V*A3_to_m3*Nb/Z

fe4o5.set_state(P,T)
print fe4o5.V, V

fe5o6=Fe5O6()
P=15.e9
T=2000.
V=2.8729*9.713*14.974 # Angstroms^3
V=V*A3_to_m3*Nb/Z

fe5o6.set_state(P,T)
print fe5o6.V, V

P=11.4e9
T=300.
V=414. # Angstroms^3
V=V*A3_to_m3*Nb/Z
fe5o6.set_state(P,T)
print fe5o6.V, V

P=1.e5
T=300.
V=440.6 # Angstroms^3
V=V*A3_to_m3*Nb/Z
fe5o6.set_state(P,T)
print fe5o6.V, V

P=20.e9
T=300.
V=400.4 # Angstroms^3
V=V*A3_to_m3*Nb/Z
fe5o6.set_state(P,T)
print fe5o6.V, V

'''
P=9.5e9
T=1473.15
wus.set_composition([0.0, 1.0, 0.0])
wus.set_state(P, T)
fe5o6.set_state(P,T)
fe4o5.set_state(P,T)
'''

def wus_fe5o6_frw_stv_eqm(arg, T):
    XMgO=0.0
    c=arg[0]
    P=arg[1]
    wus.set_composition([XMgO, (1.0-c)*(1.-XMgO), c*(1.-XMgO)])
    wus.set_state(P,T)
    frw.set_state(P,T)
    stv.set_state(P,T)
    mu_fe5o6=chemical_potentials([wus],[dictionarize_formula('Fe5O6')])[0]
    mu_FeO=chemical_potentials([wus],[dictionarize_formula('FeO')])[0]
    mu_FeO_2=chemical_potentials([frw, stv],[dictionarize_formula('FeO')])[0]
    return [mu_fe5o6-fe5o6.calcgibbs(P,T), mu_FeO - mu_FeO_2]

def wus_iron_frw_stv_eqm(arg, T):
    XMgO=0.0
    c=arg[0]
    P=arg[1]
    wus.set_composition([XMgO, (1.0-c)*(1.-XMgO), c*(1.-XMgO)])
    wus.set_state(P,T)
    frw.set_state(P,T)
    stv.set_state(P,T)
    mu_iron=chemical_potentials([wus],[dictionarize_formula('Fe')])[0]
    mu_FeO=chemical_potentials([wus],[dictionarize_formula('FeO')])[0]
    mu_FeO_2=chemical_potentials([frw, stv],[dictionarize_formula('FeO')])[0]
    return [mu_iron-iron.calcgibbs(P,T), mu_FeO - mu_FeO_2]

def hem_mt_fe4o5_rhenium_univariant(T):
    return optimize.fsolve(eqm_pressure, 10.e9, args=(T, [mt, Re, fe4o5, ReO2], [8., 1., -6., -1.]))[0] \
        - optimize.fsolve(eqm_pressure, 10.e9, args=(T, [hem, Re, fe4o5, ReO2], [4., 1., -2., -1.]))[0]

univariant=optimize.fsolve(hem_mt_fe4o5_rhenium_univariant, 1100., args=())[0]

temperatures=np.linspace(873.15, 1773.15, 91)
low_temperatures=np.linspace(873.15, univariant, 91)
high_temperatures=np.linspace(univariant, 1773.15, 91)

fe4o5_breakdown_pressures=np.empty_like(temperatures)
mt_high_mt_pressures=np.empty_like(temperatures)
for idx, T in enumerate(temperatures):
    fe4o5_breakdown_pressures[idx]=optimize.fsolve(wus_eqm_c_P, [0.5,10.e9], args=(T, wus, fe4o5, mt))[1]
    mt_high_mt_pressures[idx]=optimize.fsolve(eqm_pressure, 10.e9, args=(T, [mt, high_mt], [1., -1.]))[0]

stable_mt_breakdown_pressures=np.empty_like(low_temperatures)
hem_re_pressures=np.empty_like(low_temperatures)
for idx, T in enumerate(low_temperatures):
    # 2Fe3O4 -> Fe4O5 + Fe2O3
    stable_mt_breakdown_pressures[idx]=optimize.fsolve(eqm_pressure, 10.e9, args=(T, [mt, fe4o5, hem], [2., -1., -1.]))[0]
    # 8Fe3O4 + Re -> 6Fe4O5 + ReO2
    hem_re_pressures[idx]=optimize.fsolve(eqm_pressure, 10.e9, args=(T, [mt, Re, fe4o5, ReO2], [8., 1., -6., -1.]))[0]

metastable_mt_breakdown_pressures=np.empty_like(high_temperatures)
mt_re_pressures=np.empty_like(high_temperatures)
for idx, T in enumerate(high_temperatures):
    metastable_mt_breakdown_pressures[idx]=optimize.fsolve(eqm_pressure, 10.e9, args=(T, [mt, fe4o5, hem], [2., -1., -1.]))[0]
    mt_re_pressures[idx]=optimize.fsolve(eqm_pressure, 10.e9, args=(T, [mt, Re, fe4o5, ReO2], [8., 1., -6., -1.]))[0]


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

################################
# Fe-O oxygen fugacity diagram #
################################

lines=[]
T=1473.15
O2.set_state(1.e5, T)
XMgO=0.0

min_pressure = 1.e5
max_pressure = 24.e9
full_pressures = np.linspace(min_pressure, max_pressure, 101)

iron_wus_fO2=eqm_curve_wus(iron, wus, full_pressures, T, O2)
lines.append([full_pressures, iron_wus_fO2, 'iron - wus', '-W1,black'])

Re_fO2 = eqm_curve([Re, ReO2], full_pressures, T, O2)
lines.append([full_pressures, Re_fO2, 'Re - ReO2', '-W1,black,.'])

Mo_fO2 = eqm_curve([Mo, MoO2], full_pressures, T, O2)
lines.append([full_pressures, Mo_fO2, 'Mo - MoO2', '-W1,black,.'])

fmq_metastable_fO2 = eqm_curve([fa, mt, q], full_pressures, T, O2)
lines.append([full_pressures, fmq_metastable_fO2, 'FMQ (metastable)', '-W1,blue,-'])

fiq_metastable_fO2 = eqm_curve([fa, iron, q], full_pressures, T, O2)
lines.append([full_pressures, fiq_metastable_fO2, 'FIQ (metastable)', '-W1,blue,-'])

# EMOG/EMOD
DG_pressure = optimize.fsolve(eqm_pressure, 6.e9, args=(T, [gph, diam], [1., -1.]))[0]
OW_pressure = optimize.fsolve(eqm_pressure, 6.e9, args=(T, [fo, mwd], [1., -1.]))[0]
WR_pressure = optimize.fsolve(eqm_pressure, 6.e9, args=(T, [mwd, mrw], [1., -1.]))[0]


EMOG_pressures = np.linspace(min_pressure, DG_pressure, 101)
EMOG_fO2 = eqm_curve(EMOG, EMOG_pressures, T, O2)
lines.append([EMOG_pressures, EMOG_fO2, 'EMOG', '-W1,red,-'])

EMOD_pressures = np.linspace(DG_pressure, OW_pressure, 101)
EMOD_fO2 = eqm_curve(EMOD, EMOD_pressures, T, O2)
lines.append([EMOD_pressures, EMOD_fO2, 'EMOD', '-W1,red,-'])

EMWD_pressures = np.linspace(OW_pressure, WR_pressure, 101)
EMWD_fO2 = eqm_curve(EMWD, EMWD_pressures, T, O2)
lines.append([EMWD_pressures, EMWD_fO2, 'EMWD', '-W1,red,-'])

EMRD_pressures = np.linspace(WR_pressure, max_pressure, 101)
EMRD_fO2 = eqm_curve(EMRD, EMRD_pressures, T, O2)
lines.append([EMRD_pressures, EMRD_fO2, 'EMRD', '-W1,red,-'])

####################################
# REACTIONS NOT INVOLVING WUESTITE #
####################################
pressure_Fe4O5_mt_hem = optimize.fsolve(eqm_pressure, 1.e9, args=(T, [mt, fe4o5, hem], [2., -1., -1.]))[0]
pressure_wus_Fe4O5_mt = optimize.fsolve(wus_eqm_c_P, [0.16, 1.e9], args=(T, wus, fe4o5, mt))[1]
pressure_wus_Fe5O6_Fe4O5 = optimize.fsolve(wus_eqm_c_P, [0.16, 1.e9], args=(T, wus, fe5o6, fe4o5))[1]

# Fe4O5 - mt bounded by wus and hem
fe4o5_mt_pressures=np.linspace(pressure_wus_Fe4O5_mt, pressure_Fe4O5_mt_hem, 21)
fe4o5_mt_fO2=eqm_curve([fe4o5, mt], fe4o5_mt_pressures, T, O2)
lines.append([fe4o5_mt_pressures, fe4o5_mt_fO2, 'Fe4O5 - mt', '-W1,black'])

# mt - hem bounded by lower limit and Fe4O5
mt_hem_pressures=np.linspace(min_pressure, pressure_Fe4O5_mt_hem, 21)
mt_hem_fO2=eqm_curve([mt, hem], mt_hem_pressures, T, O2)
lines.append([mt_hem_pressures, mt_hem_fO2, 'mt - hem', '-W1,black'])

# Fe5O6 - Fe4O5 bounded by wus and upper limit
fe5o6_fe4o5_pressures=np.linspace(pressure_wus_Fe5O6_Fe4O5, max_pressure, 21)
fe5o6_fe4o5_fO2=eqm_curve([fe5o6, fe4o5], fe5o6_fe4o5_pressures, T, O2)
lines.append([fe5o6_fe4o5_pressures, fe5o6_fe4o5_fO2, 'Fe5O6 - Fe4O5', '-W1,black'])

# Fe4O5 - hem bounded by mt and upper limit
fe4o5_hem_pressures=np.linspace(pressure_Fe4O5_mt_hem, max_pressure, 21)
fe4o5_hem_fO2=eqm_curve([fe4o5, hem], fe4o5_hem_pressures, T, O2)
lines.append([fe4o5_hem_pressures, fe4o5_hem_fO2, 'Fe4O5 - hem', '-W1,black'])

################################
# REACTIONS INVOLVING WUESTITE #
################################

# wus - mt bounded by lower limit and Fe4O5
wus_mt_pressures=np.linspace(min_pressure, pressure_wus_Fe4O5_mt, 21)
wus_mt_fO2=eqm_curve_wus(mt, wus, wus_mt_pressures, T, O2)
lines.append([wus_mt_pressures, wus_mt_fO2, 'wus - mt', '-W1,black'])

# wus - Fe4O5 bounded by mt and Fe5O6
wus_fe4o5_pressures=np.linspace(pressure_wus_Fe4O5_mt, pressure_wus_Fe5O6_Fe4O5, 21)
wus_fe4o5_fO2=eqm_curve_wus(fe4o5, wus, wus_fe4o5_pressures, T, O2)
lines.append([wus_fe4o5_pressures, wus_fe4o5_fO2, 'wus - Fe4O5', '-W1,black'])

# wus - Fe5O6 bounded by Fe4O5 and upper limit
wus_fe5o6_pressures=np.linspace(pressure_wus_Fe5O6_Fe4O5, max_pressure, 21)
wus_fe5o6_fO2=eqm_curve_wus(fe5o6, wus, wus_fe5o6_pressures, T, O2)
lines.append([wus_fe5o6_pressures, wus_fe5o6_fO2, 'wus - Fe5O6', '-W1,black'])

######################
# FMQ-LIKE REACTIONS #
######################
QC_pressure = optimize.fsolve(eqm_pressure, 1.e9, args=(T, [q, coe], [1., -1.]))[0]
FR_pressure = optimize.fsolve(eqm_pressure, 1.e9, args=(T, [fa, frw], [1., -1.]))[0]
CS_pressure = optimize.fsolve(eqm_pressure, 1.e9, args=(T, [coe, stv], [1., -1.]))[0]
MF_pressure = optimize.fsolve(eqm_pressure, 1.e9, 
                              args=(T, [frw, mt, fe4o5, stv], [1., 2., -2., -1.]))[0]
FF_pressure = optimize.fsolve(eqm_pressure, 1.e9, 
                              args=(T, [frw, fe4o5, fe5o6, stv], [1., 2., -2., -1.]))[0]
FFW_pressure = optimize.fsolve(wus_fe5o6_frw_stv_eqm, [0.01, 1.e9], args=(T))[1]
IFW_pressure = optimize.fsolve(wus_iron_frw_stv_eqm, [0.01, 1.e9], args=(T))[1]


fmq_pressures = np.linspace(min_pressure, QC_pressure, 21)
fmq_fO2=eqm_curve([fa, mt, q], fmq_pressures, T, O2)
lines.append([fmq_pressures, fmq_fO2, 'FMQ', '-W1,blue'])

fmc_pressures = np.linspace(QC_pressure, FR_pressure, 21)
fmc_fO2=eqm_curve([fa, mt, coe], fmc_pressures, T, O2)
lines.append([fmc_pressures, fmc_fO2, 'FMC', '-W1,blue'])

rmc_pressures = np.linspace(FR_pressure, CS_pressure, 21)
rmc_fO2=eqm_curve([frw, mt, coe], rmc_pressures, T, O2)
lines.append([rmc_pressures, rmc_fO2, 'RMC', '-W1,blue'])

rms_pressures = np.linspace(CS_pressure, MF_pressure, 21)
rms_fO2=eqm_curve([frw, mt, stv], rms_pressures, T, O2)
lines.append([rms_pressures, rms_fO2, 'RMS', '-W1,blue'])

rfs_pressures = np.linspace(MF_pressure, FF_pressure, 21)
rfs_fO2=eqm_curve([frw, fe4o5, stv], rfs_pressures, T, O2)
lines.append([rfs_pressures, rfs_fO2, 'RFS', '-W1,blue'])

rffs_pressures = np.linspace(FF_pressure, FFW_pressure, 21)
rffs_fO2=eqm_curve([frw, fe5o6, stv], rffs_pressures, T, O2)
lines.append([rffs_pressures, rffs_fO2, 'RFFS', '-W1,blue'])

rws_pressures = np.linspace(FFW_pressure, IFW_pressure, 21)
rws_fO2=eqm_curve_wus_2([frw, stv], wus, rws_pressures, T, O2)
lines.append([rws_pressures, rws_fO2, 'RWS', '-W1,blue'])

ris_pressures = np.linspace(CS_pressure, IFW_pressure, 21)
ris_fO2=eqm_curve([frw, iron, stv], ris_pressures, T, O2)
lines.append([ris_pressures, ris_fO2, 'RIS', '-W1,blue'])

ric_pressures = np.linspace(FR_pressure, CS_pressure, 21)
ric_fO2=eqm_curve([frw, iron, coe], ric_pressures, T, O2)
lines.append([ric_pressures, ric_fO2, 'RIC', '-W1,blue'])

fic_pressures = np.linspace(QC_pressure, FR_pressure, 21)
fic_fO2=eqm_curve([fa, iron, coe], fic_pressures, T, O2)
lines.append([fic_pressures, fic_fO2, 'FIC', '-W1,blue'])

fiq_pressures = np.linspace(min_pressure, QC_pressure, 21)
fiq_fO2=eqm_curve([fa, iron, q], fiq_pressures, T, O2)
lines.append([fiq_pressures, fiq_fO2, 'FIQ', '-W1,blue'])

# Now add lines corresponding to fa-frw, q-coe, coe-stv
lines.append([np.array([FR_pressure, FR_pressure]), np.array([ric_fO2[0], rmc_fO2[0]]), 'FR', '-W1,blue'])
lines.append([np.array([QC_pressure, QC_pressure]), np.array([fmc_fO2[0], 20.]), 'QC', '-W1,blue'])
lines.append([np.array([QC_pressure, QC_pressure]), np.array([fic_fO2[0], -20.]), 'QC', '-W1,blue'])
lines.append([np.array([CS_pressure, CS_pressure]), np.array([rms_fO2[0], 20.]), 'CS', '-W1,blue'])
lines.append([np.array([CS_pressure, CS_pressure]), np.array([ris_fO2[0], -20.]), 'CS', '-W1,blue'])

for line in lines:
    pressures, fO2s, name, marker = line
    plt.plot( pressures/1.e9, fO2s, '-', linewidth=3., label=name)

plt.title('')
plt.xlabel("Pressure (GPa)")
plt.ylabel("log10(fO2)")
plt.legend(loc='lower right')
plt.show()

f = open('P-fO2.dat', 'w')
for line in lines:
    pressures, fO2s, name, marker = line
    f.write('>> '+marker+'\n')
    for i, P in enumerate(pressures):
        f.write(str(P/1.e9)+' '+str(fO2s[i])+'\n')
f.write('\n')
f.close()

print 'P-fO2.dat (over)written'

new_phase_region = [[fe4o5_mt_pressures, fe4o5_mt_fO2],
                    [fe4o5_hem_pressures, fe4o5_hem_fO2],
                    [[24.e9], [4.]],
                    [wus_fe5o6_pressures[::-1], wus_fe5o6_fO2[::-1]],
                    [wus_fe4o5_pressures[::-1], wus_fe4o5_fO2[::-1]]]

f = open('new_phase_region.dat', 'w')
for boundary in new_phase_region:
    pressures, fO2s = boundary
    for i, P in enumerate(pressures):
        f.write(str(P/1.e9)+' '+str(fO2s[i])+'\n')
f.write('\n')
f.close()
print 'new_phase_region.dat (over)written'

# Required constraints
# 1. bulk composition (use composition of ol polymorph)
# 2. 2*MgFe3O5 + Fe2SiO4 <-> 2Fe4O5 + Mg2SiO4

fm45=MgFeFe2O5()
ol=olivine()
wad=wadsleyite()
rw=ringwoodite()


'''
ol=minerals.SLB_2011.mg_fe_olivine()
wad=minerals.SLB_2011.mg_fe_wadsleyite()
rw=minerals.SLB_2011.mg_fe_ringwoodite()
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

