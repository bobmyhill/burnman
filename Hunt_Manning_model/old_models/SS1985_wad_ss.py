#!/usr/python
import os, sys
import numpy as np
sys.path.insert(1,os.path.abspath('..'))

from scipy.optimize import fsolve, minimize, fmin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Liquid model
from models import *
from SS1985_functions import *
from H2O_eos import lnfH2O

# Benchmarks for the solid solution class
import burnman
from burnman.minerals import SLB_2011
from burnman import tools
from burnman.processchemistry import *
from burnman.chemicalpotentials import *


# 13 GPa, fo
r=4./3. # Oxygens available for bonding (one cation basis)
n_cations = 1.
Kinf = lambda T: 100000000000.
K0 = lambda T: 0.00000000001
K1 = lambda T:1 
G = lambda T: 0. - 75.*(T-1420.)
K = lambda T: np.exp(-(G(T))/(R*T))
Wsh = lambda P: 0. # -4.e-5 * (P-13.e9)
Whs = lambda P: 0. # -4.e-5 * (P-13.e9)


pressure_wad = 15.e9 # Pa
pressure_rw = 20.e9 # Pa
pressure_rw_2 = 23.e9 # Pa
wad=SLB_2011.mg_wadsleyite()
rw=SLB_2011.mg_ringwoodite()
liquid=MgO_SiO2_liquid()
liquid.set_composition([2./3., 1./3.])


Tmelt_wad = fsolve(delta_gibbs, 2000., args=(pressure_wad, wad, liquid, 1./3., 1.))[0]
print Tmelt_wad

print liquid.S, liquid.S*3. - wad.S

Tmelt_rw = fsolve(delta_gibbs, 2000., args=(pressure_rw, rw, liquid, 1./3., 1.))[0]
print Tmelt_rw

print liquid.S, liquid.S*3. - rw.S

compositions=np.linspace(0.0001, 0.99, 101)
Gex=np.empty_like(compositions)
Gex_2=np.empty_like(compositions)
for i, X in enumerate(compositions):
    T0 = 1000.
    Gex[i]=(1-X)*excesses_nonideal(X, T0, r, K(T0), Wsh(pressure_wad), Whs(pressure_wad))[0] + X*excesses_nonideal(X, T0, r, K(T0), Wsh(pressure_wad), Whs(pressure_wad))[1]
    Gex_2[i]=(1-X)*excesses_nonideal(X, Tmelt_wad, r, K(Tmelt_wad), Wsh(pressure_wad), Whs(pressure_wad))[0] \
        + X*excesses_nonideal(X, Tmelt_wad, r, K(Tmelt_wad), Wsh(pressure_wad), Whs(pressure_wad))[1]

plt.plot( compositions, Gex, '-', linewidth=2., label='model at '+str(T0)+' K')
plt.plot( compositions, Gex_2, '-', linewidth=2., label='model at Tmelt')
plt.ylabel("Excess Gibbs (J/mol)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()




fn0=0.
temperatures_wad=np.linspace(600., Tmelt_wad, 101)
compositions_wad=np.empty_like(temperatures_wad)
compositions0_wad=np.empty_like(temperatures_wad)
compositionsinf_wad=np.empty_like(temperatures_wad)

temperatures_rw=np.linspace(600., Tmelt_rw, 101)
compositions_rw=np.empty_like(temperatures_rw)
compositions0_rw=np.empty_like(temperatures_rw)
compositionsinf_rw=np.empty_like(temperatures_rw)


for i, T in enumerate(temperatures_wad):
    compositions0_wad[i]=fsolve(solve_composition, 0.001, args=(T, pressure_wad, r, K0, fn0, fn0, wad, liquid, 1./3., 1.))
    compositionsinf_wad[i]=fsolve(solve_composition, 0.001, args=(T, pressure_wad, r, Kinf, fn0, fn0, wad, liquid, 1./3., 1.))
    compositions_wad[i]=fsolve(solve_composition, 0.001, args=(T, pressure_wad, r, K, Wsh(pressure_wad), Whs(pressure_wad), wad, liquid, 1./3., 1.))


for i, T in enumerate(temperatures_rw):
    compositions0_rw[i]=fsolve(solve_composition, 0.001, args=(T, pressure_rw, r, K0, fn0, fn0, rw, liquid, 1./3., 1.))
    compositionsinf_rw[i]=fsolve(solve_composition, 0.001, args=(T, pressure_rw, r, Kinf, fn0, fn0, rw, liquid, 1./3., 1.))
    compositions_rw[i]=fsolve(solve_composition, 0.001, args=(T, pressure_rw, r, K, Wsh(pressure_rw), Whs(pressure_rw), rw, liquid, 1./3., 1.))


plt.plot( compositions_wad, temperatures_wad, linewidth=1, label='wad, 15 GPa')

'''
plt.plot( compositionsinf_wad, temperatures_wad, linewidth=1, label='K=inf, wad, P=15 GPa')
plt.plot( compositions0_wad, temperatures_wad, linewidth=1, label='K=0, wad, P=15 GPa')
'''

plt.plot( compositions_rw, temperatures_rw, linewidth=1, label='rw, 20 GPa')

'''
plt.plot( compositionsinf_rw, temperatures_rw, linewidth=1, label='K=inf, rw, P=20 GPa')
plt.plot( compositions0_rw, temperatures_rw, linewidth=1, label='K=0, rw, P=20 GPa')
'''
'''
###################
# CALCULATE LIQUIDUS SPLINE
from scipy.interpolate import UnivariateSpline

Xs=[0.0, 0.2, 0.4, 0.55]
Ts=[2300., 1830., 1500., 1280.] # in C (Presnall and Walter, 1993 for dry melting)
spline_PW1993 = UnivariateSpline(Xs, Ts, s=1)

Xs_liquidus = np.linspace(0.0, 0.6, 101)
plt.plot(Xs_liquidus, spline_PW1993(Xs_liquidus)+273.15)
###################

forsterite = []
enstatite=[]
chondrodite=[]
superliquidus=[]
for line in open('data/13GPa_fo-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[3] == 'f' or content[3] == 'sf' or content[3] == 'f_davide':
            forsterite.append([float(content[0])+273.15, (100. - float(content[1])*7./2.)/100.])
        if content[3] == 'e' or content[3] == 'se' or content[3] == 'e_davide':
            enstatite.append([float(content[0])+273.15, (100. - float(content[1])*7./2.)/100.])
        if content[3] == 'c':
            chondrodite.append([float(content[0])+273.15, (100. - float(content[1])*7./2.)/100.])
        if content[3] == 'l' or content[3] == 'l_davide':
            superliquidus.append([float(content[0])+273.15,(100. - float(content[1])*7./2.)/100.])

forsterite=np.array(zip(*forsterite))
enstatite=np.array(zip(*enstatite))
chondrodite=np.array(zip(*chondrodite))
superliquidus=np.array(zip(*superliquidus))
add_T = 50.
plt.plot( forsterite[1], forsterite[0]+add_T, marker='.', linestyle='none', label='fo+liquid')
plt.plot( enstatite[1], enstatite[0]+add_T, marker='.', linestyle='none', label='en+liquid')
plt.plot( chondrodite[1], chondrodite[0]+add_T+add_T, marker='.', linestyle='none', label='chond+liquid')
plt.plot( superliquidus[1], superliquidus[0]+add_T, marker='.', linestyle='none', label='superliquidus')
'''

##################
# Find partition coefficients for olivine, wadsleyite and ringwoodite 
# inputs: weight percent H2O for solid and liquid phases
##################
ohtani = np.array([[1300+273.15, 1370+273.15, 1370+273.15, 1450+273.15, ],
                   [2.6, 1.9, 1.9, 1.6],
                   [100., 68.5, 83.4, 38.6]])
demouchy = np.array([[900+273.15, 1000+273.15, 1100+273.15, 1200+273.15, 1300+273.15, 1400+273.15],
                     [2.23, 2.13, 2.41, 2.24, 1.66, 0.93],
                     [101., 101., 31.52, 28.62, 20.1, 12.]])

litasov = np.array([[1200+273.15, 1300+273.15, 1300+273.15, 1400+273.15, 1400+273.15],
                     [2.07, 1.02, 1.13, 0.58, 0.72],
                     [101., 24.2, 26.5, 13.3, 10.6]])

studynames=['Ohtani', 'Demouchy', 'Litasov']

wtH2O=18.02
wtfo=140.6931/3.
studies_mol = []
for j, study in enumerate([ohtani, demouchy, litasov]):
    study_mol = np.copy(study)
    for i, phase in enumerate(study[1]):
        study_mol[1][i]=(phase/wtH2O)/((phase/wtH2O) + ((100-phase)/wtfo))
    for i, phase in enumerate(study[2]):
        study_mol[2][i]=(phase/wtH2O)/((phase/wtH2O) + ((100-phase)/wtfo))
    studies_mol.append(study_mol)
        
    plt.plot( study_mol[1], study_mol[0], marker='o', linestyle='none', label=studynames[j])
    plt.plot( study_mol[2], study_mol[0], marker='o', linestyle='none', label=studynames[j])

demouchy_adjusted_mol = np.empty_like(demouchy[0])
ohtani_adjusted_mol = np.empty_like(ohtani[0])
for i, temperature in enumerate(demouchy[0]):
    demouchy_adjusted_mol[i] = fsolve(solve_composition, 0.001, args=(temperature, pressure_wad, r, K, Wsh(pressure_wad), Whs(pressure_wad), wad, liquid, 1./3., 1.))[0]
for i, temperature in enumerate(ohtani[0]):
    ohtani_adjusted_mol[i] = fsolve(solve_composition, 0.001, args=(temperature, pressure_rw, r, K, Wsh(pressure_rw), Whs(pressure_rw), rw, liquid, 1./3., 1.))[0]

#plt.plot( ohtani_adjusted, ohtani[0], marker='o', linestyle='none', label='Ohtani fit')
#plt.plot( demouchy_adjusted, demouchy[0], marker='o', linestyle='none', label='Demouchy fit')

demouchy_adjusted_wt = np.empty_like(demouchy[0])
ohtani_adjusted_wt = np.empty_like(ohtani[0])
for i, phase in enumerate(ohtani_adjusted_mol): # new liquid
    ohtani_adjusted_wt[i]=(phase*wtH2O)/((phase*wtH2O) + ((1.-phase)*wtfo))
for i, phase in enumerate(demouchy_adjusted_mol): # new liquid
    demouchy_adjusted_wt[i]=(phase*wtH2O)/((phase*wtH2O) + ((1.-phase)*wtfo))

print 'old partition coefficients'
print 'T:', demouchy[0], 'Dwad/melt:', demouchy[1]/demouchy[2]
print 'T:', ohtani[0], 'Drw/melt:', ohtani[1]/ohtani[2]

print 'new partition coefficients'
print 'T:', demouchy[0], 'Dwad/melt:', demouchy[1]/demouchy_adjusted_wt
print 'T:', ohtani[0], 'Drw/melt:', ohtani[1]/ohtani_adjusted_wt

# Take the average of these values, plot the new water contents
demouchy_D = np.mean(demouchy[1]/demouchy_adjusted_wt)
ohtani_D = np.mean(ohtani[1]/ohtani_adjusted_wt)

print 'Dwad/melt, Drw/melt, Dwad/rw'
print demouchy_D, ohtani_D, demouchy_D/ohtani_D

# let's see how much water there should be in ringwoodite and wadsleyite
compositions_wad_solid = np.empty_like(compositions_wad)
for i, c in enumerate(compositions_wad):
    wt_percent_melt = (c*wtH2O)/((c*wtH2O) + ((1.-c)*wtfo))
    wt_percent_solid = demouchy_D*wt_percent_melt
    w = wt_percent_solid
    compositions_wad_solid[i] = (w/wtH2O)/((w/wtH2O) + ((1.-w)/wtfo))

compositions_rw_solid = np.empty_like(compositions_rw)
for i, c in enumerate(compositions_rw):
    wt_percent_melt = (c*wtH2O)/((c*wtH2O) + ((1.-c)*wtfo))
    wt_percent_solid = ohtani_D*wt_percent_melt
    w = wt_percent_solid
    compositions_rw_solid[i] = (w/wtH2O)/((w/wtH2O) + ((1.-w)/wtfo))

plt.plot( compositions_wad_solid, temperatures_wad, linewidth=1, label='Wadsleyite')
plt.plot( compositions_rw_solid, temperatures_rw, linewidth=1, label='Ringwoodite')

plt.ylim(1000.,3000.)
plt.xlim(0.,1.)
plt.ylabel("Temperature (K)")
plt.xlabel("X")
plt.legend(loc='upper right')


##### 
# Now use activity of water in the melt to do the same thing
#####
deltaH_wadP = 160.e3 
deltaV_wad = 10.0e-6
n_wad = 1.
c1000 = 0.04 # mole fraction at 1000 K
c1000_wad = (c1000*wtH2O)/((c1000*wtH2O) + ((1.-c1000)*wtfo))
deltaH_wad = deltaH_wadP - deltaV_wad*pressure_wad
A_wad = c1000_wad \
    / (np.exp(lnfH2O(pressure_wad, 1000.)) \
           * np.exp(-(deltaH_wad + pressure_wad*deltaV_wad) \
                         / (constants.gas_constant*1000.)))

deltaH_rwP = 203.e3 
deltaV_rw = 10.0e-6
n_rw = 1.
c1000 = 0.048 # mole fraction at 1000 K
c1000_rw = (c1000*wtH2O)/((c1000*wtH2O) + ((1.-c1000)*wtfo))
deltaH_rw = deltaH_rwP - deltaV_rw*pressure_rw
A_rw = c1000_rw \
    / (np.exp(lnfH2O(pressure_rw, 1000.)) \
           * np.exp(-(deltaH_rw + pressure_rw*deltaV_rw) \
                         / (constants.gas_constant*1000.)))


# Print Dwad/rw at different temperatures
pressure = 18.e9 # roughly 520 km depth
temperatures = np.linspace(1200., 2200., 11)
for temperature in temperatures:
    print pressure/1.e9, temperature,  (A_wad*np.exp(-(deltaH_wad + pressure*deltaV_wad)/(constants.gas_constant*temperature))) /  (A_rw*np.exp(-(deltaH_rw + pressure*deltaV_rw)/(constants.gas_constant*temperature)))

    
compositions_wad_solid = np.empty_like(compositions_wad)
for i, c in enumerate(compositions_wad):
    a_H2O = activities(c, r, K(temperatures_wad[i]))[1] # H2O activity in the melt
    f_H2O = np.exp(lnfH2O(pressure_wad, temperatures_wad[i]))
    w = A_wad*np.power(a_H2O*f_H2O, n_wad)*np.exp(-(deltaH_wad + pressure_wad*deltaV_wad)/(constants.gas_constant*temperatures_wad[i]))
    compositions_wad_solid[i] = (w/wtH2O)/((w/wtH2O) + ((1.-w)/wtfo))

compositions_rw_solid = np.empty_like(compositions_rw)
for i, c in enumerate(compositions_rw):
    a_H2O = activities(c, r, K(temperatures_rw[i]))[1] # H2O activity in the melt
    f_H2O = np.exp(lnfH2O(pressure_rw, temperatures_rw[i]))
    w = A_rw*np.power(a_H2O*f_H2O, n_rw)*np.exp(-(deltaH_rw + pressure_rw*deltaV_rw)/(constants.gas_constant*temperatures_rw[i]))
    compositions_rw_solid[i] = (w/wtH2O)/((w/wtH2O) + ((1.-w)/wtfo))

compositions_rw_solid_2 = np.empty_like(compositions_rw)
for i, c in enumerate(compositions_rw):
    a_H2O = activities(c, r, K(temperatures_rw[i]))[1] # H2O activity in the melt
    f_H2O = np.exp(lnfH2O(pressure_rw_2, temperatures_rw[i]))
    w = A_rw*np.power(a_H2O*f_H2O, n_rw)*np.exp(-(deltaH_rw + pressure_rw_2*deltaV_rw)/(constants.gas_constant*temperatures_rw[i]))
    compositions_rw_solid_2[i] = (w/wtH2O)/((w/wtH2O) + ((1.-w)/wtfo))


plt.plot( compositions_wad_solid, temperatures_wad, linewidth=1, label='Wadsleyite (from activity)')
plt.plot( compositions_rw_solid, temperatures_rw, linewidth=1, label='Ringwoodite (from activity, 20 GPa)')
plt.plot( compositions_rw_solid_2, temperatures_rw, linewidth=1, label='Ringwoodite (from activity, 23 GPa)')



plt.ylim(1000.,3000.)
plt.xlim(0.,1.)
plt.ylabel("Temperature (K)")
plt.xlabel("X")
plt.legend(loc='upper right')
plt.show()

##########

##########

print 'Prefactors:', A_wad, A_rw

pressure = 18.e9 # 18 GPa is ~520 km
temperatures_Dwadrw = np.linspace(1273.15, 2273.15, 21)
Dwadmelt = np.empty_like(temperatures_Dwadrw)
Drwmelt = np.empty_like(temperatures_Dwadrw)
Dwadrw = np.empty_like(temperatures_Dwadrw)
for i, temperature in enumerate(temperatures_Dwadrw):
    c=fsolve(solve_composition, 0.001, args=(temperature, pressure, r, K, Wsh(pressure), Whs(pressure), wad, liquid, 1./3., 1.))[0] # melt composition

    melt_weight_percent= (c*wtH2O)/((c*wtH2O) + ((1.-c)*wtfo))

    a_H2O = activities(c, r, K(temperature))[1] # H2O activity in the melt
    f_H2O = np.exp(lnfH2O(pressure, temperature))
    
    wad_weight_percent = A_wad*np.power(a_H2O*f_H2O, n_wad)*np.exp(-(deltaH_wad + pressure*deltaV_wad)/(constants.gas_constant*temperature))
    rw_weight_percent = A_rw*np.power(a_H2O*f_H2O, n_rw)*np.exp(-(deltaH_rw + pressure*deltaV_rw)/(constants.gas_constant*temperature))
    
    Dwadrw[i] = (A_wad*np.exp(-(deltaH_wad + pressure*deltaV_wad)/(constants.gas_constant*temperature))) /  (A_rw*np.exp(-(deltaH_rw + pressure*deltaV_rw)/(constants.gas_constant*temperature)))
    Dwadrw[i] =  wad_weight_percent / rw_weight_percent
    Dwadmelt[i] = wad_weight_percent / melt_weight_percent
    Drwmelt[i] = rw_weight_percent / melt_weight_percent
    
    
    print pressure/1.e9, "GPa,", temperature-273.15, "C", Dwadrw[i]

plt.plot(temperatures_Dwadrw-273.15, Dwadrw)
plt.plot(temperatures_Dwadrw-273.15, Dwadmelt*10.)
plt.plot(temperatures_Dwadrw-273.15, Drwmelt*10.)
plt.xlabel('Temperatures (C)')
plt.ylabel('Dwad/rw')
plt.show()

##########
# This little aside looks at the amount of water in wadsleyite at 1200 C
# It is useful to estimate the deltaV of hydration
###########
temperature = 1200. + 273.15
demouchy_pressure = np.array([[14.e9, 14.e9, 15.e9, 16.e9, 17.e9, 18.e9],
                  [2.4, 2.68, 2.24, 2.60, 2.43, 1.24]])
weight_percents = np.empty_like(demouchy_pressure[0])
for i, pressure in enumerate(demouchy_pressure[0]):
    c = fsolve(solve_composition, 0.001, args=(temperature, pressure, r, K, Wsh(pressure), Whs(pressure), wad, liquid, 1./3., 1.))[0]
    a_H2O = activities(c, r, K(temperature))[1]
    f_H2O = np.exp(lnfH2O(pressure, temperature))
    weight_percents[i] = 100.*A_wad*np.power(a_H2O*f_H2O, n_wad)*np.exp(-(deltaH_wad + pressure*deltaV_wad)/(constants.gas_constant*temperature))
    print c, a_H2O, f_H2O, weight_percents[i]

plt.plot(demouchy_pressure[0], weight_percents, linewidth=1)
plt.plot(demouchy_pressure[0], demouchy_pressure[1], linestyle='None', marker='o')
plt.show()


#####

#############################################
# Print data
#############################################

model_filename='wad_rw_models.xT'
data=[['-W1,grey,-', compositions0_wad, temperatures_wad],  # wad melt at 15 GPa
      ['-W1,grey,-', compositions0_rw, temperatures_rw], # rw melt at 20 GPa
      ['-W1,grey', compositionsinf_wad, temperatures_wad],  # wad melt at 15 GPa
      ['-W1,grey', compositionsinf_rw, temperatures_rw], # rw melt at 20 GPa
      ['-W1,red,-', compositions_wad, temperatures_wad],  # wad melt at 15 GPa
      ['-W1,black,-', compositions_rw, temperatures_rw], # rw melt at 20 GPa
      ['-W1,red', compositions_wad_solid, temperatures_wad],
      ['-W1,black', compositions_rw_solid, temperatures_rw]]


f = open(model_filename,'w')
for datapair in data:
    linetype, compositions, temperatures=datapair
    f.write('>> '+str(linetype)+' \n')
    for i, X in enumerate(compositions):
        f.write( str(compositions[i])+' '+str(temperatures[i]-273.15)+'\n' ) # output in C
f.close()

model_filename='Ohtani_rw_data.xT'
data=[[studies_mol[0][1], studies_mol[0][0]],
      [studies_mol[0][2], studies_mol[0][0]]]


f = open(model_filename,'w')
for datapair in data:
    compositions, temperatures=datapair
    for i, X in enumerate(compositions):
        f.write( str(compositions[i])+' '+str(temperatures[i]-273.15)+'\n' ) # output in C
f.close()

model_filename='Demouchy_wad_data.xT'
data=[[studies_mol[1][1], studies_mol[1][0]],
      [studies_mol[1][2], studies_mol[1][0]]]


f = open(model_filename,'w')
for datapair in data:
    compositions, temperatures=datapair
    for i, X in enumerate(compositions):
        f.write( str(compositions[i])+' '+str(temperatures[i]-273.15)+'\n' ) # output in C
f.close()

model_filename='Litasov_wad_data.xT'
data=[[studies_mol[2][1], studies_mol[2][0]],
      [studies_mol[2][2], studies_mol[2][0]]]

f = open(model_filename,'w')
for datapair in data:
    compositions, temperatures=datapair
    for i, X in enumerate(compositions):
        f.write( str(compositions[i])+' '+str(temperatures[i]-273.15)+'\n' ) # output in C
f.close()




# And for the inset:
# Partition coefficients for wadsleyite, ringwoodite and melt.

model_filename='wad_rw_partitioning.TD'
data=[['-W1,black', temperatures_Dwadrw, Dwadrw]]


f = open(model_filename,'w')
for datapair in data:
    linetype, temperatures, Ds=datapair
    f.write( '>> '+linetype+'\n' ) # output in C
    for i, T in enumerate(temperatures):
        f.write( str(T-273.15)+' '+str(Ds[i])+'\n' ) # output in C
f.close()

model_filename='solid_melt_partitioning.TD'
data=[['-W1,red,-', temperatures_Dwadrw, Dwadmelt],
      ['-W1,black,-', temperatures_Dwadrw, Drwmelt]]


f = open(model_filename,'w')
for datapair in data:
    linetype, temperatures, Ds=datapair
    f.write( '>> '+linetype+'\n' ) # output in C
    for i, T in enumerate(temperatures):
        f.write( str(T-273.15)+' '+str(Ds[i])+'\n' ) # output in C
f.close()
