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
from SP1994_eos import lnfH2O

# Benchmarks for the solid solution class
import burnman
from burnman.minerals import SLB_2011
from burnman.minerals import DKS_2013_liquids
from burnman import tools
from burnman.processchemistry import *
from burnman.chemicalpotentials import *


# 13 GPa, fo
r=4./3. # Oxygens available for bonding (one cation basis)
n_cations = 1.
Kinf = lambda T: 100000000000.
K0 = lambda T: 0.00000000001
K1 = lambda T:1 
G = lambda T: 0. - 75.*(T-1200.)
K = lambda T: np.exp(-(G(T))/(R*T))
Wsh = lambda P: 0. # -4.e-5 * (P-13.e9)
Whs = lambda P: 0. # -4.e-5 * (P-13.e9)


pressure_fo = 12.e9 # Pa
pressure_fo_2 = 14.e9 # Pa
pressure_wad = 15.e9 # Pa
pressure_rw = 20.e9 # Pa
pressure_rw_2 = 23.e9 # Pa

pressure_410 = 14.e9
pressure_520 = 18.e9

fo=SLB_2011.forsterite()
wad=SLB_2011.mg_wadsleyite()
rw=SLB_2011.mg_ringwoodite()

liquid=DKS_2013_liquids.Mg2SiO4_liquid()

Pmelt = 13.e9
Tmelt = 2290. + 273.15
fo.set_state(Pmelt, Tmelt)
liquid.set_state(Pmelt, Tmelt)
liquid.params['a'][0][0] += fo.gibbs - liquid.gibbs


Tmelt_fo = tools.equilibrium_temperature([fo, liquid], [1., -1.], pressure_fo)
print Tmelt_fo

print liquid.S, liquid.S - fo.S

Tmelt_wad = tools.equilibrium_temperature([wad, liquid], [1., -1.], pressure_wad)
print Tmelt_wad

print liquid.S, liquid.S - wad.S

Tmelt_rw = tools.equilibrium_temperature([rw, liquid], [1., -1.], pressure_rw)
print Tmelt_rw

print liquid.S, liquid.S - rw.S

# First, find the compositions of melts
fn0=0.

temperatures_fo=np.linspace(600., Tmelt_fo, 101)
compositions_fo=np.empty_like(temperatures_fo)
compositions_fo_2=np.empty_like(temperatures_fo)
compositions0_fo=np.empty_like(temperatures_fo)
compositionsinf_fo=np.empty_like(temperatures_fo)

temperatures_wad=np.linspace(600., Tmelt_wad, 101)
compositions_wad=np.empty_like(temperatures_wad)
compositions0_wad=np.empty_like(temperatures_wad)
compositionsinf_wad=np.empty_like(temperatures_wad)

temperatures_rw=np.linspace(600., Tmelt_rw, 101)
compositions_rw=np.empty_like(temperatures_rw)
compositions_rw_2=np.empty_like(temperatures_rw)
compositions0_rw=np.empty_like(temperatures_rw)
compositionsinf_rw=np.empty_like(temperatures_rw)


for i, T in enumerate(temperatures_fo):
    compositions0_fo[i]=fsolve(solve_composition, 0.001, args=(T, pressure_fo, r, K0, fn0, fn0, fo, liquid, 1./3., 1./3.))
    compositionsinf_fo[i]=fsolve(solve_composition, 0.001, args=(T, pressure_fo, r, Kinf, fn0, fn0, fo, liquid, 1./3., 1./3.))
    compositions_fo[i]=fsolve(solve_composition, 0.001, args=(T, pressure_fo, r, K, Wsh(pressure_fo), Whs(pressure_fo), fo, liquid, 1./3., 1./3.))
    compositions_fo_2[i]=fsolve(solve_composition, 0.001, args=(T, pressure_fo_2, r, K, Wsh(pressure_fo_2), Whs(pressure_fo_2), fo, liquid, 1./3., 1./3.))

for i, T in enumerate(temperatures_wad):
    compositions0_wad[i]=fsolve(solve_composition, 0.001, args=(T, pressure_wad, r, K0, fn0, fn0, wad, liquid, 1./3., 1./3.))
    compositionsinf_wad[i]=fsolve(solve_composition, 0.001, args=(T, pressure_wad, r, Kinf, fn0, fn0, wad, liquid, 1./3., 1./3.))
    compositions_wad[i]=fsolve(solve_composition, 0.001, args=(T, pressure_wad, r, K, Wsh(pressure_wad), Whs(pressure_wad), wad, liquid, 1./3., 1./3.))

for i, T in enumerate(temperatures_rw):
    compositions0_rw[i]=fsolve(solve_composition, 0.001, args=(T, pressure_rw, r, K0, fn0, fn0, rw, liquid, 1./3., 1./3.))
    compositionsinf_rw[i]=fsolve(solve_composition, 0.001, args=(T, pressure_rw, r, Kinf, fn0, fn0, rw, liquid, 1./3., 1./3.))
    compositions_rw_2[i]=fsolve(solve_composition, 0.001, args=(T, pressure_rw_2, r, K, Wsh(pressure_rw_2), Whs(pressure_rw_2), rw, liquid, 1./3., 1./3.))
    compositions_rw[i]=fsolve(solve_composition, 0.001, args=(T, pressure_rw, r, K, Wsh(pressure_rw), Whs(pressure_rw), rw, liquid, 1./3., 1./3.))


plt.plot( compositions_fo, temperatures_fo, linewidth=1, label='fo, '+str(pressure_fo/1.e9)+' GPa')
plt.plot( compositions_wad, temperatures_wad, linewidth=1, label='wad, '+str(pressure_wad/1.e9)+' GPa')
plt.plot( compositions_rw, temperatures_rw, linewidth=1, label='rw, '+str(pressure_rw/1.e9)+' GPa')

##################
# Find partition coefficients for olivine, wadsleyite and ringwoodite 
# inputs: weight percent H2O for solid and liquid phases
##################
litasov_fo = np.array([[1200+273.15, 1400+273.15, 1600+273.15, 1800+273.15],
                       [0.537, 0.456, 0.113, 0.103],
                       [100., 100., 100., 100.]])

smyth = np.array([[1250+273.15, 1250+273.15, 1400+273.15, 1100+273.15, 1100+273.15, 1400+273.15, 1400+273.15, 1600+273.15, ],
                   [0.89, 0.85, 0.45, 0.577, 0.556, 0.44, 0.34, 0.1],
                   [100., 100., 100., 100., 100., 100., 100., 100.]])

demouchy = np.array([[900+273.15, 1000+273.15, 1100+273.15, 1200+273.15, 1300+273.15, 1400+273.15],
                     [2.23, 2.13, 2.41, 2.24, 1.66, 0.93],
                     [101., 101., 31.52, 28.62, 20.1, 12.]])

litasov = np.array([[1200+273.15, 1300+273.15, 1300+273.15, 1400+273.15, 1400+273.15],
                     [2.07, 1.02, 1.13, 0.58, 0.72],
                     [101., 24.2, 26.5, 13.3, 10.6]])

ohtani = np.array([[1300+273.15, 1370+273.15, 1370+273.15, 1450+273.15, ],
                   [2.6, 1.9, 1.9, 1.6],
                   [100., 68.5, 83.4, 38.6]])

studynames=['Litasov_fo', 'Smyth', 'Demouchy', 'Litasov', 'Ohtani']

wtH2O=18.02
wtfo=140.6931/3.
studies_mol = []
for j, study in enumerate([litasov_fo, smyth, demouchy, litasov, ohtani]):
    study_mol = np.copy(study)
    for i, phase in enumerate(study[1]):
        study_mol[1][i]=(phase/wtH2O)/((phase/wtH2O) + ((100-phase)/wtfo))
    for i, phase in enumerate(study[2]):
        study_mol[2][i]=(phase/wtH2O)/((phase/wtH2O) + ((100-phase)/wtfo))
    studies_mol.append(study_mol)
        
    plt.plot( study_mol[1], study_mol[0], marker='o', linestyle='none', label=studynames[j])
    plt.plot( study_mol[2], study_mol[0], marker='o', linestyle='none', label=studynames[j])

smyth_adjusted_mol = np.empty_like(smyth[0])
demouchy_adjusted_mol = np.empty_like(demouchy[0])
ohtani_adjusted_mol = np.empty_like(ohtani[0])
for i, temperature in enumerate(smyth[0]):
    smyth_adjusted_mol[i] = fsolve(solve_composition, 0.001, args=(temperature, pressure_fo, r, K, Wsh(pressure_fo), Whs(pressure_fo), fo, liquid, 1./3., 1./3.))[0]
for i, temperature in enumerate(demouchy[0]):
    demouchy_adjusted_mol[i] = fsolve(solve_composition, 0.001, args=(temperature, pressure_wad, r, K, Wsh(pressure_wad), Whs(pressure_wad), wad, liquid, 1./3., 1./3.))[0]
for i, temperature in enumerate(ohtani[0]):
    ohtani_adjusted_mol[i] = fsolve(solve_composition, 0.001, args=(temperature, pressure_rw, r, K, Wsh(pressure_rw), Whs(pressure_rw), rw, liquid, 1./3., 1./3.))[0]


smyth_adjusted_wt = np.empty_like(smyth[0])
demouchy_adjusted_wt = np.empty_like(demouchy[0])
ohtani_adjusted_wt = np.empty_like(ohtani[0])
for i, phase in enumerate(smyth_adjusted_mol): # new liquid
    smyth_adjusted_wt[i]=(phase*wtH2O)/((phase*wtH2O) + ((1.-phase)*wtfo))
for i, phase in enumerate(demouchy_adjusted_mol): # new liquid
    demouchy_adjusted_wt[i]=(phase*wtH2O)/((phase*wtH2O) + ((1.-phase)*wtfo))
for i, phase in enumerate(ohtani_adjusted_mol): # new liquid
    ohtani_adjusted_wt[i]=(phase*wtH2O)/((phase*wtH2O) + ((1.-phase)*wtfo))

print 'old partition coefficients'
print 'T:', smyth[0], 'Dfo/melt:', smyth[1]/smyth[2]
print 'T:', demouchy[0], 'Dwad/melt:', demouchy[1]/demouchy[2]
print 'T:', ohtani[0], 'Drw/melt:', ohtani[1]/ohtani[2]

print 'new partition coefficients'
print 'T:', smyth[0], 'Dfo/melt:', smyth[1]/smyth_adjusted_wt
print 'T:', demouchy[0], 'Dwad/melt:', demouchy[1]/demouchy_adjusted_wt
print 'T:', ohtani[0], 'Drw/melt:', ohtani[1]/ohtani_adjusted_wt

# Take the average of these values, plot the new water contents
smyth_D = np.mean(smyth[1]/smyth_adjusted_wt)
demouchy_D = np.mean(demouchy[1]/demouchy_adjusted_wt)
ohtani_D = np.mean(ohtani[1]/ohtani_adjusted_wt)

print 'Dwad/melt, Dfo/melt, Dfo/wad'
print demouchy_D, smyth_D, smyth_D/demouchy_D

print 'Dwad/melt, Drw/melt, Dwad/rw'
print demouchy_D, ohtani_D, demouchy_D/ohtani_D

# let's see how much water there should be in 
# forsterite, wadsleyite and ringwoodite 


compositions_fo_solid = np.empty_like(compositions_fo)
for i, c in enumerate(compositions_fo):
    wt_percent_melt = (c*wtH2O)/((c*wtH2O) + ((1.-c)*wtfo))
    wt_percent_solid = smyth_D*wt_percent_melt
    w = wt_percent_solid
    compositions_fo_solid[i] = (w/wtH2O)/((w/wtH2O) + ((1.-w)/wtfo))

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

plt.plot( compositions_fo_solid, temperatures_fo, linewidth=1, label='Forsterite')
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
deltaH_foP = 150.e3 
deltaV_fo = 10.0e-6 
n_fo = 1.
c1673_fo = 0.024 # wt fraction at 1400 C if pure water
print (c1673_fo/wtH2O)/((c1673_fo/wtH2O) + ((1.-c1673_fo)/wtfo))
deltaH_fo = deltaH_foP - deltaV_fo*pressure_fo
A_fo = c1673_fo \
    / (np.exp(lnfH2O(pressure_fo, 1673.)) \
           * np.exp(-(deltaH_fo + pressure_fo*deltaV_fo) \
                         / (constants.gas_constant*1673.)))

# Zhao et al., 2004
#A_fo = 5.e-12
#deltaH_fo = 170.e3 
#deltaV_fo = 10.e-6


deltaH_wadP = 165.e3 
deltaV_wad = 10.0e-6
n_wad = 1.
c1000 = 0.050 # mole fraction at 1000 K
c1000_wad = (c1000*wtH2O)/((c1000*wtH2O) + ((1.-c1000)*wtfo))
deltaH_wad = deltaH_wadP - deltaV_wad*pressure_wad
A_wad = c1000_wad \
    / (np.exp(lnfH2O(pressure_wad, 1000.)) \
           * np.exp(-(deltaH_wad + pressure_wad*deltaV_wad) \
                         / (constants.gas_constant*1000.)))

deltaH_rwP = 200.e3 
deltaV_rw = 10.0e-6
n_rw = 1.
c1000 = 0.09 # mole fraction at 1000 K
c1000_rw = (c1000*wtH2O)/((c1000*wtH2O) + ((1.-c1000)*wtfo))
deltaH_rw = deltaH_rwP - deltaV_rw*pressure_rw
A_rw = c1000_rw \
    / (np.exp(lnfH2O(pressure_rw, 1000.)) \
           * np.exp(-(deltaH_rw + pressure_rw*deltaV_rw) \
                         / (constants.gas_constant*1000.)))

print 'Preexponential terms:'
print 'Pressures (GPa)'
print pressure_fo/1.e9, pressure_wad/1.e9, pressure_rw/1.e9
print 'A_fo, A_wad, A_rw'
print A_fo, A_wad, A_rw

# Dwad/fo at 410 km
temperatures = np.linspace(1200., 2200., 11)
for temperature in temperatures:
    print pressure_410/1.e9, temperature,  (A_wad*np.exp(-(deltaH_wad + pressure_410*deltaV_wad)/(constants.gas_constant*temperature))) /  (A_fo*np.exp(-(deltaH_fo + pressure_410*deltaV_fo)/(constants.gas_constant*temperature)))

# Dwad/rw at 520 km
for temperature in temperatures:
    print pressure_520/1.e9, temperature,  (A_wad*np.exp(-(deltaH_wad + pressure_520*deltaV_wad)/(constants.gas_constant*temperature))) /  (A_rw*np.exp(-(deltaH_rw + pressure_520*deltaV_rw)/(constants.gas_constant*temperature)))


compositions_fo_solid = np.empty_like(compositions_fo)
for i, c in enumerate(compositions_fo):
    a_H2O = activities(c, r, K(temperatures_fo[i]))[1] # H2O activity in the melt
    f_H2O = np.exp(lnfH2O(pressure_fo, temperatures_fo[i]))
    w = A_fo*np.power(a_H2O*f_H2O, n_fo)*np.exp(-(deltaH_fo + pressure_fo*deltaV_fo)/(constants.gas_constant*temperatures_fo[i]))
    compositions_fo_solid[i] = (w/wtH2O)/((w/wtH2O) + ((1.-w)/wtfo))

compositions_fo_solid_2 = np.empty_like(compositions_fo)
for i, c in enumerate(compositions_fo_2):
    a_H2O = activities(c, r, K(temperatures_fo[i]))[1] # H2O activity in the melt
    f_H2O = np.exp(lnfH2O(pressure_fo_2, temperatures_fo[i]))
    w = A_fo*np.power(a_H2O*f_H2O, n_fo)*np.exp(-(deltaH_fo + pressure_fo_2*deltaV_fo)/(constants.gas_constant*temperatures_fo[i]))
    compositions_fo_solid_2[i] = (w/wtH2O)/((w/wtH2O) + ((1.-w)/wtfo))

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
for i, c in enumerate(compositions_rw_2):
    a_H2O = activities(c, r, K(temperatures_rw[i]))[1] # H2O activity in the melt
    f_H2O = np.exp(lnfH2O(pressure_rw_2, temperatures_rw[i]))
    w = A_rw*np.power(a_H2O*f_H2O, n_rw)*np.exp(-(deltaH_rw + pressure_rw_2*deltaV_rw)/(constants.gas_constant*temperatures_rw[i]))
    compositions_rw_solid_2[i] = (w/wtH2O)/((w/wtH2O) + ((1.-w)/wtfo))



plt.plot( compositions_fo_solid, temperatures_fo, linewidth=1, label='Forsterite (from activity, '+str(pressure_fo/1.e9)+' GPa)')
plt.plot( compositions_fo_solid_2, temperatures_fo, linewidth=1, label='Forsterite (from activity, '+str(pressure_fo_2/1.e9)+' GPa)')
plt.plot( compositions_wad_solid, temperatures_wad, linewidth=1, label='Wadsleyite (from activity'+str(pressure_wad/1.e9)+' GPa)')
plt.plot( compositions_rw_solid, temperatures_rw, linewidth=1, label='Ringwoodite (from activity, '+str(pressure_rw/1.e9)+' GPa)')
plt.plot( compositions_rw_solid_2, temperatures_rw, linewidth=1, label='Ringwoodite (from activity, '+str(pressure_rw_2/1.e9)+' GPa)')


plt.ylim(1000.,3000.)
#plt.xlim(0.,0.1)
plt.ylabel("Temperature (K)")
plt.xlabel("X")
plt.legend(loc='upper right')
plt.show()

##########
# 410 discontinuity
##########

temperatures_Dwadfo_410 = np.linspace(1273.15, 2273.15, 21)
Dwadmelt_410 = np.empty_like(temperatures_Dwadfo_410)
Dfomelt_410 = np.empty_like(temperatures_Dwadfo_410)
Dwadfo_410 = np.empty_like(temperatures_Dwadfo_410)
for i, temperature in enumerate(temperatures_Dwadfo_410):
    c=fsolve(solve_composition, 0.001, args=(temperature, pressure_410, r, K, Wsh(pressure_410), Whs(pressure_410), wad, liquid, 1./3., 1./3.))[0] # melt composition

    melt_weight_percent= (c*wtH2O)/((c*wtH2O) + ((1.-c)*wtfo))

    a_H2O = activities(c, r, K(temperature))[1] # H2O activity in the melt
    f_H2O = np.exp(lnfH2O(pressure_410, temperature))
    
    wad_weight_percent = A_wad*np.power(a_H2O*f_H2O, n_wad)*np.exp(-(deltaH_wad + pressure_410*deltaV_wad)/(constants.gas_constant*temperature))
    fo_weight_percent = A_fo*np.power(a_H2O*f_H2O, n_fo)*np.exp(-(deltaH_fo + pressure_410*deltaV_fo)/(constants.gas_constant*temperature))
    
    Dwadfo_410[i] = (A_wad*np.exp(-(deltaH_wad + pressure_410*deltaV_wad)/(constants.gas_constant*temperature))) / (A_fo*np.exp(-(deltaH_fo + pressure_410*deltaV_fo)/(constants.gas_constant*temperature)))
    Dwadfo_410[i] =  wad_weight_percent / fo_weight_percent 
    Dwadmelt_410[i] = wad_weight_percent / melt_weight_percent
    Dfomelt_410[i] = fo_weight_percent / melt_weight_percent
    
    
    print pressure_410/1.e9, "GPa,", temperature-273.15, "C", Dwadfo_410[i]

plt.plot(temperatures_Dwadfo_410-273.15, Dwadfo_410)
plt.plot(temperatures_Dwadfo_410-273.15, Dfomelt_410*10.)
plt.plot(temperatures_Dwadfo_410-273.15, Dwadmelt_410*10.)
plt.xlabel('Temperatures (C)')
plt.ylabel('Dfo/wad')
plt.show()


##########
# 520 discontinuity
##########

temperatures_Drwwad_520 = np.linspace(1273.15, 2273.15, 21)
Dwadmelt_520 = np.empty_like(temperatures_Drwwad_520)
Drwmelt_520 = np.empty_like(temperatures_Drwwad_520)
Drwwad_520 = np.empty_like(temperatures_Drwwad_520)
for i, temperature in enumerate(temperatures_Drwwad_520):
    c=fsolve(solve_composition, 0.001, args=(temperature, pressure_520, r, K, Wsh(pressure_520), Whs(pressure_520), wad, liquid, 1./3., 1./3.))[0] # melt composition

    melt_weight_percent= (c*wtH2O)/((c*wtH2O) + ((1.-c)*wtfo))

    a_H2O = activities(c, r, K(temperature))[1] # H2O activity in the melt
    f_H2O = np.exp(lnfH2O(pressure_520, temperature))
    
    wad_weight_percent = A_wad*np.power(a_H2O*f_H2O, n_wad)*np.exp(-(deltaH_wad + pressure_520*deltaV_wad)/(constants.gas_constant*temperature))
    rw_weight_percent = A_rw*np.power(a_H2O*f_H2O, n_rw)*np.exp(-(deltaH_rw + pressure_520*deltaV_rw)/(constants.gas_constant*temperature))
    
    Drwwad_520[i] = (A_rw*np.exp(-(deltaH_rw + pressure_520*deltaV_rw)/(constants.gas_constant*temperature))) / (A_wad*np.exp(-(deltaH_wad + pressure_520*deltaV_wad)/(constants.gas_constant*temperature)))
    Drwwad_520[i] =  rw_weight_percent / wad_weight_percent 
    Dwadmelt_520[i] = wad_weight_percent / melt_weight_percent
    Drwmelt_520[i] = rw_weight_percent / melt_weight_percent
    
    
    print pressure_520/1.e9, "GPa,", temperature-273.15, "C", Drwwad_520[i]

plt.plot(temperatures_Drwwad_520-273.15, Drwwad_520)
plt.plot(temperatures_Drwwad_520-273.15, Dwadmelt_520*10.)
plt.plot(temperatures_Drwwad_520-273.15, Drwmelt_520*10.)
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
    c = fsolve(solve_composition, 0.001, args=(temperature, pressure, r, K, Wsh(pressure), Whs(pressure), wad, liquid, 1./3., 1./3.))[0]
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
# Single segment xy files

outfiles = []
data=[[studies_mol[0][1], studies_mol[0][0]],
      [studies_mol[0][2], studies_mol[0][0]]]
outfiles.append(['Litasov_fo_data.xT', data])

data=[[studies_mol[1][1], studies_mol[1][0]],
      [studies_mol[1][2], studies_mol[1][0]]]
outfiles.append(['Smyth_fo_data.xT', data])

data=[[studies_mol[2][1], studies_mol[2][0]],
      [studies_mol[2][2], studies_mol[2][0]]]
outfiles.append(['Demouchy_wad_data.xT', data])

data=[[studies_mol[3][1], studies_mol[3][0]],
      [studies_mol[3][2], studies_mol[3][0]]]
outfiles.append(['Litasov_wad_data.xT', data])

data=[[studies_mol[4][1], studies_mol[4][0]],
      [studies_mol[4][2], studies_mol[4][0]]]
outfiles.append(['Ohtani_rw_data.xT', data])

for line in outfiles:
    model_filename, data = line
    f = open(model_filename,'w')
    for datapair in data:
        compositions, temperatures=datapair
        for i, T in enumerate(temperatures):
            f.write( str(compositions[i])+' '+str(T - 273.15)+'\n' ) # output in C
    f.close()

# Multisegment xy files
outfiles = []

# Compositions for forsterite, wadsleyite and melt.
data=[['-W1,grey,-', compositions0_fo, temperatures_fo], # fo melt at 12 GPa
      ['-W1,grey,-', compositions0_wad, temperatures_wad],  # wad melt at 15 GPa
      ['-W1,grey,-', compositions0_rw, temperatures_rw], # rw melt at 20 GPa
      ['-W1,grey', compositionsinf_fo, temperatures_fo], # fo melt at 12 GPa
      ['-W1,grey', compositionsinf_wad, temperatures_wad],  # wad melt at 15 GPa
      ['-W1,grey', compositionsinf_rw, temperatures_rw], # rw melt at 20 GPa
      ['-W1,blue', compositions_fo, temperatures_fo], # fo melt at 12 GPa
      ['-W1,red', compositions_wad, temperatures_wad],  # wad melt at 15 GPa
      ['-W1,black', compositions_rw, temperatures_rw], # fo melt at 12 GPa
      ['-W1,blue', compositions_fo_solid, temperatures_fo],
      ['-W1,red', compositions_wad_solid, temperatures_wad],
      ['-W1,black', compositions_rw_solid, temperatures_rw]]

outfiles.append(['fo_wad_rw_models.xT', data])

for line in outfiles:
    model_filename, data = line
    f = open(model_filename,'w')
    for datapair in data:
        linetype, compositions, temperatures=datapair
        f.write( '>> '+linetype+'\n' ) # output in C
        for i, T in enumerate(temperatures):
            f.write( str(compositions[i])+' '+str(T - 273.15)+'\n' ) # output in C
    f.close()

# Partition coefficients for forsterite, wadsleyite, ringwoodite and melt.
outfiles = []
data=[['-W1,black', temperatures_Dwadfo_410, Dwadfo_410]]
outfiles.append(['wad_fo_partitioning_410.TD', data])

data=[['-W1,red,-', temperatures_Dwadfo_410, Dwadmelt_410],
      ['-W1,blue,-', temperatures_Dwadfo_410, Dfomelt_410]]
outfiles.append(['wad_fo_melt_partitioning_410.TD', data])

data=[['-W1,black', temperatures_Drwwad_520, Drwwad_520]]
outfiles.append(['rw_wad_partitioning_520.TD', data])

data=[['-W1,red,-', temperatures_Drwwad_520, Dwadmelt_520],
      ['-W1,black,-', temperatures_Drwwad_520, Drwmelt_520]]
outfiles.append(['rw_wad_melt_partitioning_520.TD', data])


for line in outfiles:
    model_filename, data = line
    f = open(model_filename,'w')
    for datapair in data:
        linetype, temperatures, Ds=datapair
        f.write( '>> '+linetype+'\n' ) # output in C
        for i, T in enumerate(temperatures):
            f.write( str(T-273.15)+' '+str(Ds[i])+'\n' ) # output in C
    f.close()
