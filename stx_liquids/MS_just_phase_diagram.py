import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    DKS_2013_liquids_tweaked, \
    DKS_2013_liquids, \
    DKS_2013_solids, \
    SLB_2011, \
    HP_2011_ds62
from models import *
from burnman import constants
import numpy as np
import math
from scipy.optimize import fsolve, curve_fit
from scipy.interpolate import UnivariateSpline, interp1d, splrep, splev
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def find_temperature(temperature, pressure, phase1, phase2, factor):
    phase1.set_state(pressure, temperature[0])
    phase2.set_state(pressure, temperature[0])
    return phase1.gibbs*factor - phase2.gibbs

liquid = MgO_SiO2_liquid()

'''
#####
# Periclase
per=SLB_2011.periclase()
liquid.set_composition([1.0, 0.0])
P0 = 1.e5 
T0 = fsolve(find_temperature, 3069., args=(P0, per, liquid, 1.0))[0]
print T0

liquid.set_state(13.e9, 2000.)
print liquid.gibbs

P1 = 2.e8 
T1 = fsolve(find_temperature, 2500., args=(P1, per, liquid, 1.0))[0]
print T1
DeltaV = liquid.endmembers[0][0].V - per.V
DeltaS = liquid.endmembers[0][0].S - per.S
print DeltaS / DeltaV
print (P1-P0)/(T1-T0) 


T0 = 3070. # K
P0 = 1.e5 # Pa
P1 = 13.e9 
T1_used = fsolve(find_temperature, 3000., args=(P1, per, liquid, 1.0))[0]
DeltaV = liquid.endmembers[0][0].V - per.V
DeltaS = liquid.endmembers[0][0].S - per.S
dTdP_used = DeltaV / DeltaS 

print T1_used, DeltaS, liquid.endmembers[0][0].S

T1_ZF2008 = [T1_used, 5373.]
for T1 in T1_ZF2008:
    dTdP_scaled = dTdP_used * (T1 - T0) / (T1_used - T0)
    DeltaV = liquid.endmembers[0][0].V - per.V # assume constant with T above model melting point
    DeltaS = DeltaV / dTdP_scaled 
    print T1, DeltaS, DeltaV


exit()

######
'''


oen=SLB_2011.enstatite()
cen=SLB_2011.hp_clinoenstatite()

oen_pressures = np.linspace(2.e9, 18.e9, 20)
oen_temperatures = np.empty_like(oen_pressures)
for i, pressure in enumerate(oen_pressures):
    print pressure
    liquid.set_composition([0.5, 0.5]) 
    oen_temperatures[i] = fsolve(find_temperature, 2000., args=(pressure, oen, liquid, 0.25))[0]


cen_pressures = np.linspace(11.e9, 18.e9, 20)
cen_temperatures = np.empty_like(cen_pressures)
for i, pressure in enumerate(cen_pressures):
    print pressure
    liquid.set_composition([0.5, 0.5]) 
    cen_temperatures[i] = fsolve(find_temperature, 2000., args=(pressure, cen, liquid, 0.25))[0]

plt.plot(oen_pressures, oen_temperatures, linewidth=1)
plt.plot(cen_pressures, cen_temperatures, linewidth=1)

KK_en_liquidus_pressures = []
KK_en_liquidus_temperatures = []
f = open('data/Kato_Kumazawa_1986_en_melting.dat', 'r')
datalines = [ line.strip().split() for idx, line in enumerate(f.read().split('\n')) if line.strip() and idx>0 ]
for content in datalines:
    KK_en_liquidus_pressures.append(float(content[0])*1.e9)
    KK_en_liquidus_temperatures.append(float(content[1])+273.15)

plt.plot(KK_en_liquidus_pressures, KK_en_liquidus_temperatures, marker='o', linestyle='None')


oen_liquidus_pressures = []
oen_liquidus_temperatures = []
cen_liquidus_pressures = []
cen_liquidus_temperatures = []
f = open('data/Presnall_Gasparik_1990_en_melting.dat', 'r')
datalines = [ line.strip().split() for idx, line in enumerate(f.read().split('\n')) if line.strip() and idx>0 ]
for content in datalines:
    if content[2] == 'oen':
        oen_liquidus_pressures.append(float(content[0])*1.e9)
        oen_liquidus_temperatures.append(float(content[1])+273.15)
    elif content[2] == 'cen':
        cen_liquidus_pressures.append(float(content[0])*1.e9)
        cen_liquidus_temperatures.append(float(content[1])+273.15)

plt.plot(oen_liquidus_pressures, oen_liquidus_temperatures, marker='o', linestyle='None')
plt.plot(cen_liquidus_pressures, cen_liquidus_temperatures, marker='o', linestyle='None')
plt.show()

####################

fo=SLB_2011.forsterite()

fo_pressures = np.linspace(3.e9, 15.e9, 40)
fo_temperatures = np.empty_like(fo_pressures)
for i, pressure in enumerate(fo_pressures):
    print pressure
    liquid.set_composition([2./3, 1./3.]) 
    fo_temperatures[i] = fsolve(find_temperature, 2000., args=(pressure, fo, liquid, 1./3.))[0]

plt.plot(fo_pressures, fo_temperatures, linewidth=1)

fo_liquidus_pressures = []
fo_liquidus_temperatures = []
fo_liquidus_sigmaT=[]
fo_solidus_pressures = []
fo_solidus_temperatures = []
f = open('data/Presnall_Walter_1993_fo_melting.dat', 'r')
datalines = [ line.strip().split() for idx, line in enumerate(f.read().split('\n')) if line.strip() and idx>0 ]
for content in datalines:
    if content[3] == 'fo':
        fo_liquidus_pressures.append(float(content[0])*1.e9)
        fo_liquidus_temperatures.append(float(content[1])+273.15)
        fo_liquidus_sigmaT.append(float(content[2]))
    else:
        fo_solidus_pressures.append(float(content[0])*1.e9)
        fo_solidus_temperatures.append(float(content[1])+273.15)

plt.plot(fo_liquidus_pressures, fo_liquidus_temperatures, marker='o', linestyle='None')
plt.plot(fo_solidus_pressures, fo_solidus_temperatures, marker='o', linestyle='None')
plt.show()


#####
# PHASE DIAGRAM
#####

liquid = MgO_SiO2_liquid()
per=SLB_2011.periclase()
fo=SLB_2011.forsterite()
en=SLB_2011.hp_clinoenstatite()
stv=DKS_2013_liquids_tweaked.stishovite()
#stv=SLB_2011.stishovite()
coe=SLB_2011.coesite()


# Liquidus
def find_temperature(temperature, pressure, liquid, solid, solid_composition):
    liquid.set_state(pressure, temperature[0])
    solid.set_state(pressure, temperature[0])
    solid_potential=burnman.chemicalpotentials.chemical_potentials([liquid], solid_composition)[0]
    return solid.gibbs - solid_potential


def liquidus_temperatures(solid, liquid_compositions, pressure):
    temperatures=np.empty_like(liquid_compositions)
    for i, c in enumerate(liquid_compositions):
        liquid.set_composition([1.-c, c])
        solid_composition=[solid.params['formula']]
        temperatures[i] = fsolve(find_temperature, 3000., args=(pressure, liquid, solid, solid_composition))[0]
    return temperatures 


c_ranges=[[per, (0.0, 0.45, 10)],
          [fo, (0.25, 0.60, 8)],
          [en, (0.40, 0.70, 7)],
          [stv, (0.55, 1.0, 10)],
          [coe, (0.55, 1.0, 10)]]

pressure = 13.0e9

per_melt = liquidus_temperatures(per, [0.], pressure)[0] - 273.15
coe_melt = liquidus_temperatures(coe, [1.], pressure)[0] - 273.15
stv_melt = liquidus_temperatures(stv, [1.], pressure)[0] - 273.15
fo_melt = liquidus_temperatures(fo, [1./3.], pressure)[0] - 273.15
cen_melt=liquidus_temperatures(en, [1./2.], pressure)[0] - 273.15

splines = []
for c_range in c_ranges:
    phase = c_range[0]
    print phase.params['name']
    liquid_compositions = np.linspace(c_range[1][0], c_range[1][1], c_range[1][2])
    temperatures = liquidus_temperatures(phase, liquid_compositions, pressure)
    s = UnivariateSpline(liquid_compositions, temperatures-273.15, s=1.) # note change from K to C
    print liquid_compositions
    print s(liquid_compositions)
    splines.append(s)

def spline_cross(c, s0, s1):
    return s0(c) - s1(c)

# Find start and end points for each spline
cotectics=[]
cotectics.append([0.0, per_melt])
for i in range(4): # 5 splines, four crossing points
    c = fsolve(spline_cross, 0.1, args=(splines[i], splines[i+1]))[0]
    T = splines[i](c)
    cotectics.append([c, T])
cotectics.append([1.0, coe_melt])

liquidi = []
for i, s in enumerate(splines):
    phase = c_ranges[i][0]
    liquid_compositions = np.linspace(cotectics[i][0], cotectics[i+1][0], 10.)
    temperatures = s(liquid_compositions)
    liquidi.append([liquid_compositions, temperatures])
    plt.plot(liquid_compositions, temperatures, label=phase.params['name'])
    trunc_old = 0.
    print ''
    print phase.params['name']
    T_step = 50.
    for i, temperature in enumerate(temperatures):
        trunc = math.trunc(temperature/T_step)
        if trunc != trunc_old and trunc_old != 0 :
            print (float(trunc+trunc_old)/2. + 0.5)*T_step, liquid_compositions[i]
        trunc_old = trunc


# cotectics are 
# per
# per-fo
# fo-en
# en-stv
# stv-coe

# composition, then temperature

lines=[[[0.0, cotectics[1][0]], [cotectics[1][1], cotectics[1][1]]],
       [[0.0, 0.5], [cotectics[2][1], cotectics[2][1]]],
       [[1./3., 1./3.],[500., cotectics[2][1]]],
       [[0.5, 0.5],[500., cen_melt]],
       [[0.5, 1.0],[cotectics[3][1], cotectics[3][1]]],
       [[cotectics[4][0], 1.0],[cotectics[4][1], cotectics[4][1]]]]

for line in lines:
    plt.plot(line[0], line[1])

plt.xlim(0.0, 1.0)
plt.show()

# Output data 
outfiles=[]

data=[['-W1,black,-', oen_pressures/1.e9, oen_temperatures],
      ['-W1,black', cen_pressures/1.e9, cen_temperatures]]
outfiles.append(['enstatite_melting.PT', data])

data=[['-W1,black', fo_pressures/1.e9, fo_temperatures]]
outfiles.append(['forsterite_melting.PT', data])

liquidi
lines
data = []
for line in lines:
    data.append(['-W1,black', line[0], line[1]])
for liquidus in liquidi:
    data.append(['-W1,black', liquidus[0], liquidus[1]])
outfiles.append(['MS_melting_13GPa.xT', data])

for outfile in outfiles:
    model_filename, data = outfile

    f = open(model_filename,'w')
    for datapair in data:
        linetype, compositions, temperatures=datapair
        f.write('>> '+str(linetype)+' \n')
        for i, X in enumerate(compositions):
            f.write( str(compositions[i])+' '+str(temperatures[i]-273.15)+'\n' ) # output in C
    f.close()

