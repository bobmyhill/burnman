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


'''
liquid = MgO_SiO2_liquid()

cen=SLB_2011.hp_clinoenstatite()

pressures = np.linspace(3.e9, 15.e9, 7)
temperatures = np.empty_like(pressures)
for i, pressure in enumerate(pressures):
    print pressure
    liquid.set_composition([0.5, 0.5]) 
    temperatures[i] = fsolve(find_temperature, 2000., args=(pressure, cen, liquid, 0.25))[0]

plt.plot(pressures, temperatures, linewidth=1)

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

pressures = np.linspace(3.e9, 15.e9, 7)
temperatures = np.empty_like(pressures)
for i, pressure in enumerate(pressures):
    print pressure
    liquid.set_composition([2./3, 1./3.]) 
    temperatures[i] = fsolve(find_temperature, 2000., args=(pressure, fo, liquid, 1./3.))[0]

plt.plot(pressures, temperatures, linewidth=1)

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
'''

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

print 'per', liquidus_temperatures(per, [0.], pressure)[0] - 273.15
print 'coe', liquidus_temperatures(coe, [1.], pressure)[0] - 273.15
print 'stv', liquidus_temperatures(stv, [1.], pressure)[0] - 273.15
print 'fo', liquidus_temperatures(fo, [1./3.], pressure)[0] - 273.15
print 'hen', liquidus_temperatures(en, [1./2.], pressure)[0] - 273.15

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


for i, s in enumerate(splines):
    c_range = c_ranges[i]
    phase = c_range[0]
    liquid_compositions = np.linspace(c_range[1][0], c_range[1][1], c_range[1][2]*100.)
    temperatures = s(liquid_compositions)
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

def spline_cross(c, s0, s1):
    return s0(c) - s1(c)

c = fsolve(spline_cross, 0.1, args=(splines[0], splines[1]))[0]
T = splines[0](c)
print 'per-fo', c, T
c = fsolve(spline_cross, 0.4, args=(splines[1], splines[2]))[0]
T = splines[1](c)
print 'fo-en', c, T
c = fsolve(spline_cross, 0.5, args=(splines[2], splines[3]))[0]
T = splines[2](c)
print 'en-stv', c, T
c = fsolve(spline_cross, 0.7, args=(splines[3], splines[4]))[0]
T = splines[3](c)
print 'stv-coe', c, T

plt.xlim(0.0, 1.0)
plt.show()



