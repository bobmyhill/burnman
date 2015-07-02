import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    DKS_2013_liquids_tweaked, \
    DKS_2013_liquids, \
    DKS_2013_solids, \
    SLB_2011, \
    HP_2011_ds62
from burnman import constants
import numpy as np
from scipy.optimize import fsolve, curve_fit
from scipy.interpolate import UnivariateSpline, interp1d, splrep, splev
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def find_temperature(temperature, pressure, phase1, phase2, factor):
    phase1.set_state(pressure, temperature[0])
    phase2.set_state(pressure, temperature[0])
    return phase1.gibbs*factor - phase2.gibbs

class MgO_SiO2_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Subregular MgO-SiO2 liquid'
        self.type='subregular'

        self.endmembers = [[DKS_2013_liquids_tweaked.MgO_liquid(), '[Mg]O'],
                           [DKS_2013_liquids_tweaked.SiO2_liquid(), '[Si]O2']]

        self.enthalpy_interaction = [[[-87653.77 + 14.e9*3.2e-6, -191052. + 14.e9*+3.2e-6]]] # 14 GPa
        self.volume_interaction   = [[[-3.2e-6, -3.2e-6]]]
        self.entropy_interaction  = [[[0., 0.]]]
                        
        burnman.SolidSolution.__init__(self, molar_fractions)

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


#####
# PHASE DIAGRAM
#####

per=SLB_2011.periclase()
fo=SLB_2011.forsterite()
en=SLB_2011.enstatite()
stv=DKS_2013_liquids_tweaked.stishovite()
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
        temperatures[i] = fsolve(find_temperature, 2000., args=(pressure, liquid, solid, solid_composition))[0]
        print c, temperatures[i]
    return temperatures 


c_ranges=[[per, (0.10, 0.45, 10)],
          [fo, (0.25, 0.60, 10)],
          [en, (0.40, 0.70, 10)],
          [coe, (0.55, 1.0, 10)],
          [stv, (0.55, 1.0, 10)]]

pressure = 14.e9
for c_range in c_ranges:
    phase = c_range[0]
    print phase.params['name']
    liquid_compositions = np.linspace(c_range[1][0], c_range[1][1], c_range[1][2])
    temperatures = liquidus_temperatures(phase, liquid_compositions, pressure)
    plt.plot(liquid_compositions, temperatures, label=phase.params['name'])

plt.xlim(0.0, 1.0)
plt.show()


