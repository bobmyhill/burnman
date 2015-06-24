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


def density_crossover(pressure, temperature, phases, factor):
    phases[0].set_state(pressure, temperature)
    phases[1].set_state(pressure, temperature)
    return phases[0].V-phases[1].V*factor



per_liq = DKS_2013_liquids_tweaked.MgO_liquid()
per = SLB_2011.periclase()

per_liq.set_state(1.e5, 3070.)
per.set_state(1.e5, 3070.)
print per_liq.gibbs - per.gibbs

# STISHOVITE
f = open('data/stv_melting.dat', 'r')
datalines = [ line.strip().split() for idx, line in enumerate(f.read().split('\n')) if line.strip() and idx>0 ]

stv_P=[]
stv_Perr=[]
stv_T=[]
stv_Terr=[]
coe_P=[]
coe_Perr=[]
coe_T=[]
coe_Terr=[]
for content in datalines:
    if content[5] == 'stv' and content[4] != '4' :
        stv_P.append(float(content[0])*1.e9)
        stv_Perr.append(float(content[1])*1.e9)
        stv_T.append(float(content[2]))
        stv_Terr.append(float(content[3]))
    elif content[5] == 'coe' :
        coe_P.append(float(content[0])*1.e9)
        coe_Perr.append(float(content[1])*1.e9)
        coe_T.append(float(content[2]))
        coe_Terr.append(float(content[3]))
# NOTE
# The volumes of SiO2 melt appear to be too small. 
# If the entropies are correct, then
# dP/dT = DS/DV = (Smelt-Ssolid)/(Vmelt - Vsolid)
# The denominator on the RHS will be too small, thus dP/dT will be too big
# This is the right sign to explain the underestimates of melting temperature...
plt.errorbar(stv_P, stv_T, xerr=stv_Perr, yerr=stv_Terr, marker='o', linestyle='None')
plt.errorbar(coe_P, coe_T, xerr=coe_Perr, yerr=coe_Terr, marker='o', linestyle='None')

# Remember Schreinemakers - there should be no break in slope...
s0 = UnivariateSpline([13.7e9, 20.e9, 37.e9, 50.e9, 70.e9], [2800.+273.15, 3620., 4150., 4270., 4400.], s=1.)
s1 = UnivariateSpline([13.7e9, 20.e9, 37.e9, 50.e9, 70.e9], [2800.+273.15, 3670., 4250., 4370., 4500.], s=1.)
s2 = UnivariateSpline([13.7e9, 20.e9, 37.e9, 50.e9, 70.e9], [2800.+273.15, 3750., 4350., 4470., 4600.], s=1.)

pressures = np.linspace(13.7e9,80.e9, 200) 
plt.plot(pressures, s0(pressures), linewidth=1)
plt.plot(pressures, s1(pressures), linewidth=1)
plt.plot(pressures, s2(pressures), linewidth=1)

# Models

coe_SLB = SLB_2011.coesite()
stv_SLB = SLB_2011.stishovite()
stv = DKS_2013_solids.stishovite()
SiO2_liq = DKS_2013_liquids_tweaked.SiO2_liquid()

P_triple =  13.7e9
T_triple = 2800.+273.15
dP = 1.e5
dTdP0 = (s0(P_triple+dP) - s0(P_triple))/dP
dTdP1 = (s1(P_triple+dP) - s1(P_triple))/dP
dTdP2 = (s2(P_triple+dP) - s2(P_triple))/dP

coe_SLB.set_state(P_triple, T_triple)
stv_SLB.set_state(P_triple, T_triple)
SiO2_liq.set_state(P_triple, T_triple)
liq_V = coe_SLB.V
stv_V = stv_SLB.V
S_melting0 = (liq_V - stv_V)/dTdP0
S_melting1 = (liq_V - stv_V)/dTdP1
S_melting2 = (liq_V - stv_V)/dTdP2
print 'S_melt', S_melting0, S_melting1, S_melting2


SiO2_liq.set_state(0., 300.)
print 'V_0 check:', SiO2_liq.V, SiO2_liq.params['V_0']

SiO2_liq.set_state(1.e5, 1700.+273.15)
print 'Check at 1973 K:', SiO2_liq.V, '2.73e-5 m^3/mol, see Tomlinson et al., 1958'

phases = [coe_SLB, SiO2_liq, stv_SLB, stv]
print 'Triple point volumes ('+str(P_triple/1.e9), "GPa,", T_triple, "K)"
for phase in phases:
    phase.set_state(P_triple, T_triple)
    print phase.params['name'], phase.V   
print ''

def find_temperature(temperature, pressure, solid, liquid):
    liquid.set_state(pressure, temperature[0])
    solid.set_state(pressure, temperature[0])
    return solid.gibbs - liquid.gibbs



pressures = np.linspace(10.e9,80.e9, 20) 
temperatures = np.empty_like(pressures)
for i, pressure in enumerate(pressures):
    T_melt = fsolve(find_temperature, 3000., args=(pressure, stv_SLB, SiO2_liq))[0]
    temperatures[i] = T_melt
plt.plot(pressures, temperatures)

pressures = np.linspace(4.e9,16.e9, 10) 
temperatures = np.empty_like(pressures)
for i, pressure in enumerate(pressures):
    T_melt = fsolve(find_temperature, 4000., args=(pressure, coe_SLB, SiO2_liq))[0]
    temperatures[i] = T_melt
plt.plot(pressures, temperatures)


plt.xlim(4.e9, 80.e9)
plt.ylim(2000., 5000.)

plt.show()
    
'''
pressure = 14.e9
temperatures = np.linspace(2000.,7000., 11) 
for temperature in temperatures:
    dT = 1.
    SiO2_liq.set_state(pressure, temperature)
    stv_SLB.set_state(pressure, temperature)
    print temperature, SiO2_liq.C_p
'''

# FORSTERITE
fo=SLB_2011.forsterite()
fo_liq=DKS_2013_liquids.Mg2SiO4_liquid()

temperature=1.
fo_liq.set_state(1.e5, temperature)
print fo_liq.S
print fo_liq.method._bonding_entropy(temperature, fo_liq.V, fo_liq.params)
print fo_liq.method._electronic_excitation_entropy(temperature, fo_liq.V, fo_liq.params)


crossover_Ts = np.linspace(1800.+273.15, 2400.+273.15, 21)
crossover_Ps = np.empty_like(crossover_Ts) 
for i, temperature in enumerate(crossover_Ts):
    crossover_Ps[i] = fsolve(density_crossover, 20.e9, args=(temperature, [fo, fo_liq], 1.))[0]

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

# Remember Schreinemakers - there should be no break in slope...
s = UnivariateSpline(np.array(fo_liquidus_pressures), np.array(fo_liquidus_temperatures), s=1000.)

pressures = np.linspace(0., 20.e9, 101)
plt.plot(crossover_Ps, crossover_Ts, linewidth=1)
plt.plot(pressures, s(pressures), linewidth=1)
plt.plot(fo_liquidus_pressures, fo_liquidus_temperatures, marker='o', linestyle='None')
plt.plot(fo_solidus_pressures, fo_solidus_temperatures, marker='o', linestyle='None')
plt.show()

# Forsterite liquid entropy at 14 GPa
#dp/dT = ds/dV

p = 14.e9
dp = 1.e7
fo.set_state(p, s(p))
fo_liq.set_state(p, s(p))
dTdP = (s(p+dp) - s(p-dp)) / (2.*dp)
S_fo_melting = (fo_liq.V - fo.V)/dTdP
S_fo_liquid = S_fo_melting + fo.S
print 'Conditions:', p, s(p)
print 'Entropy of melting (fo):', S_fo_melting
print 'Melt entropy (Mg2SiO4):', S_fo_liquid
print 'Melt entropy (DKS2013):', fo_liq.S

# CLINOENSTATITE (Mg2Si2O6)
oen=SLB_2011.enstatite()
cen=SLB_2011.hp_clinoenstatite()
cen_liq=DKS_2013_liquids.MgSiO3_liquid()

crossover_Ts = np.linspace(1800.+273.15, 2400.+273.15, 21)
oen_crossover_Ps = np.empty_like(crossover_Ts) 
cen_crossover_Ps = np.empty_like(crossover_Ts) 
for i, temperature in enumerate(crossover_Ts):
    oen_crossover_Ps[i] = fsolve(density_crossover, 20.e9, args=(temperature, [oen, cen_liq], 2.))[0]
    cen_crossover_Ps[i] = fsolve(density_crossover, 20.e9, args=(temperature, [oen, cen_liq], 2.))[0]

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

oen_s = UnivariateSpline(np.array(oen_liquidus_pressures+cen_liquidus_pressures), np.array(oen_liquidus_temperatures+cen_liquidus_temperatures), s=2000.)



plt.plot(oen_crossover_Ps, crossover_Ts, linewidth=2)
plt.plot(cen_crossover_Ps, crossover_Ts, linewidth=1)

pressures = np.linspace(0., 20.e9, 101)
plt.plot(pressures, oen_s(pressures), linewidth=2)
plt.plot(oen_liquidus_pressures, oen_liquidus_temperatures, marker='o', linestyle='None')
plt.plot(cen_liquidus_pressures, cen_liquidus_temperatures, marker='o', linestyle='None')
plt.show()

# Mg2Si2O6 liquid entropy from cen at 14 GPa
#dp/dT = ds/dV

p = 14.e9
dp = 1.e7
oen.set_state(p, oen_s(p))
cen_liq.set_state(p, oen_s(p))
cen.set_state(p, oen_s(p))
dTdP = (oen_s(p+dp) - oen_s(p-dp)) / (2.*dp)
S_oen_melting = (2.*cen_liq.V - oen.V) / dTdP

# dP/dT for cen -> liquid (assuming same temperature, roughly true)
S_cen_melting = S_oen_melting + oen.S - cen.S
S_en_liquid = (S_cen_melting + cen.S)/2.
print 'Conditions:', p, oen_s(p)
print 'Entropy of melting (cen):', S_cen_melting
print 'Melt entropy (MgSiO3):', S_en_liquid
print 'Melt entropy (DKS2013):', cen_liq.S


per_liq = DKS_2013_liquids_tweaked.MgO_liquid()
per_liq.set_state(14.9, 2580.)
fo.set_state(14.9, 2580.)
cen.set_state(14.9, 2580.)
print 'Melt entropy (MgO):', per_liq.S
print 'Melt entropy (Mg2SiO4/3):', S_fo_liquid / 3.
print 'Melt entropy (MgSiO3/2):', S_en_liquid / 2.

print 'Melt gibbs (MgO, DKS):', per_liq.gibbs
print 'Melt gibbs (Mg2SiO4/3):', fo.gibbs / 3.
print 'Melt gibbs (MgSiO3/2):', cen.gibbs / 4.

plt.plot([0., 0.33, 0.5], [per_liq.gibbs, fo.gibbs / 3., cen.gibbs / 4.])
plt.show()




# Constraint from eutectic temperature
pressure = 13.9e9
temperature = 2185+273.15
c = 30.7 # +/-1.7 wt % Mg2SiO4 (with MgSiO3)
M_fo = 140.6931 # Mg2SiO4
M_en = 100.3887 # MgSiO3

nMg = 2.*(c/M_fo) + 1.*(100.-c)/M_en
nSi = 1.*(c/M_fo) + 1.*(100.-c)/M_en

c = nSi/(nMg + nSi)
print 'Composition (fraction SiO2):', c

fo.set_state(pressure, temperature)
cen.set_state(pressure, temperature)
per_liq.set_state(pressure, temperature)

mu_MgO = burnman.chemicalpotentials.chemical_potentials([fo, cen], [{'Mg':1., 'O':1.}])[0]


print mu_MgO, per_liq.gibbs



class MgO_SiO2_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Subregular MgO-SiO2 liquid'
        self.type='subregular'

        self.endmembers = [[DKS_2013_liquids_tweaked.MgO_liquid(), '[Mg]O'],
                           [DKS_2013_liquids_tweaked.SiO2_liquid(), '[Si]O2']]
                           

        self.enthalpy_interaction = [[[00000., -200000.]]]
        self.volume_interaction   = [[[0., 0.]]]
        self.entropy_interaction  = [[[0., 0.]]]
                        
        burnman.SolidSolution.__init__(self, molar_fractions)


per=SLB_2011.periclase()
fo=SLB_2011.forsterite()
en=SLB_2011.enstatite()
stv=SLB_2011.stishovite()

per.set_state(1.e5, 1000.)
print per.gibbs


# Melting checks
checks=[[per, 1.e5, 3070.],
        [fo, 1.e5, 2163.],
        [en, 1.e5, 1560.+273.15], # just unstable
        [stv, 14.e9, 3120.],
        [en, 13.e9, 2270.+273.15]]

liq=MgO_SiO2_liquid()


for check in checks:
    phase, pressure, temperature = check
    if 'Mg' in phase.params['formula']:
        nMg = phase.params['formula']['Mg']
    else:
        nMg = 0.
    if 'Si' in phase.params['formula']:
        nSi = phase.params['formula']['Si']
    else:
        nSi = 0.
    c = nSi/(nMg + nSi)
    liq.set_composition([1.-c, c])
    liq.set_state(pressure, temperature)
    phase.set_state(pressure, temperature)
    print phase.params['name'], liq.gibbs - phase.gibbs/(nSi+nMg)


# Liquidus
def find_temperature(temperature, pressure, liquid, solid, solid_composition):
    liquid.set_state(pressure, temperature[0])
    solid.set_state(pressure, temperature[0])
    solid_potential=burnman.chemicalpotentials.chemical_potentials([liq], solid_composition)[0]
    return solid.gibbs - solid_potential


def liquidus_temperatures(solid, liquid_compositions, pressure):
    temperatures=np.empty_like(liquid_compositions)
    for i, c in enumerate(liquid_compositions):
        liq.set_composition([1.-c, c])
        solid_composition=[solid.params['formula']]
        temperatures[i] = fsolve(find_temperature, 2000., args=(pressure, liq, solid, solid_composition))[0]
        print c, temperatures[i]
    return temperatures 


c_ranges=[[per, (0.10, 0.45, 10)],
          [fo, (0.25, 0.60, 10)],
          [en, (0.40, 0.70, 10)]]
          #[stv, (0.7, 1.0, 10)]]

pressure = 14.e9
for c_range in c_ranges:
    phase = c_range[0]
    print phase.params['name']
    liquid_compositions = np.linspace(c_range[1][0], c_range[1][1], c_range[1][2])
    temperatures = liquidus_temperatures(phase, liquid_compositions, pressure)
    plt.plot(liquid_compositions, temperatures, label=phase.params['name'])

plt.xlim(0.0, 1.0)
plt.show()



