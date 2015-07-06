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


def find_temperature(temperature, pressure, solid, liquid):
    liquid.set_state(pressure, temperature[0])
    solid.set_state(pressure, temperature[0])
    return solid.gibbs - liquid.gibbs

def find_temperature_mul(temperature, pressure, solid, liquid, factor):
    liquid.set_state(pressure, temperature[0])
    solid.set_state(pressure, temperature[0])
    return solid.gibbs - (liquid.gibbs)*factor

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
stv_SLB = DKS_2013_liquids_tweaked.stishovite()
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

print 'Clapeyron slope', dTdP1

S_melting0 = (liq_V - stv_V)/dTdP0
S_melting1 = (liq_V - stv_V)/dTdP1
S_melting2 = (liq_V - stv_V)/dTdP2
print 'S_melt', S_melting0, S_melting1, S_melting2


SiO2_liq.set_state(0., 300.)
print 'V_0 check:', SiO2_liq.V, SiO2_liq.params['V_0']

SiO2_liq.set_state(-2718028.7506/1e3, 3000.)
print SiO2_liq.V, SiO2_liq.pressure

SiO2_liq.set_state(1.e5, 1700.+273.15)
print 'Check at 1973 K:', SiO2_liq.V, '2.73e-5 m^3/mol, see Tomlinson et al., 1958'

phases = [coe_SLB, SiO2_liq, stv_SLB, stv]
print 'Triple point volumes ('+str(P_triple/1.e9), "GPa,", T_triple, "K)"
for phase in phases:
    phase.set_state(P_triple, T_triple)
    print phase.params['name'], phase.V   
print ''




pressures = np.linspace(10.e9,200.e9, 20) 
temperatures = np.empty_like(pressures)
for i, pressure in enumerate(pressures):
    T_melt = fsolve(find_temperature, 3000., args=(pressure, stv_SLB, SiO2_liq))[0]
    temperatures[i] = T_melt
plt.plot(pressures, temperatures)

pressures = np.linspace(2.e9,16.e9, 10) 
temperatures = np.empty_like(pressures)
for i, pressure in enumerate(pressures):
    T_melt = fsolve(find_temperature, 4000., args=(pressure, coe_SLB, SiO2_liq))[0]
    temperatures[i] = T_melt
plt.plot(pressures, temperatures)


plt.xlim(1.e9, 20.e9)
plt.ylim(2000., 5000.)

plt.show()


# PLOT DENSITY
pressures = np.linspace(2.e9,50.e9, 101) 
temperature = 4000.
volumes = np.empty_like(pressures)
for i, pressure in enumerate(pressures):
    SiO2_liq.set_state(pressure, temperature)
    volumes[i] = SiO2_liq.K_T

plt.plot(pressures, volumes)
plt.show()


# FORSTERITE
fo=SLB_2011.forsterite()
fo_liq=DKS_2013_liquids_tweaked.Mg2SiO4_liquid()

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
fo_s = UnivariateSpline(np.array(fo_liquidus_pressures), np.array(fo_liquidus_temperatures), s=1000.)

pressures = np.linspace(1.e5, 20.e9, 101)


temperatures = np.empty_like(pressures)
for i, pressure in enumerate(pressures):
    T_melt = fsolve(find_temperature, 2000., args=(pressure, fo, fo_liq))[0]
    temperatures[i] = T_melt
plt.plot(pressures, temperatures)


plt.plot(crossover_Ps, crossover_Ts, linewidth=1)
plt.plot(pressures, fo_s(pressures), linewidth=1)
plt.plot(fo_liquidus_pressures, fo_liquidus_temperatures, marker='o', linestyle='None')
plt.plot(fo_solidus_pressures, fo_solidus_temperatures, marker='o', linestyle='None')
plt.show()

# Forsterite liquid entropy at 14 GPa
#dp/dT = ds/dV

p = 14.e9
print 'Forsterite melting at 14 GPa:', fo_s(p)

dp = 1.e7
fo.set_state(p, fo_s(p))
fo_liq.set_state(p, fo_s(p))
dTdP = (fo_s(p+dp) - fo_s(p-dp)) / (2.*dp)
S_fo_melting = (fo_liq.V - fo.V)/dTdP
S_fo_liquid = S_fo_melting + fo.S
print 'Conditions:', p, fo_s(p)
print 'Entropy of melting (fo):', S_fo_melting
print 'Melt entropy (Mg2SiO4):', S_fo_liquid
print 'Melt entropy (DKS2013):', fo_liq.S

# CLINOENSTATITE (Mg2Si2O6)
oen=SLB_2011.enstatite()
cen=SLB_2011.hp_clinoenstatite()
cen_liq=DKS_2013_liquids_tweaked.MgSiO3_liquid()

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

pressures = np.linspace(1.e5, 20.e9, 101)


temperatures = np.empty_like(pressures)
for i, pressure in enumerate(pressures):
    T_melt = fsolve(find_temperature_mul, 2000., args=(pressure, oen, cen_liq, 2.))[0]
    temperatures[i] = T_melt
plt.plot(pressures, temperatures)



plt.plot(pressures, oen_s(pressures), linewidth=2)
plt.plot(oen_liquidus_pressures, oen_liquidus_temperatures, marker='o', linestyle='None')
plt.plot(cen_liquidus_pressures, cen_liquidus_temperatures, marker='o', linestyle='None')
plt.show()




# Mg2Si2O6 liquid entropy from cen at 14 GPa
#dp/dT = ds/dV

p= 2.e9
print 'Orthoenstatite melting at 2 GPa:', oen_s(p)
p = 14.e9
print 'Clinoenstatite melting at 14 GPa:', oen_s(p)


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


T = (oen_s(p) + fo_s(p))/2.


p = 14.e9
T = 2580. # K
per_liq = DKS_2013_liquids_tweaked.MgO_liquid()
per_liq.set_state(p, T)

stv_liq = DKS_2013_liquids_tweaked.SiO2_liquid()
stv_liq.set_state(p, T)

fo_liq = DKS_2013_liquids_tweaked.Mg2SiO4_liquid()
fo_liq.set_state(p, T)

en_liq = DKS_2013_liquids_tweaked.MgSiO3_liquid()
en_liq.set_state(p, T)


fo.set_state(p, T)
cen.set_state(p, T)



obs_compositions = [0.0, 0.33, 0.5, 1.0]
obs_gibbs = [per_liq.gibbs, fo.gibbs / 3., cen.gibbs / 4., stv_liq.gibbs]
obs_entropy = [per_liq.S, S_fo_liquid / 3., S_en_liquid / 2., stv_liq.S]
print obs_gibbs, obs_entropy
calc_entropy = [per_liq.S, fo_liq.S / 3., en_liq.S / 2., stv_liq.S]
obs_excess_gibbs = []
obs_excess_entropy = []
calc_excess_entropy = []
for i, c in enumerate(obs_compositions):
    obs_excess_entropy.append( obs_entropy[i] - ((1.-c)*obs_entropy[0] + c*obs_entropy[3]) )
    calc_excess_entropy.append( calc_entropy[i] - ((1.-c)*calc_entropy[0] + c*calc_entropy[3]) )
    obs_excess_gibbs.append( obs_gibbs[i] - ((1.-c)*obs_gibbs[0] + c*obs_gibbs[3]) )

plt.plot(obs_compositions, obs_excess_gibbs, marker='o', label=str(T)+' K')

# Constraint from eutectic temperature
pressure = 14.e9 # 13.9 in paper
temperature = 2185+273.15
c = 30.7 # +/-1.7 wt % Mg2SiO4 (with MgSiO3)
c = 32.4
M_fo = 140.6931 # Mg2SiO4
M_en = 100.3887 # MgSiO3

nMg = 2.*(c/M_fo) + 1.*(100.-c)/M_en
nSi = 1.*(c/M_fo) + 1.*(100.-c)/M_en

c = nSi/(nMg + nSi)
print 'Composition (fraction SiO2):', c

fo.set_state(pressure, temperature)
cen.set_state(pressure, temperature)

compositions = [0., c, 1.]
formulae=[{'Mg':1., 'O':1.},
          {'Mg':1.-c, 'Si':c, 'O':1.*(1.-c)+2.*c},
          {'Si':1., 'O':2.}]

mus = burnman.chemicalpotentials.chemical_potentials([fo, cen], formulae)
per_liq.set_state(pressure, temperature)
stv_liq.set_state(pressure, temperature)

excess_gibbs = [0., mus[1] - ((1.-c)*per_liq.gibbs + c*stv_liq.gibbs), 0.]
plt.plot(compositions, excess_gibbs, marker='o', label=str(temperature)+' K')

excess_gibbs = [mus[0] - per_liq.gibbs,
                mus[1] - ((1.-c)*per_liq.gibbs + c*stv_liq.gibbs),
                mus[2] - stv_liq.gibbs]

plt.plot(compositions, excess_gibbs, marker='o', label=str(temperature)+' K')


class MgO_SiO2_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Subregular MgO-SiO2 liquid'
        self.type='subregular'

        self.endmembers = [[DKS_2013_liquids_tweaked.MgO_liquid(), '[Mg]O'],
                           [DKS_2013_liquids_tweaked.SiO2_liquid(), '[Si]O2']]
                           

        #self.enthalpy_interaction = [[[-55000., -220000.]]]
        self.enthalpy_interaction = [[[-91838.69740738, -197996.71219146]]]
        self.volume_interaction   = [[[0., 0.]]]
        self.entropy_interaction  = [[[0., 0.]]]
                        
        burnman.SolidSolution.__init__(self, molar_fractions)

liquid = MgO_SiO2_liquid()

pressure = 14.e9 # GPa
compositions = np.linspace(0., 1., 101)
excess_gibbs=np.empty_like(compositions)

for temperature in np.array([2458., 2579.]):
    for i, c in enumerate(compositions):
        liquid.set_composition([1.-c, c])
        liquid.set_state(pressure, temperature)
        excess_gibbs[i]=liquid.excess_gibbs

    plt.plot(compositions, excess_gibbs, label=str(temperature)+' K')

plt.legend(loc='lower right')
plt.title('At 14 GPa and 2580 K, crossover of fo and en melting')
plt.show()


######
# Excess entropy
######

plt.plot(obs_compositions, obs_excess_entropy, marker='o', linestyle='None', label='obs')
plt.plot(obs_compositions, calc_excess_entropy, marker='o', linestyle='None', label='calc')

compositions = np.linspace(0., 1., 101)
excess_entropy=np.empty_like(compositions)
for temperature in np.array([2458., 2579.]):
    for i, c in enumerate(compositions):
        liquid.set_composition([1.-c, c])
        liquid.set_state(pressure, temperature)
        excess_entropy[i]=liquid.excess_entropy

    plt.plot(compositions, excess_entropy, label=str(temperature)+' K')

plt.legend(loc='lower right')
plt.show()




per=SLB_2011.periclase()
fo=SLB_2011.forsterite()
en=SLB_2011.enstatite()
stv=DKS_2013_liquids_tweaked.stishovite()


################
# Melting checks
################

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




##########################
# PHASE DIAGRAM
##########################

class MgO_SiO2_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Subregular MgO-SiO2 liquid'
        self.type='subregular'

        self.endmembers = [[DKS_2013_liquids_tweaked.MgO_liquid(), '[Mg]O'],
                           [DKS_2013_liquids_tweaked.SiO2_liquid(), '[Si]O2']]
                           

        #self.enthalpy_interaction = [[[-55000., -200000.]]]
        self.enthalpy_interaction = [[[-92201.77, -198031.]]]
        self.volume_interaction   = [[[0., 0.]]]
        self.entropy_interaction  = [[[0., 0.]]]
                        
        burnman.SolidSolution.__init__(self, molar_fractions)

liquid = MgO_SiO2_liquid()


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


