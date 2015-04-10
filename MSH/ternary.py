# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

"""
MSH ternary model
"""

import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman.mineral import Mineral
from scipy.optimize import fsolve
from burnman.chemicalpotentials import *
atomic_masses=read_masses()

per=minerals.HP_2011_ds62.per()
fo=minerals.HP_2011_ds62.fo()
hen=minerals.HP_2011_ds62.hen()
coe=minerals.HP_2011_ds62.coe()
stv=minerals.HP_2011_ds62.stv()
br=minerals.HP_2011_ds62.br()

''' 
STISHOVITE LIQUID
'''

class stv_liquid (Mineral): # equation of state from coesite. Ok at 13 GPa
    def __init__(self):
        formula='Si1.0O2.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'liquid based on coe',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -907000.0 ,
            'S_0': 39.6 ,
            'V_0': 2.064e-05 + 0.0001e-5, # 1.401e-05 ,
            'Cp': [107.8, -0.003279, -190300.0, -1041.6], #[68.1, 0.00601, -1978200.0, -82.1] ,
            'a_0': 1.23e-05, # 1.58e-05 ,
            'K_0': 97900000000.0, # 3.09e+11 ,
            'Kprime_0': 4.19, # 4.6 ,
            'Kdprime_0': -4.3e-11, # -1.5e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

stvL=stv_liquid()

#####################################
stvL.params['S_0']=stvL.params['S_0']+80.
#####################################

# Tweak stv to fit triple point from Zhang et al., 1993
# 13.7 GPa, 2800 C
def find_H(H, P, T):
    stv.params['H_0'] = H
    stv.set_state(P, T+1)
    coe.set_state(P, T+1)
    stv.set_state(P, T)
    coe.set_state(P, T)
    return stv.gibbs-coe.gibbs

Ptr=13.7e9
Ttr=2800.+273.15

print stv.params['H_0']
stv.params['H_0'] = fsolve(find_H, stv.params['H_0'], args=(Ptr, Ttr))[0]


def eqm_2_mineral(min1, min2):
    def eqm(T, P):
        min1.set_state(P, T)
        min2.set_state(P, T)
        return min1.gibbs-min2.gibbs
    return eqm

stv.set_state(Ptr, Ttr)
stvL.set_state(Ptr, Ttr)
stvL.params['H_0'] = stvL.params['H_0'] + (stv.gibbs - stvL.gibbs)

stv.set_state(Ptr, Ttr+1.e-10)
stvL.set_state(Ptr, Ttr+1.e-10)
print stv.gibbs - stvL.gibbs, 'should be zero'

'''
pressures=np.linspace(1.0e10, 1.4e10, 5)
temperatures=np.linspace(2000.+273.15, 2800.+273.15, 5)
for P in pressures:
    for T in temperatures:
        stv.set_state(P, T)
        coe.set_state(P, T)
        print P/1.e8, T-273.15, stv.gibbs/1000., coe.gibbs/1000.
'''

SL_temperatures=[]
SL_pressures=[]
SL_t_err=[]
SL_p_err=[]
other_temperatures=[]
other_pressures=[]
other_t_err=[]
other_p_err=[]
for line in open('./stv_melting.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[4] == '2': # Shen and Lazor data
            SL_pressures.append(float(content[0])*1.e9)
            SL_p_err.append(float(content[1])*1.e9)
            SL_temperatures.append(float(content[2]))
            SL_t_err.append(float(content[3]))
        else:
            other_pressures.append(float(content[0])*1.e9)
            other_p_err.append(float(content[1])*1.e9)
            other_temperatures.append(float(content[2]))
            other_t_err.append(float(content[3]))

SL_temperatures=np.array(SL_temperatures)
SL_pressures=np.array(SL_pressures)
SL_t_err=np.array(SL_t_err)
SL_p_err=np.array(SL_p_err)
other_temperatures=np.array(other_temperatures)
other_pressures=np.array(other_pressures)
other_t_err=np.array(other_t_err)
other_p_err=np.array(other_p_err)

plt.errorbar( SL_pressures/1.e9, SL_temperatures-273.15, xerr=SL_p_err/1.e9, marker='o', linestyle='none', yerr=SL_t_err-273.15, label='Shen and Lazor (1995)')
plt.errorbar( other_pressures/1.e9, other_temperatures-273.15, xerr=other_p_err/1.e9, marker='o', linestyle='none', yerr=other_t_err-273.15, label='others')

stv_coe_pressures=np.linspace(10.e9, 14.e9, 20)
coe_L_pressures=np.linspace(10.e9, 14.e9, 20)
stv_L_pressures=np.linspace(12.e9, 30.e9, 20)
stv_coe_temperatures=np.empty_like(stv_coe_pressures)
coe_L_temperatures=np.empty_like(coe_L_pressures)
stv_L_temperatures=np.empty_like(stv_L_pressures)

for i, P in enumerate(stv_coe_pressures):
    stv_coe_temperatures[i]=fsolve(eqm_2_mineral(stv, coe), 4000., args=(P))[0]


for i, P in enumerate(coe_L_pressures):
    coe_L_temperatures[i]=fsolve(eqm_2_mineral(coe, stvL), 4000., args=(P))[0]

for i, P in enumerate(stv_L_pressures):
    stv_L_temperatures[i]=fsolve(eqm_2_mineral(stv, stvL), 4000., args=(P))[0]

plt.plot( stv_coe_pressures/1.e9, stv_coe_temperatures-273.15, '-', linewidth=1., label='coe-stv')
plt.plot( coe_L_pressures/1.e9, coe_L_temperatures-273.15, '-', linewidth=1., label='coe')
plt.plot( stv_L_pressures/1.e9, stv_L_temperatures-273.15, '-', linewidth=1., label='stv')

plt.xlim(4, 50)
plt.ylim(2000, 4000)
plt.title("SiO2 phase diagram")
plt.ylabel("Temperature (C)")
plt.xlabel("Pressure (GPa)")
plt.legend(loc='lower left')
plt.show()

P=13.e9
TstvL=fsolve(eqm_2_mineral(stv, stvL), 4000., args=(P))[0]

stv.set_state(P, TstvL)
stvL.set_state(P, TstvL)

print stvL.S-stv.S

'''
PERICLASE LIQUID
'''

class per_liquid (Mineral):
    def __init__(self):
        formula='Mg1.0O1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'per',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -601530.0 ,
            'S_0': 26.5 ,
            'V_0': 1.125e-05 ,
            'Cp': [60.5, 0.000362, -535800.0, -299.2] ,
            'a_0': 3.11e-05 ,
            'K_0': 1.616e+11 ,
            'Kprime_0': 3.95 ,
            'Kdprime_0': -2.4e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

perL=per_liquid()

P=13.e9
####################
perL.params['S_0']=perL.params['S_0']+24.8
Tmelt=4040.+273.15 # 4040 C is Alfe (2005)
####################

per.set_state(P, Tmelt)
perL.set_state(P, Tmelt)
perL.params['H_0'] = perL.params['H_0'] + (per.gibbs - perL.gibbs)

per.set_state(P, Tmelt+1.e-10)
perL.set_state(P, Tmelt+1.e-10)
print per.gibbs - perL.gibbs, 'should be zero'



'''
H2O liquid
'''

class H2O_liquid_13GPa (Mineral):
    def __init__(self):
        formula='H2.0O1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'water at 13 GPa',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -876390.0 ,
            'S_0': 24.0 ,
            'V_0': 1.401e-05 ,
            'Cp': [68.1, 0.00601, -1978200.0, -82.1] ,
            'a_0': 1.58e-05 ,
            'K_0': 3.09e+11 ,
            'Kprime_0': 4.6 ,
            'Kdprime_0': -1.5e-11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        Mineral.__init__(self)

H2OL=H2O_liquid_13GPa()

'''
DRY MgO-SiO2 SYSTEM
'''

# APPROX DE KOKER AND STIXRUDE @ 13 GPa
'''
class MS_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='MgO-SiO2 liquid'
        endmembers = [[perL, '[Mg]'],[stvL, '[Si]O2']]
        lambdas = [1.43, 1.]
        enthalpy_interaction=[[[0.e3, -200.0e3]]]
        entropy_interaction=[[[60., 20.]]]
        volume_interaction=[[[0., 0.]]]
        burnman.SolidSolution.__init__(self, endmembers, \
                                           burnman.solutionmodel.AsymmetricRegularSolution(endmembers, lambdas, enthalpy_interaction, volume_interaction, entropy_interaction), molar_fractions)

##########################
perL.params['H_0'] = perL.params['H_0'] - 000.
##########################
'''

# ATTEMPT TO TWEAK MIXING MODEL TO FIT MELTING OF FORSTERITE AND ENSTATITE
'''
class MS_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='MgO-SiO2 liquid'
        endmembers = [[perL, '[Mg]'],[stvL, '[Si]O2']]
        lambdas = [1.43, 1.]
        enthalpy_interaction=[[[10.e3, -220.0e3]]]
        entropy_interaction=[[[25., 40.]]]
        volume_interaction=[[[0., 0.]]]
        burnman.SolidSolution.__init__(self, endmembers, \
                                           burnman.solutionmodel.AsymmetricRegularSolution(endmembers, lambdas, enthalpy_interaction, volume_interaction, entropy_interaction), molar_fractions)
'''

# OWN MODEL
class MS_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='MgO-SiO2 liquid'
        endmembers = [[perL, '[Mg]'],[stvL, '[Si]O2']]
        enthalpy_interaction=[[[-60.e3, -180.0e3]]]
        entropy_interaction=[[[60., 20.]]]
        volume_interaction=[[[0., 0.]]]
        burnman.SolidSolution.__init__(self, endmembers, \
                                           burnman.solutionmodel.SubregularSolution(endmembers, enthalpy_interaction, volume_interaction, entropy_interaction), molar_fractions)

MS_L=MS_liquid()

P=13.e9
T=3000.

T_metafo_melt=2301.+273.15 # Presnall and Walter (1993)
T_hen_melt=2282.5+273.15 # Presnall and Gasparik (1990)


MS_L.set_composition([2./3., 1./3.])
MS_L.set_state(P, T_metafo_melt)
fo.set_state(P, T_metafo_melt)
perL.set_state(P, T_metafo_melt)
stvL.set_state(P, T_metafo_melt)
Gex_metafo_melt_obs=fo.gibbs/3. - (2./3.*perL.gibbs + 1./3.*stvL.gibbs)
Gex_metafo_melt_calc=MS_L.excess_gibbs

MS_L.set_composition([1./2., 1./2.])
MS_L.set_state(P, T_hen_melt)
hen.set_state(P, T_hen_melt)
perL.set_state(P, T_hen_melt)
stvL.set_state(P, T_hen_melt)
Gex_hen_melt_obs=hen.gibbs/4. - (1./2.*perL.gibbs + 1./2.*stvL.gibbs)
Gex_hen_melt_calc=MS_L.excess_gibbs

compositions=np.linspace(0.01, 0.99, 100)
Hex=np.empty_like(compositions)
Sex=np.empty_like(compositions)
Vex=np.empty_like(compositions)
Gex=np.empty_like(compositions)
for i, c in enumerate(compositions):
    MS_L.set_composition([1.-c, c])
    MS_L.set_state(P, T)

    Hex[i]=MS_L.excess_enthalpy
    Sex[i]=MS_L.excess_entropy
    Vex[i]=MS_L.excess_volume
    Gex[i]=MS_L.excess_gibbs

plt.subplot(2,2,1)
plt.plot( compositions, Hex, '-', linewidth=1.)
plt.ylabel("excess enthalpy")
plt.xlabel("X (SiO2)")

plt.subplot(2,2,2)
plt.plot( compositions, Sex, '-', linewidth=1.)
plt.ylabel("excess entropy")
plt.xlabel("X (SiO2)")

plt.subplot(2,2,3)
plt.plot( compositions, Vex, '-', linewidth=1.)
plt.ylabel("excess volume")
plt.xlabel("X (SiO2)")

plt.subplot(2,2,4)
plt.plot( compositions, Gex, '-', linewidth=1.)
plt.plot( [1./3., 1./2.], [Gex_metafo_melt_obs, Gex_hen_melt_obs], marker='o', label='obs')
plt.plot( [1./3., 1./2.], [Gex_metafo_melt_calc, Gex_hen_melt_calc], marker='o', label='calc')
plt.legend(loc='upper right')

plt.ylabel("excess gibbs")
plt.xlabel("X (SiO2)")

plt.show()





MS_L=MS_liquid()
MS_L.set_composition([2./3., 1./3.])
MS_L.set_state(P, T_metafo_melt)
fo.set_state(P, T_metafo_melt)

print fo.gibbs, MS_L.gibbs[0]*3.


MS_L.set_composition([1./2., 1./2.])
MS_L.set_state(P, T_hen_melt)
hen.set_state(P, T_hen_melt)

print hen.gibbs, MS_L.gibbs[0]*4.



########
# TERNARY MODEL
########

class MSH_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='MgO-SiO2-H2O liquid'
        endmembers = [[perL, '[Mg]O'],[stvL, '[Si]O2'],[H2OL, '[H]HO']]
        enthalpy_interaction=[[[-60.e3, -180.0e3], [-200.e3, -200.0e3]], [[-200.e3, -200.0e3]]]
        entropy_interaction=[[[60., 20.], [0.e3, 0.e3]], [[0.e3, 0.0e3]]]
        volume_interaction=[[[0., 0.], [0., 0.]], [[0., 0.]]]
        burnman.SolidSolution.__init__(self, endmembers, \
                                           burnman.solutionmodel.SubregularSolution(endmembers, enthalpy_interaction, volume_interaction, entropy_interaction), molar_fractions)

melt=MSH_liquid()

# Cotectic relationships between per+fo, fo+hen, hen+stv 
# at a given P and bulk H2O can be defined in terms of:
# knowns (activity of MgO and SiO2 from the solids) 
# and unknowns (temperature, MgO/(MgO+SiO2)), 
# and from that, also activity of H2O via Gibbs-Duhem.

def eqm_cotectic(data, P, XH2O, min1, min2):
    T, Mgnum = data # Mgnum = XMgO/(XMgO+XSiO2) = MgO/(1.-XH2O)

    XMgO=Mgnum*(1.-XH2O)
    XSiO2=1.-XH2O - XMgO

    min1.set_state(P, T)
    min2.set_state(P, T)
    muMgO, muSiO2 = chemical_potentials([min1, min2], [dictionarize_formula('MgO'), dictionarize_formula('SiO2')])
    melt.set_composition([XMgO, XSiO2, XH2O])
    melt.set_state(P, T)
    muMgO_melt, muSiO2_melt, muH2O_melt = melt.partial_gibbs

    return [muMgO_melt-muMgO, muSiO2_melt-muSiO2]


def eqm_liquidus_temperature(data, P, mineral, XMgO, XSiO2):
    T=data[0]
    XH2O = 1. - XMgO - XSiO2
    melt.set_composition([XMgO, XSiO2, XH2O])
    melt.set_state(P, T)
    mineral.set_state(P, T)
    mu_mineral_melt=chemical_potentials([melt], [mineral.params['formula']])[0]
    return mu_mineral_melt - mineral.gibbs

print ''
compositions=np.linspace(0.0, 0.45, 46)
temperatures=np.empty_like(compositions)
for i, XSiO2 in enumerate(compositions):
    temperatures[i]=fsolve(eqm_liquidus_temperature, 5000., args=(13.e9, per, 1.-XSiO2, XSiO2))[0]

plt.plot( compositions, temperatures-273.15, '-', linewidth=1., label='per')

compositions=np.linspace(0.33, 0.5, 16)
temperatures=np.empty_like(compositions)
for i, XSiO2 in enumerate(compositions):
    temperatures[i]=fsolve(eqm_liquidus_temperature, 5000., args=(13.e9, fo, 1.-XSiO2, XSiO2))[0]

plt.plot( compositions, temperatures-273.15, '-', linewidth=1., label='fo')

compositions=np.linspace(0.4, 0.7, 31)
temperatures=np.empty_like(compositions)
for i, XSiO2 in enumerate(compositions):
    temperatures[i]=fsolve(eqm_liquidus_temperature, 5000., args=(13.e9, hen, 1.-XSiO2, XSiO2))[0]

plt.plot( compositions, temperatures-273.15, '-', linewidth=1., label='hen')

compositions=np.linspace(0.55, 0.9, 46)
temperatures=np.empty_like(compositions)
for i, XSiO2 in enumerate(compositions):
    temperatures[i]=fsolve(eqm_liquidus_temperature, 5000., args=(13.e9, stv, 1.-XSiO2, XSiO2))[0]

plt.plot( compositions, temperatures-273.15, '-', linewidth=1., label='stv')

compositions=np.linspace(0.8, 1.0, 21)
temperatures=np.empty_like(compositions)
for i, XSiO2 in enumerate(compositions):
    temperatures[i]=fsolve(eqm_liquidus_temperature, 5000., args=(13.e9, coe, 1.-XSiO2, XSiO2))[0]


plt.plot( compositions, temperatures-273.15, '-', linewidth=1., label='coe')


plt.title("MgO-SiO2 binary phase diagram")
plt.ylabel("Temperature (C)")
plt.xlabel("X SiO2")
plt.legend(loc='lower left')
plt.show()


# COTECTICS
compositions=np.linspace(0.0, 0.6, 21)
temperatures_per_fo=np.empty_like(compositions)
XMgOs_per_fo=np.empty_like(compositions)
temperatures_fo_hen=np.empty_like(compositions)
XMgOs_fo_hen=np.empty_like(compositions)
temperatures_hen_stv=np.empty_like(compositions)
XMgOs_hen_stv=np.empty_like(compositions)

for i, XH2O in enumerate(compositions):
    temperatures_per_fo[i], XMgOs_per_fo[i]=fsolve(eqm_cotectic, [2000., 0.3], args=(13.e9, XH2O, per, fo))
    temperatures_fo_hen[i], XMgOs_fo_hen[i]=fsolve(eqm_cotectic, [2000., 0.3], args=(13.e9, XH2O, fo, hen))
    temperatures_hen_stv[i], XMgOs_hen_stv[i]=fsolve(eqm_cotectic, [2000., 0.3], args=(13.e9, XH2O, hen, stv))

plt.subplot(2,1,1)
plt.plot( compositions, temperatures_per_fo-273.15, '-', linewidth=1., label='per-fo cotectic')
plt.plot( compositions, temperatures_fo_hen-273.15, '-', linewidth=1., label='fo-hen cotectic')
plt.plot( compositions, temperatures_hen_stv-273.15, '-', linewidth=1., label='hen-stv cotectic')
#plt.title("Periclase-forsterite cotectic temperature")
plt.ylabel("Temperature (C)")
plt.xlabel("XH2O")
plt.legend(loc='lower left')

plt.subplot(2,1,2)
plt.plot( 1.-XMgOs_per_fo, compositions, '-', linewidth=1., label='per-fo cotectic')
plt.plot( 1.-XMgOs_fo_hen, compositions, '-', linewidth=1., label='fo-hen cotectic')
plt.plot( 1.-XMgOs_hen_stv, compositions, '-', linewidth=1., label='hen-stv cotectic')
#plt.title("Periclase-forsterite cotectic compositions")
plt.ylabel("XH2O")
plt.xlabel("XSiO2")
plt.legend(loc='lower left')

plt.show()
