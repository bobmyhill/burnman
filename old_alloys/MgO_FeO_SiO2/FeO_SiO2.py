import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import HP_2011_ds62, Myhill_silicate_liquid
import numpy as np
from scipy import optimize, integrate
import matplotlib.pyplot as plt

class liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='liquid'
        self.endmembers = [[Myhill_silicate_liquid.FeO_liquid(), '[Fe]O'],
                           [Myhill_silicate_liquid.SiO2_liquid(), '[Si]O2']]
        self.type='subregular'
        self.enthalpy_interaction=[[[10000., 10000.]]]
        self.entropy_interaction=[[[0., 0.]]]
        self.volume_interaction=[[[0., 0.]]]

        burnman.SolidSolution.__init__(self, molar_fractions)

SiO2_liq = Myhill_silicate_liquid.SiO2_liquid()
FeO_liq = Myhill_silicate_liquid.FeO_liquid()
fa = HP_2011_ds62.fa()
fs = HP_2011_ds62.fs()
frw = HP_2011_ds62.frw()
coe = HP_2011_ds62.coe()
stv = HP_2011_ds62.stv()
liq = liquid()

# Fe2SiO4 - 0.5*Fe2Si2O6 = FeO
# Fe2Si2O6 - Fe2SiO4 = SiO2

P = 9.e9
T = 1520+273.15
mol_fraction_Fe2SiO4 = 0.55
Fe2SiO4_phase = frw

f = mol_fraction_Fe2SiO4
moles_FeO = (2.*f + 1.*(1.-f))
moles_SiO2 = 1.
mole_fraction_FeO = moles_FeO/(moles_FeO + moles_SiO2)
c = [mole_fraction_FeO, 1.-mole_fraction_FeO]

Fe2SiO4_phase.set_state(P, T)
fs.set_state(P, T)
liq.set_composition(c)
liq.set_state(P, T)

mu_FeO = Fe2SiO4_phase.gibbs - 0.5*fs.gibbs
mu_SiO2 = fs.gibbs - Fe2SiO4_phase.gibbs
print liq.partial_gibbs, mu_FeO, mu_SiO2



def eqm(data):
    liq.enthalpy_interaction = [[data]]
    burnman.SolidSolution.__init__(liq, c)
    liq.set_state(P, T)
    return [mu_FeO - liq.partial_gibbs[0],
            mu_SiO2 - liq.partial_gibbs[1]]

print optimize.fsolve(eqm, [-10000., -10000.]) 


def cotectic_temperature(phase1, phase2):
    def fit(data, P):
        T, f = data # f = mol_fraction_Fe2SiO4
        moles_FeO = (2.*f + 1.*(1.-f))
        moles_SiO2 = 1.
        mole_fraction_FeO = moles_FeO/(moles_FeO + moles_SiO2)
        c = [mole_fraction_FeO, 1.-mole_fraction_FeO]
        phase1.set_state(P, T)
        phase2.set_state(P, T)
        liq.set_composition(c)
        liq.set_state(P, T)

        mu = burnman.chemicalpotentials.chemical_potentials([liq], [phase1.params['formula'], phase2.params['formula']])

        return [mu[0] - phase1.gibbs,
                mu[1] - phase2.gibbs]

    return fit

T, c = optimize.fsolve(cotectic_temperature(Fe2SiO4_phase, fs), [1600., 0.01], args=(6.e9))
print T-273.15, 'C', c

def melting_temperature(T, P, phase, c):
    liq.set_composition(c)
    liq.set_state(P, T[0])
    phase.set_state(P, T[0])
    mu_phase = burnman.chemicalpotentials.chemical_potentials([liq], [phase.params['formula']])[0]
    return  mu_phase - phase.gibbs

def liquidus_composition(c, P, T, phase):
    liq.set_composition([1.-c[0], c[0]])
    liq.set_state(P, T)
    phase.set_state(P, T)
    mu_phase = burnman.chemicalpotentials.chemical_potentials([liq], [phase.params['formula']])[0]
    return  mu_phase - phase.gibbs

#liq.enthalpy_interaction = [[[-35000.,0000.]]]
#burnman.SolidSolution.__init__(liq)

print 'Pressure:', P/1.e9, 'GPa'
print '"fa" melting', optimize.fsolve(melting_temperature, [2000.], args=(P, Fe2SiO4_phase, [2./3., 1./3.]))[0] - 273.15
T, c = optimize.fsolve(cotectic_temperature(Fe2SiO4_phase, fs), [1600., 0.01], args=(P))
print '"fa"-fs eutectic', T - 273.15, c
print 'fs melting', optimize.fsolve(melting_temperature, [1600.], args=(P, fs, [1./2., 1./2.]))[0]-273.15
T, c = optimize.fsolve(cotectic_temperature(fs, coe), [1600., 0.01], args=(P))
print 'coe-fs cotectic', T - 273.15, c
print ''


compositions = np.linspace(0.01, 0.99, 21)
temperatures_fa = np.empty_like(compositions)
temperatures_fs = np.empty_like(compositions)
temperatures_coe = np.empty_like(compositions)

for i, f in enumerate(compositions):
    moles_FeO = (2.*f + 1.*(1.-f))
    moles_SiO2 = 1.
    mole_fraction_FeO = moles_FeO/(moles_FeO + moles_SiO2)
    c = [mole_fraction_FeO, 1.-mole_fraction_FeO]
    temperatures_fs[i] = optimize.fsolve(melting_temperature, [1600.], args=(P, fs, c))[0]
    temperatures_coe[i] = optimize.fsolve(melting_temperature, [1600.], args=(P, coe, c))[0]
    temperatures_fa[i] = optimize.fsolve(melting_temperature, [1600.], args=(P, Fe2SiO4_phase, c))[0]
    
plt.plot(1.-compositions, temperatures_fa-273.15, label='"fa"')
plt.plot(1.-compositions, temperatures_fs-273.15, label='fs')
plt.plot(1.-compositions, temperatures_coe-273.15, label='coe')
plt.legend(loc='lower right')
plt.ylim(1200., 1800.)
plt.show()





# FAYALITE MELTING - FRW-STV-LIQ invariant (Ohtani, 1979)
print 'Fayalite melting'
P = 13.4e9
#liq.enthalpy_interaction = [[[0000., -20000.]]] 
#burnman.SolidSolution.__init__(liq)
T = optimize.fsolve(melting_temperature, [2000.], args=(P, frw, [2./3., 1./3.]))[0]
print T


stv.set_state(P, T)
print liq.partial_gibbs[1], stv.gibbs

P = 7.e9
T = 1600. + 273.15
liq.set_composition([2./3., 1./3.])
liq.set_state(P, T)
fa.set_state(P, T)
frw.set_state(P, T)
print fa.gibbs, frw.gibbs, liq.gibbs*3.
print fa.V, frw.V, liq.V*3.

pressures = np.linspace(1.e9, 20.e9, 20)
temperatures_fa = np.empty_like(pressures)
temperatures_frw = np.empty_like(pressures)

for i, P in enumerate(pressures):
    c = [2./3., 1./3.]
    temperatures_fa[i] = optimize.fsolve(melting_temperature, [1600.], args=(P, fa, c))[0]
    temperatures_frw[i] = optimize.fsolve(melting_temperature, [1600.], args=(P, frw, c))[0]

plt.plot(pressures, temperatures_fa-273.15)
plt.plot(pressures, temperatures_frw-273.15)
plt.show()
