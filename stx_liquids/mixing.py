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

per_liq = DKS_2013_liquids_tweaked.MgO_liquid()
fo_liq = DKS_2013_liquids_tweaked.Mg2SiO4_liquid()
en_liq = DKS_2013_liquids_tweaked.MgSiO3_liquid()
stv_liq = DKS_2013_liquids_tweaked.SiO2_liquid()

per = SLB_2011.periclase()
fo = SLB_2011.forsterite()
cen = SLB_2011.hp_clinoenstatite()
stv = DKS_2013_liquids_tweaked.stishovite()

class MgO_SiO2_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Subregular MgO-SiO2 liquid'
        self.type='subregular'

        self.endmembers = [[per_liq, '[Mg]O'],
                           [stv_liq, '[Si]O2']]
                           

        self.enthalpy_interaction = [[[-55000., -220000.]]]
        self.volume_interaction   = [[[0., 0.]]]
        self.entropy_interaction  = [[[0., 0.]]]
                        
        burnman.SolidSolution.__init__(self, molar_fractions)

liquid = MgO_SiO2_liquid()

pressure = 14.e9
temperature = 2580.

phases = [per_liq, fo_liq, en_liq, stv_liq, per, fo, cen, stv]
for phase in phases:
    phase.set_state(pressure, temperature)

obs_compositions = [0.0, 0.33, 0.5, 1.0]
obs_gibbs = [per_liq.gibbs, fo.gibbs / 3., cen.gibbs / 4., stv_liq.gibbs]

obs_entropy = [148.97072200859682, 169.36543270572091, 163.52121987497708, 199.83480153876246]

calc_entropy = [per_liq.S, fo_liq.S / 3., en_liq.S / 2., stv_liq.S]
obs_excess_gibbs = []
obs_excess_entropy = []
calc_excess_entropy = []
for i, c in enumerate(obs_compositions):
    obs_excess_entropy.append( obs_entropy[i] - ((1.-c)*obs_entropy[0] + c*obs_entropy[3]) )
    calc_excess_entropy.append( calc_entropy[i] - ((1.-c)*calc_entropy[0] + c*calc_entropy[3]) )
    obs_excess_gibbs.append( obs_gibbs[i] - ((1.-c)*obs_gibbs[0] + c*obs_gibbs[3]) )

plt.plot(obs_compositions, obs_excess_gibbs, marker='o', label=str(temperature)+' K')

# Constraint from eutectic temperature
pressure = 14.e9 # 13.9 in paper
temperature = 2185+273.15 +50 # note 50 K tweak to better agree with fo and en melting
c = 30.7 # +/-1.7 wt % Mg2SiO4 (with MgSiO3)
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
liquid.set_composition([1.-c, c])
liquid.set_state(pressure, temperature)

def fit_gradient(W, pressure, temperature, c, mus):
    liquid.enthalpy_interaction = [[[W[0], W[1]]]]
    burnman.SolidSolution.__init__(liquid, [1.-c, c])
    liquid.set_state(pressure, temperature)
    print liquid.partial_gibbs
    return [liquid.partial_gibbs[0] - mus[0], liquid.partial_gibbs[1] - mus[2]]


print fsolve(fit_gradient, [-100000., -120000.], args=(pressure, temperature, c, mus))

excess_gibbs = [0., mus[1] - ((1.-c)*per_liq.gibbs + c*stv_liq.gibbs), 0.]
plt.plot(compositions, excess_gibbs, marker='o', label=str(temperature)+' K')

excess_gibbs = [mus[0] - per_liq.gibbs,
                mus[1] - ((1.-c)*per_liq.gibbs + c*stv_liq.gibbs),
                mus[2] - stv_liq.gibbs]

plt.plot(compositions, excess_gibbs, marker='o', label=str(temperature)+' K')


pressure = 14.e9 # GPa
compositions = np.linspace(0., 1., 101)
excess_gibbs=np.empty_like(compositions)

for temperature in np.array([2458., 2580.]):
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
for temperature in np.array([2458., 2580.]):
    for i, c in enumerate(compositions):
        liquid.set_composition([1.-c, c])
        liquid.set_state(pressure, temperature)
        excess_entropy[i]=liquid.excess_entropy

    plt.plot(compositions, excess_entropy, label=str(temperature)+' K')

plt.legend(loc='lower right')
plt.show()
