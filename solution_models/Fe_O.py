import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

from scipy.optimize import fsolve, curve_fit
import burnman
from burnman import minerals



from B1_wuestite import B1_wuestite
from liq_Fe5O5_AA1994 import liq_Fe5O5
from liq_wuestite_AA1994 import liq_FeO
from fcc_iron import fcc_iron
from hcp_iron import hcp_iron
from liq_iron_AA1994 import liq_iron

Fe_hcp = hcp_iron()
Fe_fcc = fcc_iron()
Fe_liq = liq_iron()
FeO_B1 = B1_wuestite()
Fe5O5_liq = liq_Fe5O5()
FeO_liq = liq_FeO()


eutectic_PT = []
f=open('data/Fe_FeO_eutectic_temperature.dat', 'r')
datastream = f.read()  # We need to open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] != "%":
        eutectic_PT.append([float(line[0])*1.e9, float(line[1]), 
                            float(line[2]), float(line[3])])


eutectic_PTc = []
f=open('data/Fe_FeO_eutectic.dat', 'r') 
datastream = f.read()  # We need to open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] != "%":
        # P, Perr, T, Terr, FeO (mol fraction), err
        eutectic_PTc.append([float(line[1])*1.e9, 2.0e9, 
                             float(line[2]), float(line[3]), 
                             float(line[4])/100., float(line[5])/100.])

solvus_PTcc = []
f=open('data/Fe_FeO_solvus.dat', 'r')
datastream = f.read()  # We need to open the file
f.close()
datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
for line in datalines:
    if line[0] != "%":
        # P, Perr, T, Terr, FeO (mol fraction), err, FeO (mol fraction), err
        solvus_PTcc.append([float(line[1])*1.e9, 2.0e9, 
                            float(line[2]), float(line[3]), 
                            float(line[4])/100., float(line[5])/100.,
                            float(line[8])/100., float(line[9])/100.])



eutectic_PT = np.array(eutectic_PT).T
eutectic_PTc = np.array(eutectic_PTc).T

class Fe_FeO_liquid5(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Fe-FeO solution'
        self.type='ideal'
        self.endmembers = [[liq_iron(), '[Fe]0.5[Fe]0.5'], [liq_Fe5O5(), '[Fe]0.5[O]0.5']]
        burnman.SolidSolution.__init__(self, molar_fractions)

liq5 = Fe_FeO_liquid5()

class Fe_FeO_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Fe-FeO solution'
        self.type='ideal'
        self.endmembers = [[liq_iron(), '[Fe]'], [liq_FeO(), 'Fe[O]']]
        burnman.SolidSolution.__init__(self, molar_fractions)

liq = Fe_FeO_liquid()


# Plot eutectic temperatures and compositions
def eutectic_liquid_Fe5O5(cT, P, liq, Fe_phase, FeO_phase):
    c, T = cT

    liq.set_composition([1.-c, c])
    liq.set_state(P, T)
    Fe_phase.set_state(P, T)
    FeO_phase.set_state(P, T)

    equations = [ Fe_phase.gibbs - liq.partial_gibbs[0],
                  FeO_phase.gibbs - 2.*liq.partial_gibbs[1]]
    return equations

def eutectic_liquid_FeO(cT, P, liq, Fe_phase, FeO_phase):
    c, T = cT

    liq.set_composition([1.-c, c])
    liq.set_state(P, T)
    Fe_phase.set_state(P, T)
    FeO_phase.set_state(P, T)

    equations = [ Fe_phase.gibbs - liq.partial_gibbs[0],
                  FeO_phase.gibbs - liq.partial_gibbs[1]]
    return equations


pressures = np.linspace(30.e9, 250.e9, 16)
eutectic_compositions = np.empty_like(pressures)
eutectic_temperatures = np.empty_like(pressures)
eutectic_compositions_2 = np.empty_like(pressures)
eutectic_temperatures_2 = np.empty_like(pressures)

c, T = [0.5, 4000.]
for i, P in enumerate(pressures):
    print c, T
    c, T = fsolve(eutectic_liquid_Fe5O5, [c, T], args=(P, liq5, Fe_hcp, FeO_B1))

    # c here is fraction of Fe0.5O0.5, but we want fraction of FeO
    nFe = (1.-c) + 0.5*c
    nO = 0.5*c
    xFeO = nO
    xFe = nFe - nO

    total = xFeO + xFe
    
    eutectic_compositions[i] = xFeO/total
    eutectic_temperatures[i] = T

    c, T = fsolve(eutectic_liquid_FeO, [c, T], args=(P, liq, Fe_hcp, FeO_B1))
    
    eutectic_compositions_2[i] = c
    eutectic_temperatures_2[i] = T

    
plt.plot(pressures/1.e9, eutectic_compositions, label='Fe - Fe0.5O0.5')
plt.plot(pressures/1.e9, eutectic_compositions_2, label='Fe - FeO')
plt.plot(eutectic_PTc[0]/1.e9, eutectic_PTc[4], marker='o', linestyle='None', label='Seagle')
plt.legend(loc='lower right')
plt.show()

plt.plot(pressures/1.e9, eutectic_temperatures, label='Fe - Fe0.5O0.5')
plt.plot(pressures/1.e9, eutectic_temperatures_2, label='Fe - FeO')
plt.plot(eutectic_PT[0]/1.e9, eutectic_PT[2], marker='o', linestyle='None', label='Seagle')
plt.plot(eutectic_PTc[0]/1.e9, eutectic_PTc[2], marker='o', linestyle='None', label='Seagle')
plt.legend(loc='lower right')
plt.show()

exit()


'''
# Phase diagram at 50 GPa
'''
def mineral_fugacity(c, mineral, liquid, P, T):
    mineral.set_state(P, T)
    liq.set_composition([1. - c[0], c[0]])
    liq.set_state(P, T)
    return [burnman.chemicalpotentials.fugacity(mineral, [liquid]) - 1.]

def molFeO2wtO(molFeO):
    molFe = 1.
    molO = molFeO
    massFe = molFe*55.845
    massO = molO*15.9994
    return massO/(massFe+massO)*100.
def molFeO2wtFeO(molFeO):
    molFe = 1. - molFeO
    massFe = molFe*55.845
    massFeO = molFeO*(15.9994+55.845)
    return massFeO/(massFe+massFeO)*100.

P = 50.e9
Fe_phase = Fe_hcp
    
c, T_eutectic = fsolve(eutectic_liquid, [0.1, 2500.], args=(P, liq, Fe_phase, FeO_B1))
T_Fe_melt = burnman.tools.equilibrium_temperature([Fe_phase, Fe_liq], [1., -1.], P)
T_FeO_melt = burnman.tools.equilibrium_temperature([FeO_B1, FeO_liq], [1., -1.], P)

print molFeO2wtO(c), T_eutectic, T_Fe_melt, T_FeO_melt

temperatures = np.linspace(T_eutectic, T_Fe_melt, 20)
Fe_liquidus_compositions = np.empty_like(temperatures)
c=0.01
for i, T in enumerate(temperatures):
    print i, T
    c = fsolve(mineral_fugacity, [c], args=(Fe_phase, liq, P, T))[0]
    Fe_liquidus_compositions[i] = c 
plt.plot(molFeO2wtO(Fe_liquidus_compositions), temperatures)

temperatures = np.linspace(T_eutectic, T_FeO_melt, 20)
c=0.99
FeO_liquidus_compositions = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    print i, T
    c = fsolve(mineral_fugacity, [c], args=(FeO_B1, liq, P, T))[0]
    FeO_liquidus_compositions[i] = c
plt.plot(molFeO2wtO(FeO_liquidus_compositions), temperatures)
plt.xlim(0., 23.)
plt.xlabel('wt % FeO')

# Seagle data
lo_eutectic = np.loadtxt(fname='data/Seagle_2008_low_eutectic_bounds.dat', comments='%')
hi_eutectic = np.loadtxt(fname='data/Seagle_2008_high_eutectic_bounds.dat', comments='%')
lo_liquidus = np.loadtxt(fname='data/Seagle_2008_low_liquidus_bounds.dat', comments='%')
hi_liquidus = np.loadtxt(fname='data/Seagle_2008_high_liquidus_bounds.dat', comments='%')

for a in [lo_eutectic, hi_eutectic, lo_liquidus, hi_liquidus]:
    a = a[a[:,0]>=45.0, :]
    a = a[a[:,0]<=55.0, :]
    plt.plot(a[:,3], a[:,2], marker='s', markersize=10, linestyle='None')


plt.show()
'''

# Plot solvus
def eqm_two_liquid(cc, P, T, model):
    c1, c2 = cc

    model.set_composition([1.-c1, c1])
    model.set_state(P, T)

    partial_excesses_1 = model.excess_partial_gibbs
    model.set_composition([1.-c2, c2])
    model.set_state(P, T)
    partial_excesses_2 = model.excess_partial_gibbs
    equations = [ partial_excesses_1[0] - partial_excesses_2[0],
                  partial_excesses_1[1] - partial_excesses_2[1]]
    return equations

temperatures = [2173., 2573.]
pressures = np.linspace(1.e5, 28.e9, 8)

compositions_1 = np.empty_like(pressures)
compositions_2 = np.empty_like(pressures)


for T in temperatures:
    print T
    c1=0.01
    c2=0.99
    for i, P in enumerate(pressures):
        print P
        c1, c2 = fsolve(eqm_two_liquid, [c1, c2],
                        args=(P, T, liq), factor = 0.1, xtol=1.e-12)
        compositions_1[i] = c1
        compositions_2[i] = c2
    plt.plot(compositions_1, pressures/1.e9, label='Metallic at '+str(T)+' K')
    plt.plot(compositions_2, pressures/1.e9, label='Ionic at '+str(T)+' K')

    
plt.plot(solvus_PTcc[4], solvus_PTcc[0]/1.e9, marker='o', linestyle='None')
plt.plot(solvus_PTcc[6], solvus_PTcc[0]/1.e9, marker='o', linestyle='None')
plt.legend(loc='upper right')
plt.show()
'''
