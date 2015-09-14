import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from scipy.optimize import fsolve, curve_fit

jadeite = minerals.HP_2011_ds62.jd()
aegirine = minerals.HP_2011_ds62.jd() # note there is no aegirine in Holland and Powell...
ae26 = minerals.HP_2011_ds62.jd()
ae65 = minerals.HP_2011_ds62.jd()
def fit_PV(mineral):
    def fit_EOS(pressures, V_0, K_0, Kprime_0):
        mineral.params['V_0'] = V_0
        mineral.params['K_0'] = K_0
        mineral.params['Kprime_0'] = Kprime_0
        mineral.params['Kdprime_0'] = -Kprime_0/K_0

        volumes=np.empty_like(pressures)
        for i, P in enumerate(pressures):
            mineral.set_state(P, 298.15)
            volumes[i] = mineral.V

        return volumes
    return fit_EOS

xtls = [jadeite, ae26, ae65, aegirine]
compositions = np.array([0.0, 0.26, 0.65, 1.0])
V_0 = np.array([402.26, 408.44, 418.57, 429.26])
Verr = np.array([0.02, 0.01, 0.02, 0.02])

Z = 4.
NA = burnman.constants.Avogadro
A3tom3 = 1e-30

V_0 = V_0/Z*A3tom3*NA
V_0err = Verr/Z*A3tom3*NA


K_0 = np.array([134.0e9, 130.4e9, 124.4e9, 116.1e9])
K_0err = np.array([0.7e9, 0.5e9, 0.6e9, 0.5e9])

all_data = []
composition = []

filenames = ['data/jd_ae_PV_data/ae000_PV.dat', 'data/jd_ae_PV_data/ae026_PV.dat', 'data/jd_ae_PV_data/ae065_PV.dat', 'data/jd_ae_PV_data/ae100_PV.dat']
for i, filename in enumerate(filenames):
    f = open(filename)
    data = []
    datalines = [ line.strip().split() for idx, line in enumerate(f.read().split('\n')) if line.strip() ]
    for content in datalines:
        if content[0] != '%':
            data.append(map(float,content))
            all_data.append(map(float,content))
            composition.append(compositions[i])
            
    P, Perr, V, Verr = zip(*data)
    P = np.array(P)*1.e9

    guesses = [jadeite.params['V_0'], jadeite.params['K_0'], jadeite.params['Kprime_0']]
    popt, pcov = curve_fit(fit_PV(xtls[i]), P, V, guesses, Verr)

    
    #print popt[0]*1.e6, '+/-', np.sqrt(pcov[0][0])*1.e6
    #print popt[1]/1.e9, '+/-', np.sqrt(pcov[1][1])/1.e9 
    #print popt[2], '+/-', np.sqrt(pcov[2][2]) 

print 'Mineral, V (cm^3/mol), K_T (GPa), K\'_T'
print 'ae0', xtls[0].params['V_0']*1.e6, xtls[0].params['K_0']*1.e-9, xtls[0].params['Kprime_0']
print 'ae26', xtls[1].params['V_0']*1.e6, xtls[1].params['K_0']*1.e-9, xtls[1].params['Kprime_0']
print 'ae65', xtls[2].params['V_0']*1.e6, xtls[2].params['K_0']*1.e-9, xtls[2].params['Kprime_0']
print 'ae100', xtls[3].params['V_0']*1.e6, xtls[3].params['K_0']*1.e-9, xtls[3].params['Kprime_0']


P_obs, Perr_obs, V_obs, Verr_obs = zip(*all_data)
P_obs = np.array(P_obs)*1.e9
Perr_obs = np.array(Perr_obs)*1.e9

# Here's the model set up

# First, let's define our intermediates
intermediate0 = minerals.HP_2011_ds62.jd()
intermediate1 = minerals.HP_2011_ds62.jd()

# Now, let's set up a solid solution model
class jadeite_aegirine_binary(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Jadeite-aegirine binary'
        self.type='full_subregular'
        self.endmembers = [[jadeite,  'Na[Al]Si2O6'],
                           [aegirine, 'Na[Fe]Si2O6']]
        self.intermediates=[[[intermediate0, intermediate1]]]
        
        burnman.SolidSolution.__init__(self, molar_fractions)

pyroxene = jadeite_aegirine_binary()

Kp0 = 4.
#def fit_ss_data(data, Vj, Kj, Kpj, V0, K0, Kp0, V1, K1, Kp1, Va, Ka, Kpa):
def fit_ss_data(data, Vj, Kj, V0, K0, V1, K1, Va, Ka):
    Kpj=4.6 # good fit to data
    Kpa=4.4 # good fit to data

    V_int = 0.5*(V0+V1)
    Kp0 = V_int/(0.5*(Vj/(Kpj+1.) + Va/(Kpa+1.))) - 1.
    Kp1=Kp0
    
    jadeite.params['V_0'] = Vj
    jadeite.params['K_0'] = Kj
    jadeite.params['Kprime_0'] = Kpj
    jadeite.params['Kdprime_0'] = -Kpj/Kj

    intermediate0.params['V_0'] = V0
    intermediate0.params['K_0'] = K0
    intermediate0.params['Kprime_0'] = Kp0
    intermediate0.params['Kdprime_0'] = -Kp0/K0

    intermediate1.params['V_0'] = V1
    intermediate1.params['K_0'] = K1
    intermediate1.params['Kprime_0'] = Kp1
    intermediate1.params['Kdprime_0'] = -Kp1/K1

    aegirine.params['V_0'] = Va
    aegirine.params['K_0'] = Ka
    aegirine.params['Kprime_0'] = Kpa
    aegirine.params['Kdprime_0'] = -Kpa/Ka

    volumes = []
    for datum in data:
        c, P = datum
        pyroxene.set_composition([1.-c, c])
        pyroxene.set_state(P, 298.15)
        volumes.append(pyroxene.V)

    return volumes

cP_obs = zip(*[composition, P_obs])
guesses = [jadeite.params['V_0'], jadeite.params['K_0'], jadeite.params['Kprime_0'], \
           jadeite.params['V_0'], jadeite.params['K_0'], jadeite.params['Kprime_0'], \
           jadeite.params['V_0'], jadeite.params['K_0'], jadeite.params['Kprime_0'], \
           jadeite.params['V_0'], jadeite.params['K_0'], jadeite.params['Kprime_0']]

guesses = [jadeite.params['V_0'], jadeite.params['K_0'], \
           jadeite.params['V_0'], jadeite.params['K_0'], \
           jadeite.params['V_0'], jadeite.params['K_0'], \
           jadeite.params['V_0'], jadeite.params['K_0']] 

popt, pcov = curve_fit(fit_ss_data, cP_obs, V_obs, guesses, Verr_obs)

print popt



comps = np.linspace(0.0, 1.0, 101)
volumes = np.empty_like(comps)
bulk_moduli = np.empty_like(comps)
for i, c in enumerate(comps):
    pyroxene.set_composition([1.-c, c])
    pyroxene.set_state(1.e5, 298.15)

    volumes[i] = pyroxene.V
    bulk_moduli[i] = pyroxene.K_T

    pyroxene.set_state(10.e9, 1973.15)

    
plt.errorbar(compositions, V_0, yerr=V_0err, linestyle='None')
plt.plot(comps, volumes)
plt.show()

plt.errorbar(compositions, K_0, yerr=K_0err, linestyle='None')
plt.plot(comps, bulk_moduli)
plt.show()

# Plot volumes obtained from the model
pressures = np.linspace(1.e5, 10.e9, 101)
for c in compositions:
    pyroxene.set_composition([1.-c, c])
    volumes = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        pyroxene.set_state(P, 298.15)
        volumes[i] = pyroxene.V
    plt.plot(pressures/1.e9, volumes, label='Ae'+str(int(c*100.)))
    
plt.errorbar(P_obs/1.e9, V_obs, yerr=Verr_obs, marker='.', linestyle='None')
plt.legend(loc='lower left')
plt.show()


# Plot excess volumes in the  middle of the binary
pressures = np.linspace(1.e5, 400.e9, 101)
excess_volumes = np.empty_like(pressures)

pyroxene.set_composition([0.5, 0.5])

for i, P in enumerate(pressures):
    pyroxene.set_state(P, 298.15)
    excess_volumes[i] = pyroxene.excess_volume

plt.plot(pressures/1.e9, excess_volumes*1.e6)
plt.xlabel('Pressure (GPa)')
plt.ylabel('Excess volume (cm^3/mol)')
plt.show()


# Check KV rule of thumb
pyroxene.set_composition([0.5, 0.5])
pyroxene.set_state(1.e5, 298.15)
K_T = pyroxene.K_T
V = pyroxene.V

KV_jd = jadeite.params['K_0']*jadeite.params['V_0']
KV_ae = aegirine.params['K_0']*aegirine.params['V_0']

K_average = 0.5*(jadeite.params['K_0'] + aegirine.params['K_0'])
KV_average = 0.5*(KV_jd + KV_ae)
print 'Bulk modulus excess according to KV:', (KV_average/V - K_average)/1.e9, 'GPa'
print 'Bulk modulus excess is actually', (K_T - K_average)/1.e9, 'GPa'


