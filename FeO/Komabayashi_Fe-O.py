import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import Komabayashi_2014
import numpy as np
from scipy import optimize, integrate
import matplotlib.pyplot as plt

# READ IN DATA
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
        solvus_PTcc.append([float(line[1])*1.e9, 2.0e9, 
                            float(line[2]), float(line[3]), 
                            float(line[4])/100., float(line[5])/100.,
                            float(line[8])/100., float(line[9])/100.])


# partial_gibbs_Fe_liq = gibbs_Fe_solid
# partial_gibbs_FeO_liq = gibbs_FeO_solid

class subregular(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='frost subregular model'
        self.type='subregular'
        self.endmembers = [[burnman.minerals.HP_2011_ds62.py(), '[Fe]'],
                           [burnman.minerals.HP_2011_ds62.py(), '[O]']]

        # W_FeFeO then W_FeOFe 
        self.enthalpy_interaction = [[[0., 0.]]]

        burnman.SolidSolution.__init__(self, molar_fractions)


Fe_fcc=Komabayashi_2014.fcc_iron()
Fe_liq=Komabayashi_2014.liquid_iron()
FeO=Komabayashi_2014.FeO_solid()
FeO_liq=Komabayashi_2014.FeO_liquid()

'''
# Nasch and Manghani raw data
Fe_liq.params['V_0'] = 7.068e-6  # Komabayashi = 6.88
Fe_liq.params['K_0'] = 156.e9  # Komabayashi = 148
Fe_liq.params['Kprime_0'] = 5.8 # Komabayashi = 5.8
Fe_liq.params['a_0'] = 8.2e-5 # Komabayashi = 9
Fe_liq.params['delta_0'] = 6.3 # Komabayashi = 5.1
Fe_liq.params['kappa'] = 0.56 # Komabayashi = 0.56
'''

model = subregular()

def fit_enthalpy(args, c, P, T, model, partials):
    H0, H1 = args
    model.enthalpy_interaction = [[[H0, H1]]]
    burnman.SolidSolution.__init__(model)
    model.set_composition([1.-c, c])
    model.set_state(P, T) 

    return [partials[0] - model.excess_partial_gibbs[0], 
            partials[1] - model.excess_partial_gibbs[1]] 
  
def fit_enthalpy_solvus(args, compositions, P, T, model):
    H0, H1 = args
    c0, c1 = compositions

    model.enthalpy_interaction = [[[H0, H1]]]
    burnman.SolidSolution.__init__(model)

    model.set_composition([1.-c0, c0])
    model.set_state(P, T) 
    metallic_excesses = model.excess_partial_gibbs

    model.set_composition([1.-c1, c1])
    model.set_state(P, T) 
    ionic_excesses = model.excess_partial_gibbs


    return [metallic_excesses[0] - ionic_excesses[0],
            metallic_excesses[1] - ionic_excesses[1]] 



# FITTING DATA
pressures = []
temperatures = []
Hex0 = []
Hex1 = []
weighting=[]
eutectic_weighting = 5.

for datum in eutectic_PTc:
    P, Perr, T, Terr, c, cerr = datum

    model.set_composition([1.-c, c])
    model.set_state(P, T)


    partials = [Fe_fcc.calcgibbs(P, T) - Fe_liq.calcgibbs(P,T),
                FeO.calcgibbs(P,T) - FeO_liq.calcgibbs(P,T)]

    Hex = optimize.fsolve(fit_enthalpy, [0., 0.], args=(c, P, T, model, partials))
    pressures.append(P)
    temperatures.append(T)
    Hex0.append(Hex[0])
    Hex1.append(Hex[1])
    weighting.append(eutectic_weighting)

# and now fit the solvus:
# at a given pressure and temperature, two compositions have the same partial gibbs excesses

for datum in solvus_PTcc:
    P, Perr, T, Terr, c0, c0err, c1, c1err = datum

    Hex = optimize.fsolve(fit_enthalpy_solvus, [0., 0.], args=([c0, c1], P, T, model))
    pressures.append(P)
    temperatures.append(T)
    Hex0.append(Hex[0])
    Hex1.append(Hex[1])
    weighting.append(1.)

plt.plot(pressures, Hex0, marker='o', linestyle='None', label='0')
plt.plot(pressures, Hex1, marker='o', linestyle='None', label='1')


from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass

# Fit excess properties
class Fe_FeO_liq(burnman.Mineral): # This should be the fictitious endmember without the entropy of mixing
    def __init__(self, Hex, Sex, Vex, Kex, aex):
        formula='Fe1.0O0.5'
        formula = dictionarize_formula(formula)
        V_0 = (FeO_liq.params['V_0'] + Fe_liq.params['V_0'])/2. + Vex
        K_0 = V_0*(1. / (0.5*(FeO_liq.params['V_0']/FeO_liq.params['K_0']
                              + Fe_liq.params['V_0']/Fe_liq.params['K_0']))) + Kex
        a_0 = 1./V_0*(0.5*(FeO_liq.params['a_0']*FeO_liq.params['V_0'] 
                           + Fe_liq.params['a_0']*Fe_liq.params['V_0'])) + aex
        Kprime_0 = V_0*(1. / (0.5*(FeO_liq.params['V_0']/(FeO_liq.params['Kprime_0']+1.)
                                   + Fe_liq.params['V_0']/(Fe_liq.params['Kprime_0']+1.)))) \
                                   - 1.
        #Kprime_0 = 2./(1./FeO_liq.params['Kprime_0'] + 1./Fe_liq.params['Kprime_0'])
        self.params = {
            'name': 'Liquid iron excesses',
            'formula': formula,
            'equation_of_state': 'v_ag',
            'H_0': (FeO_liq.params['H_0'] + Fe_liq.params['H_0'])/2. + Hex,
            'S_0': (FeO_liq.params['S_0'] + Fe_liq.params['S_0'])/2. - Sex,
            'Cp': [(FeO_liq.params['Cp'][0] + Fe_liq.params['Cp'][0])/2.,
                   (FeO_liq.params['Cp'][1] + Fe_liq.params['Cp'][1])/2.,
                   (FeO_liq.params['Cp'][2] + Fe_liq.params['Cp'][2])/2.,
                   (FeO_liq.params['Cp'][3] + Fe_liq.params['Cp'][3])/2.],
            'V_0': V_0,
            'K_0': K_0 ,
            'Kprime_0': Kprime_0, # 4.425
            'a_0': a_0,
            'delta_0': (FeO_liq.params['delta_0'] + Fe_liq.params['delta_0'])/2.,
            'kappa': (FeO_liq.params['kappa'] + Fe_liq.params['kappa'])/2.,
            'T_0': (FeO_liq.params['T_0'] + Fe_liq.params['T_0'])/2.}
        burnman.Mineral.__init__(self)
        
# Ambient pressure values are from Kowalski and Spencer (1995)
# Taken from Frost et al, 2010
# e.g. H0 is WFe-FeO/4 at 0K
#H0 = 83307./4.
#S0 = 8.978/4.
#H1 = 135943./4.
#S1 = 31.122/4.

def fit_data(xdata, H0, H1, S0, S1, V0, V1, K0, K1):
    print V0, V1, K0, K1
    # initialise properties
    intermediate_0 = Fe_FeO_liq(H0, S0, V0, K0, 0.0)
    intermediate_1 = Fe_FeO_liq(H1, S1, V1, K1, 0.0)
    
    excesses = []
    for x in xdata:
        P = x[0]
        T = x[1]
        
        if x[-1] == 0:
            Gex = 4.*(intermediate_0.calcgibbs(P,T) - 0.5*(Fe_liq.calcgibbs(P,T) + FeO_liq.calcgibbs(P,T)))
        else:
            Gex = 4.*(intermediate_1.calcgibbs(P,T) - 0.5*(Fe_liq.calcgibbs(P,T) + FeO_liq.calcgibbs(P,T)))
        excesses.append(Gex)
    return excesses



xdata=[]
ydata=[]
sigmas=[]
for i, P in enumerate(pressures):
    xdata.append([pressures[i], temperatures[i], 0])
    ydata.append(Hex0[i])
    xdata.append([pressures[i], temperatures[i], 1])
    ydata.append(Hex1[i])
    sigmas.append(1./weighting[i])
    sigmas.append(1./weighting[i])

guesses = [0., 0., 0., 0., -0.84e-6, -0.56e-6, 75.e9, 66.e9]
popt, pcov = optimize.curve_fit(fit_data, xdata, ydata, guesses, sigmas)

print popt, pcov

H0, H1, S0, S1, V0, V1, K0, K1 = popt
intermediate_0 = Fe_FeO_liq(H0, S0, V0, K0, 0.0)
intermediate_1 = Fe_FeO_liq(H1, S1, V1, K1, 0.0)


Gex0 = []
Gex1 = []
for i, P in enumerate(pressures):
    T = temperatures[i]
    
    Gex0.append(4.*(intermediate_0.calcgibbs(P,T) \
                    - 0.5*(Fe_liq.calcgibbs(P,T) + FeO_liq.calcgibbs(P,T))))
    Gex1.append(4.*(intermediate_1.calcgibbs(P,T) \
                    - 0.5*(Fe_liq.calcgibbs(P,T) + FeO_liq.calcgibbs(P,T))))


plt.plot(pressures, Gex0, marker='o', linestyle='None', label='0, Fit')
plt.plot(pressures, Gex1, marker='o', linestyle='None', label='1 , Fit')



pressures = np.linspace(1.e9, 200.e9, 100)
temperatures = [2273., 3273.]

W0 = np.empty_like(pressures)
W1 = np.empty_like(pressures)
W0_Frost = np.empty_like(pressures)
W1_Frost = np.empty_like(pressures)
ideal = np.empty_like(pressures)

for T in temperatures:
    print 'Calculating interaction parameters at', T, 'K'
    for i, P in enumerate(pressures):

        
        W0[i] = 4.*(intermediate_0.calcgibbs(P,T) \
                    - 0.5*(Fe_liq.calcgibbs(P,T) + FeO_liq.calcgibbs(P,T)))
        W1[i] = 4.*(intermediate_1.calcgibbs(P,T) \
                    - 0.5*(Fe_liq.calcgibbs(P,T) + FeO_liq.calcgibbs(P,T)))
        
        W0_Frost[i] = 135943. - 31.122*T - 0.059*P/1.e5 # P in bars
        W1_Frost[i] = 83307. - 8.978*T - 0.09*P/1.e5
        
        ideal[i] = 0.
        
    plt.plot(pressures, W0, '--', label='W0, '+str(T))
    plt.plot(pressures, W1, '.', label='W1, '+str(T))
    plt.plot(pressures, W0_Frost, label='W0 Frost, '+str(T))
    plt.plot(pressures, W1_Frost, label='W1 Frost, '+str(T))
    plt.plot(pressures, ideal, label='ideal (Komabayashi)')

plt.legend(loc='lower left')
plt.show()



####
# PLOT COMPARISONS WITH PUBLISHED DATA
####
eutectic_PT = np.array(eutectic_PT).T
eutectic_PTc = np.array(eutectic_PTc).T
solvus_PTcc = np.array(solvus_PTcc).T

def eqm_liquid(cT, P, model, Fe_phase, FeO_phase):
    c, T = cT

    model.set_composition([1.-c, c])
    model.set_state(P, T)
    partial_excesses = model.excess_partial_gibbs
    equations = [ Fe_phase.calcgibbs(P, T) - ( Fe_liq.calcgibbs(P, T) + partial_excesses[0] ),
                  FeO_phase.calcgibbs(P, T) - ( FeO_liq.calcgibbs(P, T) + partial_excesses[1] )]
    return equations



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

def eutectic_liquid(cT, P, model, intermediate_0, intermediate_1, Fe_phase, FeO_phase):
    c, T = cT

    Gex_0 = 4.*(intermediate_0.calcgibbs(P, T) \
                - 0.5*(Fe_liq.calcgibbs(P, T) + FeO_liq.calcgibbs(P, T)))
    Gex_1 = 4.*(intermediate_1.calcgibbs(P, T) \
                - 0.5*(Fe_liq.calcgibbs(P, T) + FeO_liq.calcgibbs(P, T)))

    model.enthalpy_interaction = [[[Gex_0, Gex_1]]]
    burnman.SolidSolution.__init__(model, [1.-c, c])
    model.set_state(P, T)

    partial_excesses = model.excess_partial_gibbs
    equations = [ Fe_phase.calcgibbs(P, T) - ( Fe_liq.calcgibbs(P, T) + partial_excesses[0] ),
                  FeO_phase.calcgibbs(P, T) - ( FeO_liq.calcgibbs(P, T) + partial_excesses[1] ) ]
    return equations



intermediate_0 = Fe_FeO_liq(H0, S0, V0, K0, 0.0)
intermediate_1 = Fe_FeO_liq(H1, S1, V1, K1, 0.0)

print ''
print Fe_liq.params['V_0'], Fe_liq.params['K_0'], Fe_liq.params['V_0']*Fe_liq.params['K_0']
print FeO_liq.params['V_0'], FeO_liq.params['K_0'], FeO_liq.params['V_0']*FeO_liq.params['K_0']
print intermediate_0.params['V_0'], intermediate_0.params['K_0'], intermediate_0.params['V_0']*intermediate_0.params['K_0']
print intermediate_1.params['V_0'], intermediate_1.params['K_0'], intermediate_1.params['V_0']*intermediate_1.params['K_0']
print ''
print Fe_liq.params['Kprime_0'], FeO_liq.params['Kprime_0'], intermediate_0.params['Kprime_0'],  intermediate_1.params['Kprime_0']



# Plot solvus
temperatures = [2173., 2273., 2373., 2473., 2573.]
pressures = np.linspace(1.e5, 28.e9, 30)

compositions_1 = np.empty_like(pressures)
compositions_2 = np.empty_like(pressures)


for T in temperatures:
    c1=0.01
    c2=0.99
    for i, P in enumerate(pressures):

        
        Gex_0 = 4.*(intermediate_0.calcgibbs(P, T) \
                    - 0.5*(Fe_liq.calcgibbs(P, T) + FeO_liq.calcgibbs(P, T)))
        Gex_1 = 4.*(intermediate_1.calcgibbs(P, T) \
                    - 0.5*(Fe_liq.calcgibbs(P, T) + FeO_liq.calcgibbs(P, T)))
        model.enthalpy_interaction = [[[Gex_0, Gex_1]]]
        burnman.SolidSolution.__init__(model)
        c1, c2 = optimize.fsolve(eqm_two_liquid, [c1, c2], 
                                 args=(P, T, model), factor = 0.1, xtol=1.e-12)
        compositions_1[i] = c1
        compositions_2[i] = c2
    plt.plot(compositions_1, pressures/1.e9, label='Metallic')
    plt.plot(compositions_2, pressures/1.e9, label='Ionic')

plt.plot(solvus_PTcc[4], solvus_PTcc[0]/1.e9, marker='o', linestyle='None')
plt.plot(solvus_PTcc[6], solvus_PTcc[0]/1.e9, marker='o', linestyle='None')

plt.legend(loc='lower right')
plt.show()



# Plot eutectic temperatures and compositions
pressures = np.linspace(30.e9, 250.e9, 100)
eutectic_compositions = np.empty_like(pressures)
eutectic_temperatures = np.empty_like(pressures)

c, T = [0.7, 5000.]
for i, P in enumerate(pressures):
    c, T = optimize.fsolve(eutectic_liquid, [c, T], 
                           args=(P, model, intermediate_0, intermediate_1, Fe_fcc, FeO))
    print P, T, model.enthalpy_interaction
    eutectic_compositions[i] = c
    eutectic_temperatures[i] = T


plt.plot(pressures/1.e9, eutectic_compositions)
plt.plot(eutectic_PTc[0]/1.e9, eutectic_PTc[4], marker='o', linestyle='None', label='Model')
plt.legend(loc='lower right')
plt.show()

plt.plot(pressures/1.e9, eutectic_temperatures)
plt.plot(eutectic_PT[0]/1.e9, eutectic_PT[2], marker='o', linestyle='None', label='Model')
plt.plot(eutectic_PTc[0]/1.e9, eutectic_PTc[2], marker='o', linestyle='None', label='Model')
plt.legend(loc='lower right')
plt.show()
