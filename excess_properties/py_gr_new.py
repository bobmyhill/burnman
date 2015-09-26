import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from scipy.optimize import fsolve, curve_fit
from make_intermediate import *
from fitting_functions import fit_PVT_data

# First, let's define the endmembers
pyrope = minerals.HP_2011_ds62.py()
grossular = minerals.HP_2011_ds62.gr()

print pyrope.params['K_0']/1.e9
print grossular.params['K_0']/1.e9

def symmetric(X, a, b, Wij):
    return a*(1.-X) + b*(X) + Wij*X*(1.-X)
def asymmetric(X, a, b, Wij, Wji):
    return a*(1.-X) + b*(X) + Wij*X*X*(1.-X) +  + Wji*X*(1.-X)*(1.-X)

# ENTHALPIES
# See Newton et al., 1977
c_obs = [0.0, 0.0, 0.0, 0.09, 0.18,
         0.275, 0.47, 0.8, 0.9, 1.0]
deltaH_sol = [27.73, 27.19, 27.44, 27.11, 28.33,
              29.29, 32.55, 38.37, 39.75, 42.21]
deltaH_sol_err = [0.28, 0.24, 0.11, 0.34, 0.24,
                  0.28, 0.25, 0.19, 0.15, 0.41]

c_obs = [0.0, 0.09, 0.18,
         0.275, 0.47, 0.8, 0.9, 1.0]
deltaH_sol = [27.19, 27.11, 28.33,
              29.29, 32.55, 38.37, 39.75, 42.21]
deltaH_sol_err = [0.24, 0.34, 0.24,
                  0.28, 0.25, 0.19, 0.15, 0.41]


kcaltoJ = 4184.
deltaH_sol = np.array(deltaH_sol) * kcaltoJ
deltaH_sol_err = np.array(deltaH_sol_err) * kcaltoJ
    
guesses = [266., 266., 10., 10.]
popt, pcov =  curve_fit(asymmetric, c_obs, deltaH_sol, guesses)

for i, p in enumerate(popt):
    print p, np.sqrt(pcov[i][i])


H_ex970K_0 = -popt[2]
H_ex970K_1 = -popt[3]

    
compositions = np.linspace(0.0, 1.0, 101)
plt.plot(compositions, asymmetric(compositions, *popt))
plt.errorbar(c_obs, deltaH_sol, yerr=deltaH_sol_err, linestyle='None')
plt.xlim(-0.01, 1.01)
plt.show()

# ENTROPIES
# See Dachs and Geiger (2006)
c_obs = np.array([0.0, 0.102, 0.102, 0.102, 0.264, 0.400, 0.414,
                  0.512, 0.760, 0.904, 0.904, 1.0, 1.0])
S_0 = [265.94, 268.09, 268.65, 268.06, 267.48, 265.11, 265.44, 265.48, 262.44, 259.58, 259.23, 259.47, 257.86]
S_0err = [0.23, 0.22, 0.21, 0.27, 0.24, 0.22, 0.24, 0.23, 0.25, 0.23, 0.22, 0.25, 0.26]

# Pyrope datapoint removed (0.0, 265.94, 0.23)
c_obs = np.array([0.102, 0.102, 0.102, 0.264, 0.400, 0.414,
                  0.512, 0.760, 0.904, 0.904, 1.0, 1.0])
S_0 = [268.09, 268.65, 268.06, 267.48, 265.11, 265.44, 265.48, 262.44, 259.58, 259.23, 259.47, 257.86]
S_0err = [0.22, 0.21, 0.27, 0.24, 0.22, 0.24, 0.23, 0.25, 0.23, 0.22, 0.25, 0.26]

guesses = [266., 266., 10.]
popt, pcov =  curve_fit(symmetric, c_obs, S_0, guesses, S_0err)

print ''
for i, p in enumerate(popt):
    print p, np.sqrt(pcov[i][i])


pyrope.params['S_0'] = popt[0]
grossular.params['S_0'] = popt[1]
S_ex0 = popt[2]
S_ex1 = popt[2]

compositions = np.linspace(0.0, 1.0, 101)
plt.plot(compositions, symmetric(compositions, *popt))
plt.errorbar(c_obs, S_0, yerr=S_0err, linestyle='None')
plt.xlim(-0.01, 1.01)
plt.show()

H_ex0 = H_ex970K_0
H_ex1 = H_ex970K_1 

# VOLUMES

# Now, let's read in the data from Du et al., 2015
f = open('data/py_gr_PVT_data/Du_et_al_2015_py_gr_PVT.dat')
data000 = []
data195 = []
data397 = []
data597 = []
data790 = []
data100 = []

data = []
data2 = []
RT_data = []
RP_data = []
datalines = [ line.strip().split() for idx, line in enumerate(f.read().split('\n')) if line.strip() ]
for content in datalines:
    if content[0] != '%':
        if content[0] == '0':
            data000.append([float(content[1])*1.e9, float(content[2]), float(content[3])*1.e-6, float(content[4])*1.e-6])
        if content[0] == '0.195':
            data195.append([float(content[1])*1.e9, float(content[2]), float(content[3])*1.e-6, float(content[4])*1.e-6])
        if content[0] == '0.397':
            data397.append([float(content[1])*1.e9, float(content[2]), float(content[3])*1.e-6, float(content[4])*1.e-6])
        if content[0] == '0.597':
            data597.append([float(content[1])*1.e9, float(content[2]), float(content[3])*1.e-6, float(content[4])*1.e-6])
        if content[0] == '0.790':
            data790.append([float(content[1])*1.e9, float(content[2]), float(content[3])*1.e-6, float(content[4])*1.e-6])
        if content[0] == '1':
            data100.append([float(content[1])*1.e9, float(content[2]), float(content[3])*1.e-6, float(content[4])*1.e-6])

        if content[0] != '0.790' and content[0] != '0.195':
            data.append(map(float,content))
        if content[0] != '0.397' and content[0] != '0.597':
            data2.append(map(float,content))
        if float(content[2]) < 299.:
            RT_data.append(map(float,content))
        if float(content[1]) < 0.001:
            RP_data.append(map(float,content))

data000 = zip(*data000)
data195 = zip(*data195)
data397 = zip(*data397)
data597 = zip(*data597)
data790 = zip(*data790)
data100 = zip(*data100)


# First, let's find best fits to initial equations of state:
pyrope000 = minerals.HP_2011_ds62.py()
pyrope195 = minerals.HP_2011_ds62.py()
pyrope397 = minerals.HP_2011_ds62.py()
pyrope597 = minerals.HP_2011_ds62.py()
pyrope790 = minerals.HP_2011_ds62.py()
pyrope100 = minerals.HP_2011_ds62.py()

p_py_phases = [[0.000, pyrope000, data000],
               [0.195, pyrope195, data195],
               [0.397, pyrope397, data397],
               [0.597, pyrope597, data597],
               [0.790, pyrope790, data790],
               [1.000, pyrope100, data100]]

guesses = [pyrope.params['V_0'], pyrope.params['K_0'], pyrope.params['a_0']]
cVKa = []

for p_py_phase in p_py_phases:
    p_py, phase, d = p_py_phase
    popt, pcov = curve_fit(fit_PVT_data(phase, pyrope.params['Kprime_0']*p_py + grossular.params['Kprime_0']*(1.-p_py)), \
                               zip(*[d[0], d[1]]), d[2], guesses, d[3])
    print popt
    V_0, K_0, a_0 = popt
    V_0_err = np.sqrt(pcov[0][0])
    K_0_err = np.sqrt(pcov[1][1])
    a_0_err = np.sqrt(pcov[2][2])
    cVKa.append([p_py, V_0, V_0_err, K_0, K_0_err, a_0, a_0_err])

cVKa = np.array(zip(*cVKa))

RT_data = zip(*RT_data)
RP_data = zip(*RP_data)

p_py, P, T, V, Verr = zip(*data)
p_py2, P2, T2, V2, Verr2 = zip(*data2)

p_py_obs = np.array(p_py)
P_obs = np.array(P)*1.e9
T_obs = np.array(T)
V_obs = np.array(V)*1.e-6
Verr_obs =  np.array(V)*1.e-6

p_py_obs2 = np.array(p_py2)
P_obs2 = np.array(P2)*1.e9
T_obs2 = np.array(T2)
V_obs2 = np.array(V2)*1.e-6
Verr_obs2 =  np.array(V2)*1.e-6


# First, let's define our intermediates
pygr = minerals.HP_2011_ds62.py()
grpy = minerals.HP_2011_ds62.py()

pygr2 = minerals.HP_2011_ds62.py()
grpy2 = minerals.HP_2011_ds62.py()

Sconf = -2.*burnman.constants.gas_constant*0.5*3.*np.log(0.5) # 2 atoms mixing, equal proportions (0.5) on 3 sites

# Cp_scaling = ((py.params['S_0'] + gr.params['S_0'])*0.5 + S_ex/4)/((py.params['S_0'] + gr.params['S_0'])*0.5)
# Haselton and Westrum (1980) show that the excess entropy is primarily a result of a low temperature spike in Cp
# Therefore, at >=298.15 K, Cp is well approximated by a linear combination of pyrope and grossular 
Cp_scaling = 1. 

Cp_pygr = [(pyrope.params['Cp'][0] + grossular.params['Cp'][0])*0.5*Cp_scaling,
           (pyrope.params['Cp'][1] + grossular.params['Cp'][1])*0.5*Cp_scaling,
           (pyrope.params['Cp'][2] + grossular.params['Cp'][2])*0.5*Cp_scaling,
           (pyrope.params['Cp'][3] + grossular.params['Cp'][3])*0.5*Cp_scaling]

Cp_pygr2 = [(pyrope.params['Cp'][0] + grossular.params['Cp'][0])*0.5*Cp_scaling,
           (pyrope.params['Cp'][1] + grossular.params['Cp'][1])*0.5*Cp_scaling,
           (pyrope.params['Cp'][2] + grossular.params['Cp'][2])*0.5*Cp_scaling,
           (pyrope.params['Cp'][3] + grossular.params['Cp'][3])*0.5*Cp_scaling]

pygr.params['H_0'] = 0.5*(pyrope.params['H_0'] + grossular.params['H_0']) + H_ex0/4.
grpy.params['H_0'] = 0.5*(pyrope.params['H_0'] + grossular.params['H_0']) + H_ex1/4.
pygr.params['S_0'] = 0.5*(pyrope.params['S_0'] + grossular.params['S_0']) + Sconf + S_ex0/4.
grpy.params['S_0'] = 0.5*(pyrope.params['S_0'] + grossular.params['S_0']) + Sconf + S_ex1/4.

pygr2.params['H_0'] = 0.5*(pyrope.params['H_0'] + grossular.params['H_0']) + H_ex0/4
grpy2.params['H_0'] = 0.5*(pyrope.params['H_0'] + grossular.params['H_0']) + H_ex1/4.
pygr2.params['S_0'] = 0.5*(pyrope.params['S_0'] + grossular.params['S_0']) + Sconf + S_ex0/4.
grpy2.params['S_0'] = 0.5*(pyrope.params['S_0'] + grossular.params['S_0']) + Sconf + S_ex1/4.

class mg_fe_ca_garnet_Ganguly(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Subregular pyrope-almandine-grossular garnet'
        self.type='subregular'
        self.endmembers = [[minerals.HP_2011_ds62.py(), '[Mg]3[Al]2Si3O12'],[minerals.HP_2011_ds62.alm(), '[Fe]3[Al]2Si3O12'],[minerals.HP_2011_ds62.gr(), '[Ca]3[Al]2Si3O12'], [minerals.HP_2011_ds62.spss(), '[Mn]3[Al]2Si3O12']]
        self.enthalpy_interaction=[[[2117., 695.], [9834., 21627.], [12083., 12083.]],[[6773., 873.],[539., 539.]],[[0., 0.]]]
        self.volume_interaction=[[[0.07e-5, 0.], [0.058e-5, 0.012e-5], [0.04e-5, 0.03e-5]],[[0.03e-5, 0.],[0.04e-5, 0.01e-5]],[[0., 0.]]]
        self.entropy_interaction=[[[0., 0.], [5.78, 5.78], [7.67, 7.67]],[[1.69, 1.69],[0., 0.]],[[0., 0.]]]
        
        # Published values are on a 4-oxygen (1-cation) basis
        for interaction in [self.enthalpy_interaction, self.volume_interaction, self.entropy_interaction]:
            for i in range(len(interaction)):
                for j in range(len(interaction[i])):
                    for k in range(len(interaction[i][j])):
                        interaction[i][j][k]*=3.
                        
        burnman.SolidSolution.__init__(self, molar_fractions)

garnet_ganguly = mg_fe_ca_garnet_Ganguly()
        
# Now, let's set up a solid solution model
class pyrope_grossular_binary(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Pyrope-grossular binary'
        self.type='full_subregular'
        self.endmembers = [[pyrope,  '[Mg]3Al2Si3O12'],
                           [grossular, '[Ca]3Al2Si3O12']]
        self.intermediates=[[[pygr, grpy]]]
        
        burnman.SolidSolution.__init__(self, molar_fractions)

garnet_ordered = pyrope_grossular_binary()

# Now, let's set up a second solid solution model
class pyrope_grossular_binary2(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Pyrope-grossular binary'
        self.type='full_subregular'
        self.endmembers = [[pyrope,  '[Mg]3Al2Si3O12'],
                           [grossular, '[Ca]3Al2Si3O12']]
        self.intermediates=[[[pygr2, grpy2]]]
        
        burnman.SolidSolution.__init__(self, molar_fractions)

garnet_disordered = pyrope_grossular_binary2()

def fit_model(solid_solution):
    def fit_ss_data(data, Vi0, Ki0, ai0):
        Vi1 = Vi0
        Ki1 = Ki0
        ai1 = ai0
        
        V_int = Vi0
        Kpi0 = V_int/(0.5*(pyrope.params['V_0']/(pyrope.params['Kprime_0']+1.) \
                               + grossular.params['V_0']/(grossular.params['Kprime_0']+1.))) - 1.
        Kpi1=Kpi0
        
        solid_solution.intermediates[0][0][0].params['V_0'] = Vi0
        solid_solution.intermediates[0][0][0].params['K_0'] = Ki0
        solid_solution.intermediates[0][0][0].params['Kprime_0'] = Kpi0
        solid_solution.intermediates[0][0][0].params['Kdprime_0'] = -Kpi0/Ki0
        solid_solution.intermediates[0][0][0].params['a_0'] = ai0
        
        solid_solution.intermediates[0][0][1].params['V_0'] = Vi0
        solid_solution.intermediates[0][0][1].params['K_0'] = Ki0
        solid_solution.intermediates[0][0][1].params['Kprime_0'] = Kpi0
        solid_solution.intermediates[0][0][1].params['Kdprime_0'] = -Kpi0/Ki0
        solid_solution.intermediates[0][0][1].params['a_0'] = ai0
        
        volumes = []
        for datum in data:
            c, P, T = datum
            solid_solution.set_composition([c, 1.-c]) # pyrope first, p_py from data
            solid_solution.set_state(P, T)
            volumes.append(solid_solution.V)

        return volumes
    return fit_ss_data


cPT_obs = zip(*[p_py, P_obs, T_obs])
cPT_obs2 = zip(*[p_py2, P_obs2, T_obs2])
guesses = [pyrope.params['V_0'], pyrope.params['K_0'], pyrope.params['a_0']]

popt, pcov = curve_fit(fit_model(garnet_ordered), cPT_obs, V_obs, guesses, Verr_obs)
popt2, pcov2 = curve_fit(fit_model(garnet_disordered), cPT_obs2, V_obs2, guesses, Verr_obs2)

for i, p in enumerate(popt):
    print p, np.sqrt(pcov[i][i])

for i, p in enumerate(popt2):
    print p, np.sqrt(pcov2[i][i])

#H_ex, S_ex, Sconf, V_ex, K_ex, a_ex
pygr_params = [H_ex0/4., S_ex0/4., Sconf, 0.058e-5*3./4., -2.5e9, 0.]
grpy_params = [H_ex0/4., S_ex0/4., Sconf, 0.012e-5*3./4., -2.5e9, 0.]

pygr3 = make_intermediate(pyrope, grossular, pygr_params)
grpy3 = make_intermediate(pyrope, grossular, grpy_params)

pygr3.params['a_0'] = 0.5*pygr3.params['V_0'] \
    * ( pyrope.params['a_0']/pyrope.params['V_0'] \
            + grossular.params['a_0']/grossular.params['V_0'])
grpy3.params['a_0'] = 0.5*grpy3.params['V_0'] \
    * ( pyrope.params['a_0']/pyrope.params['V_0'] \
            + grossular.params['a_0']/grossular.params['V_0'])

ksi = 6.
pygr3.params['K_0'] = 0.5*(pyrope.params['K_0'] + grossular.params['K_0']) + ksi \
    * ( ( pyrope.params['K_0']*pyrope.params['V_0'] + grossular.params['V_0']*grossular.params['K_0'] ) \
/ ( 2.*pygr3.params['V_0']) - 0.5*(pyrope.params['K_0'] + grossular.params['K_0']))

grpy3.params['K_0'] = 0.5*(pyrope.params['K_0'] + grossular.params['K_0']) + ksi \
    * ( ( pyrope.params['K_0']*pyrope.params['V_0'] + grossular.params['V_0']*grossular.params['K_0'] ) \
/ ( 2.*grpy3.params['V_0']) - 0.5*(pyrope.params['K_0'] + grossular.params['K_0']))

print pygr3.params['K_0'] , grpy3.params['K_0']
print pygr3.params['a_0'] , grpy3.params['a_0']

# Finally, let's set up a Ganguly-like model but with an excess bulk modulus according to our heuristics
class pyrope_grossular_binary3(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Pyrope-grossular binary'
        self.type='full_subregular'
        self.endmembers = [[pyrope,  '[Mg]3Al2Si3O12'],
                           [grossular, '[Ca]3Al2Si3O12']]
        self.intermediates=[[[pygr3, grpy3]]]
        
        burnman.SolidSolution.__init__(self, molar_fractions)

garnet_ganguly_inc_Kex = pyrope_grossular_binary3()


# First, let's look at the excess bulk moduli under room conditions
# Also print this data to file
filename = 'figures/data/pyrope_grossular_K_0.dat'
f = open(filename, 'w')

bulk_moduli_garnet_ganguly = np.empty_like(compositions)
bulk_moduli_garnet = np.empty_like(compositions)
bulk_moduli_garnet2 = np.empty_like(compositions)
bulk_moduli_garnet3 = np.empty_like(compositions)

for i, c in enumerate(compositions):
    garnet_ordered.set_composition([c, 1.-c])
    garnet_disordered.set_composition([c, 1.-c])
    garnet_ganguly_inc_Kex.set_composition([c, 1.-c])
    garnet_ganguly.set_composition([c, 0., 1.-c, 0.])

    garnet_ordered.set_state(1.e5, 298.15)
    garnet_disordered.set_state(1.e5, 298.15)
    garnet_ganguly_inc_Kex.set_state(1.e5, 298.15)
    garnet_ganguly.set_state(1.e5, 298.15)


    bulk_moduli_garnet[i] = garnet_ordered.K_T
    bulk_moduli_garnet2[i] = garnet_disordered.K_T
    bulk_moduli_garnet3[i] = garnet_ganguly_inc_Kex.K_T
    bulk_moduli_garnet_ganguly[i] = garnet_ganguly.K_T

    f.write(str(c)+' '+str(bulk_moduli_garnet[i]*1.e-9) \
                +' '+str(bulk_moduli_garnet2[i]*1.e-9) \
                +' '+str(bulk_moduli_garnet3[i]*1.e-9) \
                +' '+str(bulk_moduli_garnet_ganguly[i]*1.e-9) \
                +'\n')

f.write('\n')
f.close()
print 'Data (over)written to file', filename

# Also print observed values
filename = 'figures/data/pyrope_grossular_K_0_obs.dat'
f = open(filename, 'w')
for datum in zip(*cVKa):
    f.write(str(datum[0])+' '+str(datum[3]/1.e9)+' '+str(datum[4]/1.e9)+'\n')
f.close()
print 'Data (over)written to file', filename

plt.plot(compositions, bulk_moduli_garnet/1.e9, marker='o', linestyle='None')
plt.plot(compositions, bulk_moduli_garnet2/1.e9, marker='.', linestyle='None')
plt.plot(compositions, bulk_moduli_garnet3/1.e9, marker='x', linestyle='None')
plt.plot(compositions, bulk_moduli_garnet_ganguly/1.e9)
plt.errorbar(cVKa[0], cVKa[3]/1.e9, yerr=cVKa[4]/1.e9, linestyle='None', marker='o')
plt.show()
    
# Plot volumes obtained from the model
# Also print this data to file
filename = 'figures/data/pyrope_grossular_P_V.dat'
f = open(filename, 'w')

compositions = np.array([0.0, 0.195, 0.397, 0.597, 0.790, 1.0])
pressures = np.linspace(1.e5, 10.e9, 101)
for c in compositions:
    garnet_ordered.set_composition([c, 1.-c])
    volumes = np.empty_like(pressures)
    f.write('>>\n')
    for i, P in enumerate(pressures):
        garnet_ordered.set_state(P, 298.15)
        volumes[i] = garnet_ordered.V*1.e6

        f.write(str(P/1.e9)+' '+str(volumes[i])+'\n')

    plt.plot(pressures/1.e9, volumes, label='Py'+str(int(c*100.)))
    

f.write('\n')
f.close()
print 'Data (over)written to file', filename


plt.errorbar(RT_data[1], RT_data[3], yerr=RT_data[4], marker='.', linestyle='None')
plt.legend(loc='lower left')
plt.show()


temperatures = np.linspace(300., 1000., 101)
for c in compositions:
    garnet_ordered.set_composition([c, 1.-c])
    volumes = np.empty_like(pressures)
    for i, T in enumerate(temperatures):
        garnet_ordered.set_state(1.e5, T)
        volumes[i] = garnet_ordered.V*1.e6
    plt.plot(temperatures, volumes, label='Py'+str(int(c*100.)))
    
plt.errorbar(RP_data[2], RP_data[3], yerr=RP_data[4], marker='.', linestyle='None')
plt.legend(loc='lower left')
plt.show()



# Plot excess volumes, gibbs, bulk sound in the middle of the binary
# Also print this data to file
filename = 'figures/data/pyrope_grossular_bulk_sound_velocities.dat'
f = open(filename, 'w')

pressures = np.linspace(1.e5, 25.e9, 101)
excess_volumes = np.empty_like(pressures)
excess_volumes2 = np.empty_like(pressures)
excess_volumes3 = np.empty_like(pressures)

bulk_sound = np.empty_like(pressures)
bulk_sound2 = np.empty_like(pressures)
bulk_sound3 = np.empty_like(pressures)

excess_gibbs_ganguly = np.empty_like(pressures)
excess_gibbs = np.empty_like(pressures)
excess_gibbs2 = np.empty_like(pressures)
excess_gibbs3 = np.empty_like(pressures)

excess_volumes_ganguly = np.empty_like(pressures)
bulk_sound_ganguly = np.empty_like(pressures)

garnet_ordered.set_composition([0.5, 0.5])
garnet_disordered.set_composition([0.5, 0.5])
garnet_ganguly_inc_Kex.set_composition([0.5, 0.5])
garnet_ganguly.set_composition([0.5, 0.0, 0.5, 0.0])
for i, P in enumerate(pressures):
    garnet_ordered.set_state(P, 298.15)
    garnet_disordered.set_state(P, 298.15)
    garnet_ganguly_inc_Kex.set_state(P, 298.15)
    garnet_ganguly.set_state(P, 298.15)

    excess_volumes[i] = garnet_ordered.excess_volume
    excess_volumes2[i] = garnet_disordered.excess_volume
    excess_volumes3[i] = garnet_ganguly_inc_Kex.excess_volume
    excess_volumes_ganguly[i] = garnet_ganguly.excess_volume

    excess_gibbs[i] = garnet_ordered.excess_gibbs
    excess_gibbs2[i] = garnet_disordered.excess_gibbs
    excess_gibbs3[i] = garnet_ganguly_inc_Kex.excess_gibbs
    excess_gibbs_ganguly[i] = garnet_ganguly.excess_gibbs

    bulk_sound[i] = np.sqrt(garnet_ordered.K_S/garnet_ordered.rho)
    bulk_sound2[i] = np.sqrt(garnet_disordered.K_S/garnet_disordered.rho)
    bulk_sound3[i] = np.sqrt(garnet_ganguly_inc_Kex.K_S/garnet_ganguly_inc_Kex.rho)
    bulk_sound_ganguly[i] = np.sqrt(garnet_ganguly.K_S/garnet_ganguly.rho)

    f.write(str(P/1.e9)+' '+str(bulk_sound[i]*1.e-3) \
                +' '+str(bulk_sound2[i]*1.e-3) \
                +' '+str(bulk_sound3[i]*1.e-3) \
                +' '+str(bulk_sound_ganguly[i]*1.e-3) \
                +'\n')
    
f.write('\n')
f.close()
print 'Data (over)written to file', filename

plt.plot(pressures/1.e9, excess_gibbs*1.e-3)
plt.plot(pressures/1.e9, excess_gibbs2*1.e-3)
plt.plot(pressures/1.e9, excess_gibbs3*1.e-3)
plt.plot(pressures/1.e9, excess_gibbs_ganguly*1.e-3)
plt.xlabel('Pressure (GPa)')
plt.ylabel('Excess gibbs (kJ/mol)')
plt.show()

plt.plot(pressures/1.e9, excess_volumes*1.e6)
plt.plot(pressures/1.e9, excess_volumes2*1.e6)
plt.plot(pressures/1.e9, excess_volumes3*1.e6)
plt.plot(pressures/1.e9, excess_volumes_ganguly*1.e6)
plt.xlabel('Pressure (GPa)')
plt.ylabel('Excess volume (cm^3/mol)')
plt.show()

plt.plot(pressures/1.e9, bulk_sound*1.e-3)
plt.plot(pressures/1.e9, bulk_sound2*1.e-3)
plt.plot(pressures/1.e9, bulk_sound3*1.e-3)
plt.plot(pressures/1.e9, bulk_sound_ganguly*1.e-3)
plt.xlabel('Pressure (GPa)')
plt.ylabel('Bulk sound velocity (km/s)')
plt.show()


# Check KV rule of thumb
garnet_ordered.set_composition([0.5, 0.5])
garnet_ordered.set_state(1.e5, 298.15)
K_T = garnet_ordered.K_T
V = garnet_ordered.V

garnet_disordered.set_composition([0.5, 0.5])
garnet_disordered.set_state(1.e5, 298.15)
K_T2 = garnet_disordered.K_T
V2 = garnet_disordered.V

garnet_ganguly_inc_Kex.set_composition([0.5, 0.5])
garnet_ganguly_inc_Kex.set_state(1.e5, 298.15)
K_T3 = garnet_ganguly_inc_Kex.K_T
V3 = garnet_ganguly_inc_Kex.V

KV_py = pyrope.params['K_0']*pyrope.params['V_0']
KV_gr = grossular.params['K_0']*grossular.params['V_0']

K_average = 0.5*(pyrope.params['K_0'] + grossular.params['K_0'])
KV_average = 0.5*(KV_py + KV_gr)

print (K_T - K_average)/(KV_average/V - K_average)
print (K_T2 - K_average)/(KV_average/V2 - K_average)
print (K_T3 - K_average)/(KV_average/V3 - K_average)

