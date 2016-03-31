import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from scipy.optimize import fsolve, curve_fit

# First, let's define the endmembers
pyrope = minerals.HP_2011_ds62.py()
grossular = minerals.HP_2011_ds62.gr()


# VOLUMES
# Now, let's read in the data from Du et al., 2015
f = open('data/py_gr_PVT_data/Du_et_al_2015_py_gr_PVT.dat')
data = []
RT_data = []
RP_data = []
datalines = [ line.strip().split() for idx, line in enumerate(f.read().split('\n')) if line.strip() ]
for content in datalines:
    if content[0] != '%':
        if content[0] != '0.790' and content[0] != '0.597' and content[0] != '0.195':
            data.append(map(float,content))
        if float(content[2]) < 299.:
            RT_data.append(map(float,content))
        if float(content[1]) < 0.001:
            RP_data.append(map(float,content))

RT_data = zip(*RT_data)
RP_data = zip(*RP_data)


p_py, P, T, V, Verr = zip(*data)

p_py_obs = np.array(p_py)
P_obs = np.array(P)*1.e9
T_obs = np.array(T)
V_obs = np.array(V)*1.e-6
Verr_obs =  np.array(V)*1.e-6

        
# Now, let's set up a solid solution model
class pyrope_grossular_binary(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Pyrope-grossular binary'
        self.type='full_subregular'
        self.P_0 = 1.e5
        self.T_0 = 298.15
        self.n_atoms = 20.
        self.endmembers = [[pyrope,  '[Mg]3Al2Si3O12'],
                           [grossular, '[Ca]3Al2Si3O12']]
        self.energy_interaction=[[[0., 0.]]]
        self.volume_interaction=[[[1.e-6, 1.e-6]]]
        self.kprime_interaction=[[[7., 7.]]]
        
        burnman.SolidSolution.__init__(self, molar_fractions)

garnet = pyrope_grossular_binary()

def fit_ss_data(data, V1, V2, Kp):
    Kppy=4.4 # pyrope.params['Kprime_0']
    Kpgr=5.5 # grossular.params['Kprime_0']


    garnet.volume_interaction = [[[V1, V2]]]
    garnet.kprime_interaction = [[[Kp, Kp]]]
    burnman.SolidSolution.__init__(garnet)
    
    volumes = []
    for datum in data:
        c, P, T = datum
        garnet.set_composition([c, 1.-c]) # pyrope first, p_py from data
        garnet.set_state(P, T)
        volumes.append(garnet.V)

    return volumes

cPT_obs = zip(*[p_py, P_obs, T_obs])
print cPT_obs
print V_obs
print
print Verr_obs
guesses = [0., 0., 7.]


popt, pcov = curve_fit(fit_ss_data, cPT_obs, V_obs, guesses)

class pyrope_grossular_binary_constant_Vex(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Pyrope-grossular binary'
        self.type='subregular'
        self.endmembers = [[pyrope,  '[Mg]3Al2Si3O12'],
                           [grossular, '[Ca]3Al2Si3O12']]
        self.enthalpy_interaction=[[[0., 0.]]]
        self.volume_interaction=[[[popt[0], popt[1]]]]
        
        burnman.SolidSolution.__init__(self, molar_fractions)

garnet_constant_Vex = pyrope_grossular_binary_constant_Vex()



class mg_fe_ca_garnet_Ganguly(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Subregular pyrope-almandine-grossular garnet'
        self.type='subregular'
        self.endmembers = [[minerals.HP_2011_ds62.py(),   '[Mg]3[Al]2Si3O12'],
                           [minerals.HP_2011_ds62.alm(),  '[Fe]3[Al]2Si3O12'],
                           [minerals.HP_2011_ds62.gr(),   '[Ca]3[Al]2Si3O12'],
                           [minerals.HP_2011_ds62.spss(), '[Mn]3[Al]2Si3O12']]
        self.enthalpy_interaction=[[[2117., 695.],
                                    [9834., 21627.],
                                    [12083., 12083.]],
                                   [[6773., 873.],
                                    [539., 539.]],
                                   [[0., 0.]]]
        self.volume_interaction=[[[0.07e-5, 0.],
                                  [0.058e-5, 0.012e-5],
                                  [0.04e-5, 0.03e-5]],
                                 [[0.03e-5, 0.],
                                  [0.04e-5, 0.01e-5]],
                                 [[0., 0.]]]
        self.entropy_interaction=[[[0., 0.],
                                   [5.78, 5.78],
                                   [7.67, 7.67]],
                                  [[1.69, 1.69],
                                   [0., 0.]],
                                  [[0., 0.]]]
        
        # Published values are on a 4-oxygen (1-cation) basis
        for interaction in [self.enthalpy_interaction, self.volume_interaction, self.entropy_interaction]:
            for i in range(len(interaction)):
                for j in range(len(interaction[i])):
                    for k in range(len(interaction[i][j])):
                        interaction[i][j][k]*=3.
                        
        burnman.SolidSolution.__init__(self, molar_fractions)

garnet_ganguly = mg_fe_ca_garnet_Ganguly()

for i, p in enumerate(popt):
    print p, np.sqrt(pcov[i][i])

    
# Plot volumes obtained from the model
# Also print this data to file
filename = 'figures/data/pyrope_grossular_P_V.dat'
f = open(filename, 'w')

compositions = np.array([0.0, 0.195, 0.397, 0.597, 0.790, 1.0])
pressures = np.linspace(1.e5, 10.e9, 101)
for c in compositions:
    garnet.set_composition([c, 1.-c])
    volumes = np.empty_like(pressures)
    f.write('>>\n')
    for i, P in enumerate(pressures):
        garnet.set_state(P, 298.15)
        volumes[i] = garnet.V*1.e6

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
    garnet.set_composition([c, 1.-c])
    volumes = np.empty_like(pressures)
    for i, T in enumerate(temperatures):
        garnet.set_state(1.e5, T)
        volumes[i] = garnet.V*1.e6
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
excess_gibbs = np.empty_like(pressures)
bulk_sound = np.empty_like(pressures)

excess_volumes_ganguly = np.empty_like(pressures)
excess_gibbs_ganguly = np.empty_like(pressures)
bulk_sound_ganguly = np.empty_like(pressures)

bulk_sound_constant_Vex = np.empty_like(pressures)

garnet.set_composition([0.5, 0.5])
garnet_constant_Vex.set_composition([0.5, 0.5])
garnet_ganguly.set_composition([0.5, 0.0, 0.5, 0.0])
for i, P in enumerate(pressures):
    garnet.set_state(P, 298.15)
    garnet_ganguly.set_state(P, 298.15)
    garnet_constant_Vex.set_state(P, 298.15)

    excess_volumes[i] = garnet.excess_volume
    excess_volumes_ganguly[i] = garnet_ganguly.excess_volume

    excess_gibbs[i] = garnet.excess_gibbs
    excess_gibbs_ganguly[i] = garnet_ganguly.excess_gibbs

    bulk_sound[i] = np.sqrt(garnet.K_S/garnet.rho)
    bulk_sound_ganguly[i] = np.sqrt(garnet_ganguly.K_S/garnet_ganguly.rho)
    bulk_sound_constant_Vex[i] = np.sqrt(garnet_constant_Vex.K_S/garnet_constant_Vex.rho)

    f.write(str(P/1.e9)+' '+str(bulk_sound[i]*1.e-3) \
                +' '+str(bulk_sound_ganguly[i]*1.e-3) \
                +' '+str(bulk_sound_constant_Vex[i]*1.e-3) \
                +'\n')
    
f.write('\n')
f.close()
print 'Data (over)written to file', filename
