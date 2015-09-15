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

# Now, let's read in the data from Du et al., 2015
f = open('data/py_gr_PVT_data/Du_et_al_2015_py_gr_PVT.dat')
data = []
RT_data = []
RP_data = []
datalines = [ line.strip().split() for idx, line in enumerate(f.read().split('\n')) if line.strip() ]
for content in datalines:
    if content[0] != '%':
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


# First, let's define our intermediates
intermediate0 = minerals.HP_2011_ds62.py()
intermediate1 = minerals.HP_2011_ds62.py()


H_ex0, H_ex1, S_ex0, S_ex1 = [9834., 21627., 5.78, 5.78] # Ganguly params
Sconf = -2.*burnman.constants.gas_constant*0.5*3.*np.log(0.5)

# Cp_scaling = ((py.params['S_0'] + gr.params['S_0'])*0.5 + S_ex/4)/((py.params['S_0'] + gr.params['S_0'])*0.5)
# Haselton and Westrum (1980) show that the excess entropy is primarily a result of a low temperature spike in Cp
# Therefore, at >=298.15 K, Cp is well approximated by a linear combination of pyrope and grossular 
Cp_scaling = 1. 

Cp_pygr = [(pyrope.params['Cp'][0] + grossular.params['Cp'][0])*0.5*Cp_scaling,
           (pyrope.params['Cp'][1] + grossular.params['Cp'][1])*0.5*Cp_scaling,
           (pyrope.params['Cp'][2] + grossular.params['Cp'][2])*0.5*Cp_scaling,
           (pyrope.params['Cp'][3] + grossular.params['Cp'][3])*0.5*Cp_scaling]

intermediate0.params['H_0'] = 0.5*(pyrope.params['H_0'] + grossular.params['H_0']) + H_ex0/4.
intermediate1.params['H_0'] = 0.5*(pyrope.params['H_0'] + grossular.params['H_0']) + H_ex1/4.
intermediate0.params['S_0'] = 0.5*(pyrope.params['S_0'] + grossular.params['S_0']) + Sconf + S_ex0/4.
intermediate1.params['S_0'] = 0.5*(pyrope.params['S_0'] + grossular.params['S_0']) + Sconf + S_ex1/4.


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
        self.name='Jadeite-grossular binary'
        self.type='full_subregular'
        self.endmembers = [[pyrope,  '[Mg]3Al2Si3O12'],
                           [grossular, '[Ca]3Al2Si3O12']]
        self.intermediates=[[[intermediate0, intermediate1]]]
        
        burnman.SolidSolution.__init__(self, molar_fractions)

garnet = pyrope_grossular_binary()


def fit_ss_data(data, Vpy, Kpy, apy, Vi0, Ki0, ai0, Vi1, Ki1, ai1, Vgr, Kgr, agr):
    Kppy=4.4 # pyrope.params['Kprime_0']
    Kpgr=5.5 # grossular.params['Kprime_0']

    V_int = 0.5*(Vi0+Vi1)
    Kpi0 = V_int/(0.5*(Vpy/(Kppy+1.) + Vgr/(Kpgr+1.))) - 1.
    Kpi1=Kpi0
    
    pyrope.params['V_0'] = Vpy
    pyrope.params['K_0'] = Kpy
    pyrope.params['Kprime_0'] = Kppy
    pyrope.params['Kdprime_0'] = -Kppy/Kpy
    pyrope.params['a_0'] = apy
    
    intermediate0.params['V_0'] = Vi0
    intermediate0.params['K_0'] = Ki0
    intermediate0.params['Kprime_0'] = Kpi0
    intermediate0.params['Kdprime_0'] = -Kpi0/Ki0
    intermediate0.params['a_0'] = ai0

    intermediate1.params['V_0'] = Vi1
    intermediate1.params['K_0'] = Ki1
    intermediate1.params['Kprime_0'] = Kpi1
    intermediate1.params['Kdprime_0'] = -Kpi1/Ki1
    intermediate1.params['a_0'] = ai1
    
    grossular.params['V_0'] = Vgr
    grossular.params['K_0'] = Kgr
    grossular.params['Kprime_0'] = Kpgr
    grossular.params['Kdprime_0'] = -Kpgr/Kgr
    grossular.params['a_0'] = agr
    
    volumes = []
    for datum in data:
        c, P, T = datum
        garnet.set_composition([c, 1.-c]) # pyrope first, p_py from data
        garnet.set_state(P, T)
        volumes.append(garnet.V)

    return volumes

cPT_obs = zip(*[p_py, P_obs, T_obs])
guesses = [pyrope.params['V_0'], pyrope.params['K_0'], pyrope.params['a_0'], \
           pyrope.params['V_0'], pyrope.params['K_0'], pyrope.params['a_0'], \
           pyrope.params['V_0'], pyrope.params['K_0'], pyrope.params['a_0'], \
           grossular.params['V_0'], grossular.params['K_0'], grossular.params['a_0']]


popt, pcov = curve_fit(fit_ss_data, cPT_obs, V_obs, guesses, Verr_obs)

for i, p in enumerate(popt):
    print p, np.sqrt(pcov[i][i])


intermediate0.params['K_0'] += 3.e9 # Tweak to make sure excess volume doesn't go negative
intermediate1.params['K_0'] += 3.e9 # Tweak to make sure excess volume doesn't go negative




# First, let's look at the excess Gibbs free energy of mixing under room conditions
compositions = np.linspace(0.0, 1.0, 101)
excess_gibbs_garnet = np.empty_like(compositions)
excess_gibbs_garnet_ganguly = np.empty_like(compositions)
for i, c in enumerate(compositions):
    garnet.set_composition([c, 1.-c])
    garnet_ganguly.set_composition([c, 0., 1.-c, 0.])

    garnet.set_state(1.e5, 298.15)
    garnet_ganguly.set_state(1.e5, 298.15)

    excess_gibbs_garnet[i] = garnet.excess_gibbs
    excess_gibbs_garnet_ganguly[i] = garnet.excess_gibbs

plt.plot(compositions, excess_gibbs_garnet, marker='o', linestyle='None')
plt.plot(compositions, excess_gibbs_garnet_ganguly)
plt.show()
    
# Plot volumes obtained from the model
compositions = np.array([0.0, 0.195, 0.397, 0.597, 0.790, 1.0])
pressures = np.linspace(1.e5, 10.e9, 101)
for c in compositions:
    garnet.set_composition([c, 1.-c])
    volumes = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        garnet.set_state(P, 298.15)
        volumes[i] = garnet.V*1.e6
    plt.plot(pressures/1.e9, volumes, label='Py'+str(int(c*100.)))
    
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



# Plot excess volumes in the  middle of the binary
pressures = np.linspace(1.e5, 25.e9, 101)
excess_volumes = np.empty_like(pressures)
excess_gibbs = np.empty_like(pressures)
excess_volumes_ganguly = np.empty_like(pressures)
excess_gibbs_ganguly = np.empty_like(pressures)

garnet.set_composition([0.5, 0.5])
garnet_ganguly.set_composition([0.5, 0.0, 0.5, 0.0])
for i, P in enumerate(pressures):
    garnet.set_state(P, 298.15)
    garnet_ganguly.set_state(P, 298.15)
    
    excess_volumes[i] = garnet.excess_volume
    excess_volumes_ganguly[i] = garnet_ganguly.excess_volume

    excess_gibbs[i] = garnet.excess_gibbs
    excess_gibbs_ganguly[i] = garnet_ganguly.excess_gibbs
    
plt.plot(pressures/1.e9, excess_volumes*1.e6)
plt.plot(pressures/1.e9, excess_volumes_ganguly*1.e6)
plt.xlabel('Pressure (GPa)')
plt.ylabel('Excess volume (cm^3/mol)')
plt.show()

plt.plot(pressures/1.e9, excess_gibbs*1.e-3)
plt.plot(pressures/1.e9, excess_gibbs_ganguly*1.e-3)
plt.xlabel('Pressure (GPa)')
plt.ylabel('Excess Gibbs free energy (kJ/mol)')
plt.show()


# Check KV rule of thumb
garnet.set_composition([0.5, 0.5])
garnet.set_state(1.e5, 298.15)
K_T = garnet.K_T
V = garnet.V

KV_py = pyrope.params['K_0']*pyrope.params['V_0']
KV_gr = grossular.params['K_0']*grossular.params['V_0']

K_average = 0.5*(pyrope.params['K_0'] + grossular.params['K_0'])
KV_average = 0.5*(KV_py + KV_gr)
print 'Bulk modulus excess according to KV:', (KV_average/V - K_average)/1.e9, 'GPa'
print 'Bulk modulus excess is actually', (K_T - K_average)/1.e9, 'GPa'


print (K_T - K_average)/(KV_average/V - K_average)



print pyrope.params
print grossular.params
print intermediate0.params
print intermediate1.params
